#include "misc/IBM.cuh"
#include "misc/IBM_kernels.cuh"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include<numeric>
using namespace uammd;
struct ConstantKernel {
  int3 support;
  inline __host__ __device__ int phi(real r, real3 pos) const { return 1; }
};

TEST(Spreading, ConstantKernelCenter) {
  thrust::device_vector<real3> pos(1);
  thrust::device_vector<int> values(1, 1);
  int3 n = {3, 3, 3};
  real3 L = make_real3(1, 1, 1);
  thrust::device_vector<int> field(n.x * n.y * n.z, 0);
  pos[0] = make_real3(0, 0, 0);
  auto kernel = std::make_shared<ConstantKernel>(ConstantKernel{n});
  Grid grid(Box(L), n);
  IBM<ConstantKernel, Grid> ibm(kernel, grid);
  ibm.spread(pos.begin(), values.begin(), field.data().get(), pos.size());
  thrust::host_vector<int> h_field = field;
  auto sum = std::accumulate(h_field.begin(), h_field.end(), 0);
  EXPECT_EQ(sum, n.x * n.y * n.z);
}

TEST(Spreading, ConstantKernelCornerPeriodic) {
  thrust::device_vector<real3> pos(1);
  thrust::device_vector<int> values(1, 1);
  int3 n = {3, 3, 3};
  real3 L = make_real3(3, 3, 3);
  thrust::device_vector<int> field(n.x * n.y * n.z, 0);
  pos[0] = make_real3(-1, -1, -1);
  auto kernel = std::make_shared<ConstantKernel>(ConstantKernel{n});
  Grid grid(Box(L), n);
  IBM<ConstantKernel, Grid> ibm(kernel, grid);
  ibm.spread(pos.begin(), values.begin(), field.data().get(), pos.size());
  thrust::host_vector<int> h_field = field;
  auto sum = std::accumulate(h_field.begin(), h_field.end(), 0);
  EXPECT_EQ(sum, n.x * n.y * n.z);
}

TEST(Spreading, ConstantKernelCornerNonPeriodic) {
  thrust::device_vector<real3> pos(1);
  thrust::device_vector<int> values(1, 1);
  int3 n = {3, 3, 3};
  real3 L = make_real3(3, 3, 3);
  thrust::device_vector<int> field(n.x * n.y * n.z, 0);
  pos[0] = make_real3(-1, -1, -1);
  auto kernel = std::make_shared<ConstantKernel>(ConstantKernel{n});
  Box box(L);
  box.setPeriodicity(false, false, false);
  Grid grid(box, n);
  IBM<ConstantKernel, Grid> ibm(kernel, grid);
  ibm.spread(pos.begin(), values.begin(), field.data().get(), pos.size());
  thrust::host_vector<int> h_field = field;
  auto sum = std::accumulate(h_field.begin(), h_field.end(), 0);
  EXPECT_EQ(sum, 8);
}

using PeskinBase = IBM_kernels::Peskin::threePoint;
struct Peskin3pt {
  static constexpr int support = 3;
  Peskin3pt(real3 h) : m_phiX(h.x), m_phiY(h.y), m_phiZ(h.z) {}

  __host__ __device__ real phiX(real rr, real3 pos = real3()) const {
    return m_phiX.phi(rr, pos);
  }

  __host__ __device__ real phiY(real rr, real3 pos = real3()) const {
    return m_phiY.phi(rr, pos);
  }

  __host__ __device__ real phiZ(real rr, real3 pos = real3()) const {
    return m_phiZ.phi(rr, pos);
  }

private:
  PeskinBase m_phiX, m_phiY, m_phiZ;
};

TEST(Spreading, PeskinKernelSpread) {
  int n_val = 8;
  int3 n = make_int3(n_val);
  real3 L = make_real3(16.0);
  thrust::host_vector<real3> h_pos(1);
  h_pos[0] = make_real3(0.0, 0.0, 0.0);
  thrust::device_vector<real3> pos = h_pos;
  thrust::host_vector<real> h_quantity(1, 1.0);
  thrust::device_vector<real> quantity = h_quantity;
  int fieldSize = n.x * n.y * n.z;
  thrust::device_vector<real> field(fieldSize, 0.0);
  Box box(L);
  Grid grid(box, n);
  real3 h = L / make_real3(n);
  auto kernel = std::make_shared<Peskin3pt>(h);
  IBM<Peskin3pt, Grid> ibm(kernel, grid);
  ibm.spread(pos.data().get(), quantity.data().get(), field.data().get(),
             pos.size());

  std::vector<real> expected(fieldSize, 0.0);
  for (int iz = 0; iz < n.z; ++iz) {
    real z = -L.z / 2.0 + (iz + 0.5) * h.z + h_pos[0].z;
    for (int iy = 0; iy < n.y; ++iy) {
      real y = -L.y / 2.0 + (iy + 0.5) * h.y + h_pos[0].y;
      for (int ix = 0; ix < n.x; ++ix) {
        real x = -L.x / 2.0 + (ix + 0.5) * h.x + h_pos[0].x;
	real3 p = h_pos[0];
        real weight = kernel->phiX(x - p.x) *
                      kernel->phiY(y - p.y) *
                      kernel->phiZ(z - p.z);
        int idx = ix + iy * n.x + iz * n.x * n.y;
        expected[idx] = weight*h_quantity[0];
      }
    }
  }

  // Copy device field to host and compare.
  thrust::host_vector<real> h_field = field;
  for (int i = 0; i < fieldSize; ++i) {
    EXPECT_NEAR(h_field[i], expected[i], 1e-4);
  }
}

// //---------------------------------------------------------------------
// // Helper: numerically integrate the square of the Peskin kernel in 1D.
// real integratePeskinSquare(const PeskinKernel &kernel, real h) {
//   int N = 10000;
//   real a = -1.5 * h;
//   real b = 1.5 * h;
//   real dx = (b - a) / (N - 1);
//   real sum = 0.0;
//   for (int i = 0; i < N; ++i) {
//     real x = a + i * dx;
//     real val = kernel.phi(x, h);
//     sum += val * val;
//   }
//   return sum * dx;
// }

// //---------------------------------------------------------------------
// // Test 2: Verify the adjoint property of spread and interpolate in 3D.
// // A unit quantity is spread then interpolated; after dividing by the
// // kernel-squared integral dV, we should recover 1.
// TEST(Spreading, PeskinKernelAdjoint3D) {
//   int numberParticles = 1;
//   int n_val = 64;
//   int n[3] = {n_val, n_val, n_val};
//   real L_val = 16.0;
//   real L_arr[3] = {L_val, L_val, L_val};

//   // Particle at the center.
//   thrust::host_vector<real3> h_pos(1);
//   h_pos[0] = make_real3(0.0, 0.0, 0.0);
//   thrust::device_vector<real3> pos = h_pos;
//   thrust::host_vector<real> h_quantity(1, 1.0);
//   thrust::device_vector<real> quantity = h_quantity;

//   int fieldSize = n[0] * n[1] * n[2];
//   thrust::device_vector<real> field(fieldSize, 0.0);

//   int3 grid_n = {n[0], n[1], n[2]};
//   real3 grid_L = make_real3(L_arr[0], L_arr[1], L_arr[2]);
//   Box box(make_real3(L_arr[0], L_arr[1], L_arr[2]));
//   Grid grid(box, grid_n);

//   PeskinKernel peskinKernel;
//   peskinKernel.support = {3, 3, 3};

//   IBM<PeskinKernel, Grid> ibm(peskinKernel, grid);
//   ibm.spread(pos.begin(), quantity.begin(), field.data().get(), pos.size());

//   // Compute grid spacings.
//   real h_x = L_arr[0] / n[0];
//   real h_y = L_arr[1] / n[1];
//   real h_z = L_arr[2] / n[2];

//   // Compute the 1D integrals (dV factors) and then the overall dV.
//   real dV = integratePeskinSquare(peskinKernel, h_x) *
//             integratePeskinSquare(peskinKernel, h_y) *
//             integratePeskinSquare(peskinKernel, h_z);

//   // Now interpolate from the field back to the particle position.
//   thrust::device_vector<real> interp_result(1, 0.0);
//   ibm.interpolate(pos.begin(), field.data().get(), interp_result.begin(),
//                   pos.size());

//   thrust::host_vector<real> h_interp = interp_result;
//   real quantity_reconstructed = h_interp[0] / dV;

//   EXPECT_NEAR(quantity_reconstructed, 1.0, 1e-4);
// }

// //---------------------------------------------------------------------
// // Test 3: Verify that interpolating a field of ones returns ones.
// // This test works for 3D or 2D; here we demonstrate a 3D case with
// // multiple particles and multiple quantities per particle.
// TEST(Spreading, PeskinKernelInterpolation) {
//   int n_val = 8;
//   int numberParticles = 10;
//   int n[3] = {n_val, n_val, n_val};
//   real L_arr[3] = {16.0, 16.0, 16.0};
//   int nquantities = 3;

//   // Generate random positions (ensuring they lie inside the domain).
//   std::vector<real3> h_pos(numberParticles);
//   std::mt19937 gen(123);
//   std::uniform_real_distribution<real> dist(-L_arr[0] / 2.0 + 1.0,
//                                             L_arr[0] / 2.0 - 1.0);
//   for (int i = 0; i < numberParticles; ++i) {
//     h_pos[i] = make_real3(dist(gen), dist(gen), dist(gen));
//   }
//   thrust::device_vector<real3> pos = h_pos;

//   // Create a field of ones.
//   int fieldSize = n[0] * n[1] * n[2] * nquantities;
//   thrust::device_vector<real> field(fieldSize, 1.0);

//   int3 grid_n = {n[0], n[1], n[2]};
//   real3 grid_L = make_real3(L_arr[0], L_arr[1], L_arr[2]);
//   Box box(make_real3(L_arr[0], L_arr[1], L_arr[2]));
//   Grid grid(box, grid_n);

//   PeskinKernel peskinKernel;
//   peskinKernel.support = {3, 3, 3};

//   IBM<PeskinKernel, Grid> ibm(peskinKernel, grid);

//   // Interpolate: the result is expected to be a vector of size
//   (numberParticles
//   // * nquantities)
//   thrust::device_vector<real> interp_result(numberParticles * nquantities,
//   0.0); ibm.interpolate(pos.begin(), field.data().get(),
//   interp_result.begin(),
//                   pos.size());

//   thrust::host_vector<real> h_interp = interp_result;
//   for (int i = 0; i < numberParticles * nquantities; ++i) {
//     EXPECT_NEAR(h_interp[i], 1.0, 1e-4);
//   }
// }

// //---------------------------------------------------------------------
// // You may also add a 2D adjoint test by setting the z-component to 0,
// // L[2]=0 and n[2]=1. For example:
// TEST(Spreading, PeskinKernelAdjoint2D) {
//   int numberParticles = 1;
//   int n_val = 64;
//   int n[3] = {n_val, n_val, 1};
//   real L_arr[3] = {16.0, 16.0, 0.0};

//   // In 2D the particle lies in the x-y plane.
//   thrust::host_vector<real3> h_pos(1);
//   h_pos[0] = make_real3(0.0, 0.0, 0.0);
//   thrust::device_vector<real3> pos = h_pos;
//   thrust::host_vector<real> h_quantity(1, 1.0);
//   thrust::device_vector<real> quantity = h_quantity;

//   int fieldSize = n[0] * n[1] * n[2];
//   thrust::device_vector<real> field(fieldSize, 0.0);

//   int3 grid_n = {n[0], n[1], n[2]};
//   // For a 2D domain, we set L[2] = 0 but provide a dummy value (e.g. 1) for
//   the
//   // third dimension.
//   real3 grid_L = make_real3(L_arr[0], L_arr[1], 1.0);
//   Box box(make_real3(L_arr[0], L_arr[1], 1.0));
//   Grid grid(box, grid_n);

//   PeskinKernel peskinKernel;
//   // In 2D we use support {3,3,1} (only one grid cell in z).
//   peskinKernel.support = {3, 3, 1};

//   IBM<PeskinKernel, Grid> ibm(peskinKernel, grid);
//   ibm.spread(pos.begin(), quantity.begin(), field.data().get(), pos.size());

//   real h_x = L_arr[0] / n[0];
//   real h_y = L_arr[1] / n[1];

//   real dV = integratePeskinSquare(peskinKernel, h_x) *
//             integratePeskinSquare(peskinKernel, h_y);

//   thrust::device_vector<real> interp_result(1, 0.0);
//   ibm.interpolate(pos.begin(), field.data().get(), interp_result.begin(),
//                   pos.size());
//   thrust::host_vector<real> h_interp = interp_result;
//   real quantity_reconstructed = h_interp[0] / dV;

//   EXPECT_NEAR(quantity_reconstructed, 1.0, 1e-4);
// }
