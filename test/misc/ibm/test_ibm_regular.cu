#include "misc/IBM.cuh"
#include "misc/IBM_kernels.cuh"
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

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
  Peskin3pt(real3 h, bool is2D = false) : m_phiX(h.x), m_phiY(h.y), m_phiZ(h.z), is2D(is2D) {}

  __host__ __device__ real phiX(real rr, real3 pos = real3()) const {
    return m_phiX.phi(rr, pos);
  }

  __host__ __device__ real phiY(real rr, real3 pos = real3()) const {
    return m_phiY.phi(rr, pos);
  }

  __host__ __device__ real phiZ(real rr, real3 pos = real3()) const {
    return is2D?real(1.0):m_phiZ.phi(rr, pos);
  }

private:
  PeskinBase m_phiX, m_phiY, m_phiZ;
  bool is2D;
};

template <typename Container1, typename Container2>
auto manual_spread(Container1 &pos, Container2 &quantity, int3 n, real3 L) {
  std::vector<real> field(n.x * n.y * n.z, 0.0);
  real3 h = L / make_real3(n);
  for (int i = 0; i < pos.size(); ++i) {
    real3 p = pos[i];
    auto q = quantity[i];
    for (int iz = 0; iz < n.z; ++iz) {
      real z = -L.z / 2.0 + (iz + 0.5) * h.z;
      for (int iy = 0; iy < n.y; ++iy) {
        real y = -L.y / 2.0 + (iy + 0.5) * h.y;
        for (int ix = 0; ix < n.x; ++ix) {
          real x = -L.x / 2.0 + (ix + 0.5) * h.x;
          real weight = Peskin3pt(h).phiX(x - p.x) *
                        Peskin3pt(h).phiY(y - p.y) * Peskin3pt(h).phiZ(z - p.z);
          int idx = ix + iy * n.x + iz * n.x * n.y;
          field[idx] = weight * q;
        }
      }
    }
  }
  return field;
}

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
  thrust::host_vector<real> h_field = field;
  auto expected = manual_spread(h_pos, h_quantity, n, L);
  for (int i = 0; i < fieldSize; ++i) {
    EXPECT_NEAR(h_field[i], expected[i], 1e-10);
  }
}

template <typename Foo> real integrate(const Foo &foo, real width) {
  real a = -width / 2.0;
  real b = width / 2.0;
  int N = 10000;
  real dx = (b - a) / (N - 1);
  real sum = 0.0;
  for (int i = 0; i < N; ++i) {
    real x = a + i * dx;
    real val = foo(x);
    sum += val;
  }
  return sum * dx;
}

// Verify the adjoint property of spread and interpolate in 3D.
// A unit quantity is spread then interpolated; after dividing by the
// kernel-squared integral dV, we should recover 1.
// This should work regardless of the cell size in each direction
TEST(SpreadInterp, Adjoint3DNonRegular) {
  int numberParticles = 1;
  int3 n = make_int3(64, 32, 7);
  real3 L = make_real3(64.0);
  thrust::host_vector<real3> h_pos(1);
  h_pos[0] = make_real3(0.0, 0.0, 0.0);
  thrust::device_vector<real3> pos = h_pos;
  thrust::host_vector<real> h_quantity(1, 1.0);
  thrust::device_vector<real> quantity = h_quantity;
  thrust::device_vector<real> field(n.x * n.y * n.z, 0.0);
  Box box(L);
  Grid grid(box, n);
  real3 h = L / make_real3(n);
  auto kernel = std::make_shared<Peskin3pt>(h);
  IBM<Peskin3pt, Grid> ibm(kernel, grid);

  ibm.spread(pos.begin(), quantity.begin(), field.data().get(), pos.size());
  real dV =
      integrate([&](real r) { return pow(kernel->phiX(r), 2.0); }, 3.0 * h.x) *
      integrate([&](real r) { return pow(kernel->phiY(r), 2.0); }, 3.0 * h.y) *
      integrate([&](real r) { return pow(kernel->phiZ(r), 2.0); }, 3.0 * h.z);

  thrust::device_vector<real> interp_result(1, 0.0);
  ibm.gather(pos.data().get(), interp_result.data().get(), field.data().get(),
             int(pos.size()));

  thrust::host_vector<real> h_interp = interp_result;
  real quantity_reconstructed = h_interp[0] / dV;
  EXPECT_NEAR(quantity_reconstructed, 1.0, 1e-4);
}

// A 2D adjoint test by setting the z-component to 0,
// L[2]=0 and n[2]=1. For example:
TEST(SpreadInterp, PeskinKernelAdjoint2D) {
  int n_val = 8;
  int3 n = make_int3(n_val, n_val, 1);
  real3 L = make_real3(16.0, 16.0, 0.0);
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
  ibm.spread(pos.begin(), quantity.begin(), field.data().get(), pos.size());
  thrust::device_vector<real> interp_result(1, 0.0);
  ibm.gather(pos.data().get(), interp_result.data().get(), field.data().get(),
             int(pos.size()));
  real dV =
      integrate([&](real r) { return pow(kernel->phiX(r), 2.0); }, 3.0 * h.x) *
      integrate([&](real r) { return pow(kernel->phiY(r), 2.0); }, 3.0 * h.y);
  thrust::host_vector<real> h_interp = interp_result;
  real quantity_reconstructed = h_interp[0] / dV;
  EXPECT_NEAR(quantity_reconstructed, 1.0, 1e-4);
}

template <typename Container1, typename Container2>
auto manual_interpolate(Container1 &pos, Container2 &field, int3 n, real3 L) {
  std::vector<real> interp_result(pos.size(), 0.0);
  real3 h = L / make_real3(n);
  real qw = h.x * h.y * h.z;
  for (int i = 0; i < pos.size(); ++i) {
    real3 p = pos[i];
    for (int iz = 0; iz < n.z; ++iz) {
      real z = -L.z / 2.0 + (iz + 0.5) * h.z;
      for (int iy = 0; iy < n.y; ++iy) {
        real y = -L.y / 2.0 + (iy + 0.5) * h.y;
        for (int ix = 0; ix < n.x; ++ix) {
          real x = -L.x / 2.0 + (ix + 0.5) * h.x;
          real weight = Peskin3pt(h).phiX(x - p.x) *
                        Peskin3pt(h).phiY(y - p.y) * Peskin3pt(h).phiZ(z - p.z);
          int idx = ix + iy * n.x + iz * n.x * n.y;
          interp_result[i] += weight * field[idx] * qw;
        }
      }
    }
  }
  return interp_result;
}

TEST(Interpolation, RandomField) {
  int n_val = 32;
  int3 n = make_int3(n_val);
  real3 L = make_real3(16.0);
  real3 h = L / make_real3(n);
  int numberParticles = 128;
  std::vector<real3> h_pos(numberParticles);
  std::mt19937 gen(123);
  // Keep the particles away from the boundary to not deal with PBC in the manual_interpolation function
  std::uniform_real_distribution<real> dist(-L.x / 2.0 + 2*h.x, L.x / 2.0 - 2*h.x);
  for (int i = 0; i < numberParticles; ++i) {
    h_pos[i] = make_real3(dist(gen), dist(gen), dist(gen));
  }
  thrust::device_vector<real3> pos = h_pos;
  int fieldSize = n.x * n.y * n.z;
  thrust::host_vector<real> h_field(fieldSize, 0.0);
  for(int i = 0; i < fieldSize; ++i) {
    h_field[i] = dist(gen);
  }
  thrust::device_vector<real> field = h_field;
  Box box(L);
  Grid grid(box, n);
  auto kernel = std::make_shared<Peskin3pt>(h);
  IBM<Peskin3pt, Grid> ibm(kernel, grid);
  thrust::device_vector<real> interp_result(numberParticles, 0.0);
  ibm.gather(pos.data().get(), interp_result.data().get(), field.data().get(),
	     int(pos.size()));
  thrust::host_vector<real> h_interp = interp_result;
  auto expected = manual_interpolate(h_pos, h_field, n, L);
  for (int i = 0; i < numberParticles; ++i) {
    EXPECT_NEAR(h_interp[i], expected[i], 1e-10);
  }
}
