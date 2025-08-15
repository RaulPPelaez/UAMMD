#include "FastChebyshevTransform.cuh"
#include "global/defines.h"
#include "misc/ChevyshevUtils.cuh"
#include "misc/IBM.cuh"
#include "misc/IBM_kernels.cuh"
#include "utils.cuh"
#include <memory>

namespace uammd {
namespace DPStokesSlab_ns {

struct Gaussian {
  int3 support;
  Gaussian(real tolerance, real width, real h, real H, int supportxy, int nz,
           bool torqueMode = false)
      : H(H), nz(nz) {
    this->prefactor = cbrt(pow(2 * M_PI * width * width, -1.5));
    this->tau = -1.0 / (2.0 * width * width);
    rmax = supportxy * h * 0.5;
    support = {supportxy, supportxy, supportxy};
    int ct = int(nz * (acos(-2 * (-H * 0.5 + rmax) / H) / M_PI));
    support.z = 2 * ct + 1;
  }

  inline __host__ __device__ int3 getMaxSupport() const { return support; }

  inline __host__ __device__ int3 getSupport(int3 cell) const {
    real ch = real(-0.5) * H * cospi((real(cell.z)) / (nz - 1));
    real zmax = thrust::min(ch + rmax, H * real(0.5));
    int czt = int((nz) * (acos(real(-2.0) * (zmax) / H) / real(M_PI)));
    real zmin = thrust::max(ch - rmax, -H * real(0.5));
    int czb = int((nz) * (acos(real(-2.0) * (zmin) / H) / real(M_PI)));
    int sz = 2 * thrust::max(czt - cell.z, cell.z - czb) + 1;
    return make_int3(support.x, support.y, thrust::min(sz, support.z));
  }

  inline __host__ __device__ real phiX(real r) const {
    return prefactor * exp(tau * r * r);
  }

  inline __host__ __device__ real phiY(real r) const {
    return prefactor * exp(tau * r * r);
  }
  // For this algorithm we spread a particle and its image to enforce the force
  // density outside the slab is zero. A particle on the wall position will
  // spread zero force. phi(r) = phi(r) - phi(r_img);
  inline __host__ __device__ real phiZ(real r, real3 pi) const {
    if (fabs(r) >= rmax) {
      return 0;
    } else {
      real top_rimg = H - 2 * pi.z + r;
      real bot_rimg = -H - 2 * pi.z + r;
      real rimg = thrust::min(fabs(top_rimg), fabs(bot_rimg));
      real phi_img =
          rimg >= rmax ? real(0.0) : prefactor * exp(tau * rimg * rimg);
      real phi = prefactor * exp(tau * r * r);
      return phi - phi_img;
    }
  }

private:
  real prefactor;
  real tau;
  real H;
  real rmax;
  int nz;
};

//[1] Taken from https://arxiv.org/pdf/1712.04732.pdf
struct BarnettMagland {
  IBM_kernels::BarnettMagland bm;
  real ax;
  real ay;
  real az;
  int3 support;

  BarnettMagland(real w, real beta, real i_alpha, real hx, real hy, real H,
                 int nz)
      : H(H), nz(nz), bm(i_alpha, beta) {
    int supportxy = w + 0.5;
    real h_max = thrust::max(hx, hy);
    this->rmax = w * h_max * 0.5;
    support.x = support.y = supportxy;
    int ct = ceil((nz - 1) * (acos((H * 0.5 - rmax) / (0.5 * H)) / M_PI));
    support.z = 2 * ct + 1;
    this->ax = hx;
    this->ay = hy;
    this->az = thrust::min(hx, hy);
    System::log<System::MESSAGE>("BM kernel: beta: %g, alpha: %g, w: %g", beta,
                                 i_alpha, w);
  }

  inline __host__ __device__ int3 getMaxSupport() const { return support; }

  inline __host__ __device__ int3 getSupport(real3 pos, int3 cell) const {
    real bound = H * real(0.5);
    real ztop = thrust::min(pos.z + rmax, bound);
    real zbot = thrust::max(pos.z - rmax, -bound);
    int czb = int((nz - 1) * (acos(ztop / bound) / real(M_PI)));
    int czt = int((nz - 1) * (acos(zbot / bound) / real(M_PI)));
    int sz = 2 * thrust::max(cell.z - czb, czt - cell.z) + 1;
    return make_int3(support.x, support.y, sz);
  }

  inline __host__ __device__ real phiX(real r, real3 pi) const {
    return bm.phi(r / ax) / ax;
  }

  inline __host__ __device__ real phiY(real r, real3 pi) const {
    return bm.phi(r / ay) / ay;
  }
  // For this algorithm we spread a particle and its image to enforce the force
  // density outside the slab is zero. A particle on the wall position will
  // spread zero force. phi(r) = phi(r) - phi(r_img);
  inline __host__ __device__ real phiZ(real r, real3 pi) const {
    real top_rimg = H - real(2.0) * pi.z + r;
    real bot_rimg = -H - real(2.0) * pi.z + r;
    real rimg = thrust::min(fabs(top_rimg), fabs(bot_rimg));
    real phi_img = bm.phi(rimg / az) / az;
    real phi = bm.phi(r / az) / az;
    return phi - phi_img;
  }

private:
  real H;
  real rmax;
  int nz;
};

namespace detail {
// Computes the coefficients of the derivative of "f" in Cheb space.
// Stores the result in Df, which can be aliased to f
template <class Iter>
__device__ void chebyshevDerivate(const Iter f, Iter Df, int nz, real halfH) {
  complex fip1 = complex();
  complex Dpnp2 = complex();
  complex Dpnp1 = complex();
  for (int i = nz - 1; i >= 0; i--) {
    complex Dpni = complex();
    if (i <= nz - 2)
      Dpni = Dpnp2 + real(2.0) * (i + 1) * fip1 / halfH;
    if (i == 0)
      Dpni *= real(0.5);
    fip1 = f[i];
    Df[i] = Dpni;
    Dpnp2 = Dpnp1;
    if (i <= nz - 2) {
      Dpnp1 = Dpni;
    }
  }
}

// Compute the curl of the input in chebyshev space (i.e torque or velocity)
//  0.5\nabla \times T = 0.5 (i*k_x i*k_y \partial_z)\times (T_x T_y T_z) =
//  = 0.5( i*k_y*T_z - \partial_z(T_y), \partial_z(T_x) - i*k_x*T_z, i*k_x*T_y -
//  i*k_y*T_x)
// Add to the output vector
// The input vector is overwritten
__global__ void addCurlCheb(DataXYZPtr<complex> input, DataXYZPtr<complex> curl,
                            real3 L, int3 n) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  const int2 ik = make_int2(id % (n.x / 2 + 1), id / (n.x / 2 + 1));
  if (id >= n.y * (n.x / 2 + 1)) {
    return;
  }
  const auto kn = computeWaveNumber(id, n.x, n.y);
  const auto k = computeWaveVector(kn, make_real2(L));
  auto zToIndex =
      make_third_index_iterator(thrust::make_counting_iterator(0), ik.x, ik.y,
                                Index3D(n.x / 2 + 1, n.y, 2 * n.z - 2));
  // First sum the terms that do not depend on the derivative in Z
  const real half = real(0.5);
  const bool isUnpairedX = ik.x == (n.x - ik.x);
  const bool isUnpairedY = ik.y == (n.y - ik.y);
  real Dx = isUnpairedX ? 0 : k.x;
  real Dy = isUnpairedY ? 0 : k.y;
  fori(0, n.z) {
    int index = zToIndex[i];
    const auto T = input.xyz()[index];
    curl.x()[index] += {-half * Dy * T.z.y, half * Dy * T.z.x};
    curl.y()[index] += {half * Dx * T.z.y, -half * Dx * T.z.x};
    curl.z()[index] +=
        {half * (-Dx * T.y.y + Dy * T.x.y), half * (Dx * T.y.x - Dy * T.x.x)};
  }
  auto Tx = make_third_index_iterator(input.x(), ik.x, ik.y,
                                      Index3D(n.x / 2 + 1, n.y, 2 * n.z - 2));
  auto Ty = make_third_index_iterator(input.y(), ik.x, ik.y,
                                      Index3D(n.x / 2 + 1, n.y, 2 * n.z - 2));
  // Overwrite input torque with Z derivatives
  chebyshevDerivate(Tx, Tx, n.z, real(0.5) * L.z);
  chebyshevDerivate(Ty, Ty, n.z, real(0.5) * L.z);
  // Sum the rest of the terms
  fori(0, n.z) {
    int index = zToIndex[i];
    const auto DzTx = Tx[i];
    const auto DzTy = Ty[i];
    curl.x()[index] += -half * DzTy;
    curl.y()[index] += half * DzTx;
  }
}
} // namespace detail

class SpreadInterp {
  // using Kernel = Gaussian;
  using Kernel = BarnettMagland;
  // A different kernel can be used for spreading forces and torques
  using KernelTorque = BarnettMagland;
  using QuadratureWeights = chebyshev::doublyperiodic::QuadratureWeights;
  using Grid = chebyshev::doublyperiodic::Grid;
  shared_ptr<Kernel> kernel;
  shared_ptr<KernelTorque> kernelTorque;
  Grid grid;
  shared_ptr<QuadratureWeights> qw;

public:
  struct Parameters {
    real w, w_d;
    real beta = -1;
    real beta_d = -1;
    real alpha = -1;
    real alpha_d = -1;
  };

  SpreadInterp(Grid grid, Parameters par) : grid(grid) {
    initializeKernel(par);
    initializeQuadratureWeights();
    System::log<System::MESSAGE>("[DPStokes] support: %d", kernel->support.x);
  }

  template <class PosIterator, class ForceIterator>
  auto spreadForces(PosIterator pos, ForceIterator forces, int numberParticles,
                    cudaStream_t st) {
    System::log<System::DEBUG2>("[DPStokes] Spreading forces");
    const int3 n = grid.cellDim;
    DataXYZ<real> particleForces(forces, numberParticles);
    DataXYZ<real> gridForce(2 * (n.x / 2 + 1) * n.y * (2 * n.z - 2));
    gridForce.fillWithZero();
    IBM<Kernel, Grid> ibm(kernel, grid,
                          IBM_ns::LinearIndex3D(2 * (n.x / 2 + 1), n.y, n.z));
    ibm.spread(pos, particleForces.x(), gridForce.x(), numberParticles, st);
    ibm.spread(pos, particleForces.y(), gridForce.y(), numberParticles, st);
    ibm.spread(pos, particleForces.z(), gridForce.z(), numberParticles, st);
    CudaCheckError();
    return gridForce;
  }

  // Spread the curl of the torques to the grid and add it to the fluid forcing
  // in Cheb space
  template <class PosIterator, class TorqueIterator>
  void addSpreadTorquesFourier(PosIterator pos, TorqueIterator torques,
                               int numberParticles,
                               DataXYZ<complex> &gridForceCheb,
                               std::shared_ptr<FastChebyshevTransform> fct,
                               cudaStream_t st) {
    if (torques == nullptr)
      return;
    System::log<System::DEBUG2>("[DPStokes] Spreading torques");
    const int3 n = grid.cellDim;
    DataXYZ<real> particleTorques(torques, numberParticles);
    DataXYZ<real> gridTorque(2 * (n.x / 2 + 1) * n.y * (2 * n.z - 2));
    gridTorque.fillWithZero();
    IBM<KernelTorque, Grid> ibm(
        kernelTorque, grid, IBM_ns::LinearIndex3D(2 * (n.x / 2 + 1), n.y, n.z));
    ibm.spread(pos, particleTorques.x(), gridTorque.x(), numberParticles, st);
    ibm.spread(pos, particleTorques.y(), gridTorque.y(), numberParticles, st);
    ibm.spread(pos, particleTorques.z(), gridTorque.z(), numberParticles, st);
    auto gridTorqueCheb = fct->forwardTransform(gridTorque, st);
    const int blockSize = 128;
    const int numberSystems = n.y * (n.x / 2 + 1);
    const int numberBlocks = numberSystems / blockSize + 1;
    detail::addCurlCheb<<<numberBlocks, blockSize, 0, st>>>(
        DataXYZPtr<complex>(gridTorqueCheb), DataXYZPtr<complex>(gridForceCheb),
        grid.box.boxSize, n);
    CudaCheckError();
  }

  template <class PosIterator>
  auto interpolateVelocity(DataXYZ<real> &gridData, PosIterator pos,
                           int numberParticles, cudaStream_t st) {
    System::log<System::DEBUG2>("[DPStokes] Interpolating forces and energies");
    DataXYZ<real> particleVels(numberParticles);
    particleVels.fillWithZero();
    const int3 n = grid.cellDim;
    IBM<Kernel, Grid> ibm(kernel, grid,
                          IBM_ns::LinearIndex3D(2 * (n.x / 2 + 1), n.y, n.z));
    ibm.gather(pos, particleVels.x(), gridData.x(), *qw,
               IBM_ns::DefaultWeightCompute(), numberParticles, st);
    ibm.gather(pos, particleVels.y(), gridData.y(), *qw,
               IBM_ns::DefaultWeightCompute(), numberParticles, st);
    ibm.gather(pos, particleVels.z(), gridData.z(), *qw,
               IBM_ns::DefaultWeightCompute(), numberParticles, st);
    CudaCheckError();
    return toReal3Vector(particleVels);
  }

  auto computeGridAngularVelocityCheb(FluidData<complex> &fluid,
                                      cudaStream_t st) {
    System::log<System::DEBUG2>(
        "[DPStokes] Computing angular velocities as curl of velocity");
    const int3 n = grid.cellDim;
    const int blockSize = 128;
    const int numberSystems = n.y * (n.x / 2 + 1);
    const int numberBlocks = numberSystems / blockSize + 1;
    // The kernel overwrites the input vector, thus the copy
    auto gridVelsChebCopy = fluid.velocity;
    DataXYZ<complex> gridAngVelsCheb(gridVelsChebCopy.size());
    gridAngVelsCheb.fillWithZero();
    detail::addCurlCheb<<<numberBlocks, blockSize, 0, st>>>(
        DataXYZPtr<complex>(gridVelsChebCopy),
        DataXYZPtr<complex>(gridAngVelsCheb), grid.box.boxSize, n);
    CudaCheckError();
    return gridAngVelsCheb;
  }

  template <class PosIterator>
  auto interpolateAngularVelocity(DataXYZ<real> &gridAngVel, PosIterator pos,
                                  int numberParticles, cudaStream_t st) {
    System::log<System::DEBUG2>("[DPStokes] Interpolating angular velocities");
    const int3 n = grid.cellDim;
    DataXYZ<real> particleAngVels(numberParticles);
    particleAngVels.fillWithZero();
    IBM<KernelTorque, Grid> ibm(
        kernelTorque, grid,
        IBM_ns::LinearIndex3D(2 * (n.x / 2 + 1), n.y, 2 * n.z - 2));
    ibm.gather(pos, particleAngVels.x(), gridAngVel.x(), *qw,
               IBM_ns::DefaultWeightCompute(), numberParticles, st);
    ibm.gather(pos, particleAngVels.y(), gridAngVel.y(), *qw,
               IBM_ns::DefaultWeightCompute(), numberParticles, st);
    ibm.gather(pos, particleAngVels.z(), gridAngVel.z(), *qw,
               IBM_ns::DefaultWeightCompute(), numberParticles, st);
    CudaCheckError();
    return toReal3Vector(particleAngVels);
  }

private:
  void initializeKernel(Parameters par) {
    System::log<System::DEBUG>("[DPStokes] Initialize kernel");
    // double h = grid.cellSize.x;
    //  if(supportxy >= grid.cellDim.x){
    //  	System::log<System::WARNING>("[DPStokes] Support is too big,
    //  cell dims: %d %d %d, requested support: %d",
    //  grid.cellDim.x, grid.cellDim.y, grid.cellDim.z, supportxy);
    // }
    real H = grid.box.boxSize.z;
    this->kernel =
        std::make_shared<Kernel>(par.w, par.beta, par.alpha, grid.cellSize.x,
                                 grid.cellSize.y, H, grid.cellDim.z);
    this->kernelTorque = std::make_shared<KernelTorque>(
        par.w_d, par.beta_d, par.alpha_d, grid.cellSize.x, grid.cellSize.y, H,
        grid.cellDim.z);
  }

  void initializeQuadratureWeights() {
    System::log<System::DEBUG>("[DPStokes] Initialize quadrature weights");
    real hx = grid.cellSize.x;
    real hy = grid.cellSize.y;
    int nz = grid.cellDim.z;
    real H = grid.box.boxSize.z;
    qw = std::make_shared<QuadratureWeights>(H, hx, hy, nz);
  }
};
} // namespace DPStokesSlab_ns
} // namespace uammd
