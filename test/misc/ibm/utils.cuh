#pragma once

#include <uammd.cuh>
#include "misc/ChevyshevUtils.cuh"
#include "misc/IBM.cuh"
#include <thrust/complex.h>

using namespace uammd;
using complex = thrust::complex<real>;
using real = uammd::real;

struct Gaussian {
  int3 support;
  Gaussian(real width, real h, real H, real nz, int supportxy) : nz(nz) {
    this->prefactor = 1.0 / (width * sqrt(2 * M_PI));
    this->tau = -1.0 / (2.0 * width * width);
    this->rmax = supportxy * h * 0.5;
    support.x = supportxy + 1;
    support.y = support.x;
    this->Htot = H;
    real smax = std::max((H * 0.5 - rmax), -H * 0.5);
    int czmax = ceil((nz - 1) * (acos(smax / (0.5 * Htot)) / real(M_PI)));
    support.z = 2 * czmax + 1;
  }

  inline __host__ __device__ int3 getMaxSupport() const {
    return make_int3(support.x, support.y, support.z);
  }

  inline __host__ __device__ int3 getSupport(real3 pos, int3 cell) const {
    real bound = Htot * real(0.5);
    real ztop = thrust::min(pos.z + rmax, bound);
    real zbot = thrust::max(pos.z - rmax, -bound);
    int czb = int((nz - 1) * (acos(ztop / bound) / real(M_PI)));
    int czt = int((nz - 1) * (acos(zbot / bound) / real(M_PI)));
    int sz = 2 * thrust::max(cell.z - czb, czt - cell.z) + 1;
    return make_int3(support.x, support.y, sz);
  }

  inline __host__ __device__ real phiX(real r, real3 pos) const {
    real val = 0;
    if (abs(r) <= rmax * real(1.0001)) {
      val = prefactor * exp(tau * r * r);
    }
    return val;
  }

  inline __host__ __device__ real phiY(real r, real3 pos) const {
    real val = 0;
    if (abs(r) <= rmax * real(1.0001)) {
      val = prefactor * exp(tau * r * r);
    }
    return val;
  }

  __host__ __device__ real phiZ(real r, real3 pos) const {
    real val = 0;
    if (abs(r) <= rmax * real(1.0001)) {
      // val = prefactor*exp(tau*r*r);
      real bot_rimg = -this->Htot - real(2.0) * pos.z + r;
      real rimg = fabs(bot_rimg);
      real phi_img =
          rimg >= rmax ? real(0.0) : prefactor * exp(tau * rimg * rimg);
      real phi = prefactor * exp(tau * r * r);
      val = phi - phi_img;
    }
    return val;
  }

private:
  real prefactor;
  real tau;
  real rmax;
  int nz;
  real Htot;
};


// Spreads a group of particles onto a grid.
// The domain is such that some location r is r\in +-L
auto spreadParticles(thrust::device_vector<real3> &pos,
                     thrust::device_vector<complex> &values, int3 n, real sigma,
                     int supp, real3 L) {
  L.z *= 2;
  auto h = L / make_real3(n);
  auto kernel = std::make_shared<Gaussian>(sigma, h.x, L.z, n.z, supp);
  using Grid = chebyshev::doublyperiodic::Grid;
  Grid grid(Box(L), n);
  IBM<Gaussian, Grid> ibm(kernel, grid);
  auto pos_ptr = thrust::raw_pointer_cast(pos.data());
  auto values_ptr = thrust::raw_pointer_cast(values.data());
  thrust::device_vector<complex> d_fr(n.x * n.y * n.z);
  thrust::fill(d_fr.begin(), d_fr.end(), complex());
  auto fr_ptr = thrust::raw_pointer_cast(d_fr.data());
  ibm.spread(pos_ptr, (real2 *)values_ptr, (real2 *)fr_ptr, pos.size());
  std::vector<complex> fr(n.x * n.y * n.z, complex());
  thrust::copy(d_fr.begin(), d_fr.end(), fr.begin());
  return fr;
}

// Interpolates a discrete field into the locations of a group of particles.
// The domain is such that some location r is r\in +-L
auto interpolateField(thrust::device_vector<real3> &pos,
                      thrust::device_vector<complex> &field, int3 n, real sigma,
                      int supp, real3 L) {
  L.z *= 2;
  auto h = L / make_real3(n);
  auto kernel = std::make_shared<Gaussian>(sigma, h.x, L.z, n.z, supp);
  using Grid = chebyshev::doublyperiodic::Grid;
  Grid grid(Box(L), n);
  IBM<Gaussian, Grid> ibm(kernel, grid);
  thrust::device_vector<complex> d_values(pos.size());
  thrust::fill(d_values.begin(), d_values.end(), complex{});
  auto pos_ptr = thrust::raw_pointer_cast(pos.data());
  auto values_ptr = thrust::raw_pointer_cast(d_values.data());
  auto fr_ptr = thrust::raw_pointer_cast(field.data());
  chebyshev::doublyperiodic::QuadratureWeights qw(L.z, h.x, h.y, n.z);
  IBM_ns::DefaultWeightCompute wc;
  ibm.gather(pos_ptr, (real2 *)values_ptr, (real2 *)fr_ptr, qw, wc, pos.size());
  std::vector<complex> values(pos.size(), complex());
  thrust::copy(d_values.begin(), d_values.end(), values.begin());
  return values;
}
