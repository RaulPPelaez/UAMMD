/*Raul P. Pelaez 2018-2020. Immerse boundary kernels. AKA Window functions

A Kernel has the following requirements if used with the IBM module:

  -A publicly accesible member called support (either an int or int3) or a
function getSupport(int3 cell) if support depends on grid position
  -A function phi(real r) that returns the window function evaluated at that
distance


Notice that in order to use the Kernels in this file you must somehow inherit
from them to provide the support. These are meant to be a skeleton on top of
which the kernel is created for each particular use case.

REFERENCES:
[1] Charles S. Peskin. The immersed boundary method (2002).
DOI: 10.1017/S0962492902000077 [2] Fluctuating force-coupling method for
simulations of colloidal suspensions. Eric E. Keaveny. 2014.
 */
#ifndef IBMKERNELS_CUH
#define IBMKERNELS_CUH
#include "global/defines.h"
#include "misc/TabulatedFunction.cuh"
#include "uammd.cuh"
namespace uammd {
namespace IBM_kernels {

class Gaussian {
  const real prefactor;
  const real tau;

public:
  Gaussian(real width)
      : prefactor(pow(2.0 * M_PI * width * width, -0.5)),
        tau(-0.5 / (width * width)) {}

  __host__ __device__ real phi(real r, real3 pos = real3()) const {
    return prefactor * exp(tau * r * r);
  }
};

namespace detail {

// Sum all values in a container using Kahan Summation, which increases floating
// point accuracy
template <class Container> auto kahanSum(Container &v) {
  auto c = v[0] * 0;
  auto sum = c;
  for (auto f : v) {
    auto y = f - c;
    auto t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

// Integrate the function foo(x) from x=rmin to x=rmax using the Simpson rule
// with Nr intervals
template <class Foo> auto integrate(Foo foo, real rmin, real rmax, int Nr) {
  using T = decltype(foo(rmin));
  if (Nr % 2 == 1)
    Nr++; // Need an even number of points
  std::vector<T> integral_vals(Nr + 1);
  real dx = (rmax - rmin) / Nr;
  for (int i = 0; i <= Nr; i++) {
    real weight;
    if (i == 0 or i == Nr)
      weight = 1;
    else if (i % 2 == 1)
      weight = 4;
    else
      weight = 2;
    integral_vals[i] = weight * foo(rmin + i * dx);
  }
  auto integral = dx / 3.0 * kahanSum(integral_vals);
  return integral;
}

} // namespace detail

//[1] Taken from https://arxiv.org/pdf/1712.04732.pdf
__host__ __device__ real BM(real zz, real alpha, real beta) {
  const real z = zz / alpha;
  const real z2 = z * z;
  const real dz2 = real(1.0) - z2;
  auto kern = (dz2 < real(0.0)) ? 0 : (exp(beta * (sqrt(dz2) - real(1.0))));
  return kern;
}

struct BarnettMagland {
private:
  real computeNorm() const {
    auto foo = [this](real r) { return BM(r, alpha, beta); };
    real norm = 2.0 * detail::integrate(foo, 0, alpha, 20000);
    return norm;
  }

  real invnorm;

public:
  const real beta;
  real alpha;
  // Alpha is half the support ( phi(r>alpha) = 0)
  // Beta is the beta parameter of the ES kernel.
  BarnettMagland(real i_alpha, real i_beta) : alpha(i_alpha), beta(i_beta) {
    this->invnorm = 1.0 / computeNorm();
  }

  inline __host__ __device__ real phi(real zz) const {
    return BM(zz, alpha, beta) * invnorm;
  }
};

namespace Peskin {
//[1] Charles S. Peskin. The immersed boundary method (2002).
// DOI: 10.1017/S0962492902000077 Standard 3-point Peskin interpolator
struct threePoint {
  const real invh;
  static constexpr int support = 3;
  threePoint(real h) : invh(1.0 / h) {}
  __host__ __device__ real phi(real rr, real3 pos = real3()) const {
    const real r = fabs(rr) * invh;
    if (r < real(0.5)) {
      constexpr real onediv3 = real(1 / 3.0);
      return invh * onediv3 *
             (real(1.0) + sqrt(real(1.0) + real(-3.0) * r * r));
    } else if (r < real(1.5)) {
      constexpr real onediv6 = real(1 / 6.0);
      const real omr = real(1.0) - r;
      return invh * onediv6 *
             (real(5.0) - real(3.0) * r -
              sqrt(real(1.0) + real(-3.0) * omr * omr));
    } else
      return 0;
  }
};

// Standard 4-point Peskin interpolator
struct fourPoint {
  const real invh;
  static constexpr int support = 4;
  fourPoint(real h) : invh(1.0 / h) {}

  __host__ __device__ real phi(real rr, real3 pos = real3()) const {
    const real r = fabs(rr) * invh;
    constexpr real onediv8 = real(0.125);
    if (r < real(1.0)) {
      return invh * onediv8 *
             (real(3.0) - real(2.0) * r +
              sqrt(real(1.0) + real(4.0) * r * (real(1.0) - r)));
    } else if (r < real(2.0)) {
      return invh * onediv8 *
             (real(5.0) - real(2.0) * r -
              sqrt(real(-7.0) + real(12.0) * r - real(4.0) * r * r));
    } else
      return 0;
  }
};

} // namespace Peskin

namespace GaussianFlexible {
//[1] Yuanxun Bao, Jason Kaye and Charles S. Peskin. A Gaussian-like
// immersed-boundary kernel with three continuous derivatives and improved
// translational invariance. http://dx.doi.org/10.1016/j.jcp.2016.04.024 Adapted
// from https://github.com/stochasticHydroTools/IBMethod/
struct sixPoint {
private:
  static constexpr real K = 0.714075092976608; // 59.0/60.0-sqrt(29.0)/20.0;
  const TabulatedFunction<real> phi_tab;
  const real invh;
  // the new C3 6-pt kernel
  static inline __host__ __device__ real phi_impl(real r) {
    // if (r <= real(-3) || r>=real(3)) return 0;
    if (r >= real(3))
      return 0;
    real R = r - ceil(r) + real(1.0); // R between [0,1]
    real R2 = R * R;
    real R3 = R2 * R;
    const real alpha = real(28.);
    const real beta = real(9.0 / 4.0) - real(1.5) * (K + R2) +
                      (real(22. / 3) - real(7.0) * K) * R - real(7. / 3.) * R3;
    real gamma =
        real(0.25) *
        (real(0.5) *
             (real(161.) / real(36) - real(59.) / real(6) * K +
              real(5) * K * K) *
             R2 +
         real(1.) / real(3) * (real(-109.) / real(24) + real(5) * K) * R2 * R2 +
         real(5.) / real(18) * R3 * R3);

    const real discr = beta * beta - real(4.0) * alpha * gamma;

    const int sgn = ((real(1.5) - K) > 0) ? 1 : -1; // sign(3/2 - K)
    const real prefactor =
        real(1.) / (real(2) * alpha) * (-beta + sgn * sqrt(discr));
    if (r <= real(0)) {
      const real rp1 = r + real(1.0);
      return real(2.) * prefactor + real(0.25) +
             real(1. / 6) * (real(4) - real(3) * K) * rp1 -
             real(1. / 6) * rp1 * rp1 * rp1;
    } else if (r <= real(1)) {
      return real(2.0) * prefactor + real(5. / 8) - real(0.25) * (K + r * r);
    } else if (r <= real(2)) {
      const real rm1 = r + real(-1.0);
      return real(-3.0) * prefactor + real(0.25) -
             real(1. / 6.) * (real(4) - real(3) * K) * rm1 +
             real(1. / 6) * rm1 * rm1 * rm1;
    } else if (r <= real(3)) {
      const real rm2 = r + real(-2.0);
      return prefactor - real(1. / 16) + real(1. / 8) * (K + rm2 * rm2) -
             real(1. / 12) * (real(3) * K - real(1)) * rm2 -
             real(1. / 12) * rm2 * rm2 * rm2;
    }
    return real(0.0);
  }

public:
  sixPoint(real h, real tolerance = 1e-7)
      : invh(1.0 / h),
        phi_tab(int(1e5 * (-log10(tolerance) / 20.0)), 0, 3, phi_impl) {}

  ~sixPoint() = default;

  inline __host__ __device__ real phi(real r, real3 pos = real3()) const {
    return phi_impl(fabs(r) * invh) * invh;
  }

  inline __host__ __device__ real phi_tabulated(real r) const {
#ifdef __CUDA_ARCH__
    return phi_tab(fabs(r) * invh) * invh;
#else
    return phi(r * invh) * invh;
#endif
  }
};
} // namespace GaussianFlexible

} // namespace IBM_kernels
} // namespace uammd

#endif
