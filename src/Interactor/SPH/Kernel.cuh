/*Raul P. Pelaez 2017-2021. SPH Kernel implementations

  Currently available kernels:

     1- M4CubicSpline


 */
#include "Interactor/SPH.cuh"
#include "global/defines.h"

namespace uammd {
namespace SPH_ns {
namespace Kernel {
struct M4CubicSpline {
  inline __device__ __host__ real operator()(real3 r12, real h) {
    real r2 = dot(r12, r12);
    real r = sqrtf(r2);
    real q = abs(r) / h;
    if (q >= real(2.0))
      return real(0.0);
    real twomq = real(2.0) - q;
    real W = twomq * twomq * twomq;
    if (q <= real(1.0)) {
      real onemq = real(1.0) - q;
      real onemq3 = onemq * onemq * onemq;
      W -= real(4.0) * onemq3;
    }
    W *= real(1.0) / (h * h * h * real(4.0) * real(M_PI));
    return W;
  }

  inline __device__ __host__ real3 gradient(real3 r12, real h) {
    real r2 = dot(r12, r12);
    real r = sqrtf(r2);
    real invh = real(1.0) / h;
    real q = r * invh;
    if (q >= real(2.0))
      return make_real3(0.0);
    real invh3 = invh * invh * invh;
    real3 gradW = -invh3 * invh3 * real(3.0) * r12 / (real(4.0) * real(M_PI));
    if (q <= real(1.0)) {
      gradW *= (real(-4.0) * h + real(3.0) * r);
    } else if (q <= real(2.0)) {
      real f = (real(2.0) * h - r);
      gradW *= f * f;
    }
    return gradW;
  }

  static inline __device__ __host__ real getCutOff(real h) {
    return real(2.0) * h;
  }
};

} // namespace Kernel
} // namespace SPH_ns
} // namespace uammd
