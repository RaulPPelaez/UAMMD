#ifndef CUFFTCOMPLEX4_CUH
#define CUFFTCOMPLEX4_CUH

#include "global/defines.h"
#include "utils/cufftPrecisionAgnostic.h"

namespace uammd {
/*A convenient struct to pack 3 complex numbers, that is 6 real numbers*/
template <class T> struct cufftComplex4_t {
  cufftComplex_t<T> x, y, z, w;

  using cufftComplex4 = cufftComplex4_t<T>;

  friend inline __device__ __host__ cufftComplex4
  operator+(const cufftComplex4 &a, const cufftComplex4 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  }
  friend inline __device__ __host__ void operator+=(cufftComplex4 &a,
                                                    const cufftComplex4 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
  }

  friend inline __device__ __host__ cufftComplex4
  operator-(const cufftComplex4 &a, const cufftComplex4 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
  }
  friend inline __device__ __host__ void operator-=(cufftComplex4 &a,
                                                    const cufftComplex4 &b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
  }

  friend inline __device__ __host__ cufftComplex4
  operator*(const cufftComplex4 &a, real b) {
    cufftComplex4 res;
    res.x = a.x * b;
    res.y = a.y * b;
    res.z = a.z * b;
    res.w = a.w * b;
    return res;
  }
  friend inline __device__ __host__ cufftComplex4
  operator*(real b, const cufftComplex4 &a) {
    return a * b;
  }
  friend inline __device__ __host__ void operator*=(cufftComplex4 &a, real b) {
    a = a * b;
  }

  friend inline __device__ __host__ cufftComplex4
  operator/(const cufftComplex4 &a, real b) {
    cufftComplex4 res;
    res.x = a.x / b;
    res.y = a.y / b;
    res.z = a.z / b;
    res.w = a.w / b;
    return res;
  }
  friend inline __device__ __host__ cufftComplex4
  operator/(real b, const cufftComplex4 &a) {
    cufftComplex4 res;
    res.x = b / a.x;
    res.y = b / a.y;
    res.z = b / a.z;
    res.w = b / a.w;
    return res;
  }
  friend inline __device__ __host__ void operator/=(cufftComplex4 &a, real b) {
    a = a / b;
  }
};

} // namespace uammd

#endif