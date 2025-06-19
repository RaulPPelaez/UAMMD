#ifndef CUFFTCOMPLEX3_CUH
#define CUFFTCOMPLEX3_CUH

#include "global/defines.h"
#include "utils/cufftPrecisionAgnostic.h"

namespace uammd {
/*A convenient struct to pack 3 complex numbers, that is 6 real numbers*/
template <class T> struct cufftComplex3_t {
  cufftComplex_t<T> x, y, z;

  using cufftComplex3 = cufftComplex3_t<T>;

  friend inline __device__ __host__ cufftComplex3
  operator+(const cufftComplex3 &a, const cufftComplex3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }
  friend inline __device__ __host__ void operator+=(cufftComplex3 &a,
                                                    const cufftComplex3 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
  }

  friend inline __device__ __host__ cufftComplex3
  operator-(const cufftComplex3 &a, const cufftComplex3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
  }
  friend inline __device__ __host__ void operator-=(cufftComplex3 &a,
                                                    const cufftComplex3 &b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
  }

  friend inline __device__ __host__ cufftComplex3
  operator*(const cufftComplex3 &a, real b) {
    cufftComplex3 res;
    res.x = a.x * b;
    res.y = a.y * b;
    res.z = a.z * b;
    return res;
  }
  friend inline __device__ __host__ cufftComplex3
  operator*(real b, const cufftComplex3 &a) {
    return a * b;
  }
  friend inline __device__ __host__ void operator*=(cufftComplex3 &a, real b) {
    a = a * b;
  }

  friend inline __device__ __host__ cufftComplex3
  operator/(const cufftComplex3 &a, real b) {
    cufftComplex3 res;
    res.x = a.x / b;
    res.y = a.y / b;
    res.z = a.z / b;
    return res;
  }
  friend inline __device__ __host__ cufftComplex3
  operator/(real b, const cufftComplex3 &a) {
    cufftComplex3 res;
    res.x = b / a.x;
    res.y = b / a.y;
    res.z = b / a.z;
    return res;
  }
  friend inline __device__ __host__ void operator/=(cufftComplex3 &a, real b) {
    a = a / b;
  }
};

} // namespace uammd

#endif