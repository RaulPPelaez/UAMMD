#ifndef UAMMD_ATOMICS
#define UAMMD_ATOMICS
#include "global/defines.h"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
namespace uammd {
template <class T> inline __device__ T atomicAdd(T *address, T val) {
  return ::atomicAdd(address, val);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
inline __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 900
inline __device__ float2 atomicAdd(float2 *address, float2 val) {
  float2 newval;
  if (val.x)
    newval.x = atomicAdd(&(*address).x, val.x);
  if (val.y)
    newval.y = atomicAdd(&(*address).y, val.y);
  return newval;
}

inline __device__ float4 atomicAdd(float4 *address, float4 val) {
  float4 newval;
  if (val.x)
    newval.x = atomicAdd(&(*address).x, val.x);
  if (val.y)
    newval.y = atomicAdd(&(*address).y, val.y);
  if (val.z)
    newval.z = atomicAdd(&(*address).z, val.z);
  if (val.w)
    newval.z = atomicAdd(&(*address).w, val.w);
  return newval;
}
#endif

inline __device__ double2 atomicAdd(double2 *address, double2 val) {
  double2 newval;
  if (val.x)
    newval.x = atomicAdd(&(*address).x, val.x);
  if (val.y)
    newval.y = atomicAdd(&(*address).y, val.y);
  return newval;
}

inline __device__ real3 atomicAdd(real3 *address, real3 val) {
  real3 newval;
  if (val.x)
    newval.x = atomicAdd(&(*address).x, val.x);
  if (val.y)
    newval.y = atomicAdd(&(*address).y, val.y);
  if (val.z)
    newval.z = atomicAdd(&(*address).z, val.z);
  return newval;
}

inline __device__ double4_16a atomicAdd(double4_16a *address, double4_16a val) {
  double4_16a newval;
  if (val.x)
    newval.x = atomicAdd(&(*address).x, val.x);
  if (val.y)
    newval.y = atomicAdd(&(*address).y, val.y);
  if (val.z)
    newval.z = atomicAdd(&(*address).z, val.z);
  if (val.w)
    newval.z = atomicAdd(&(*address).w, val.w);
  return newval;
}

template <class T, class T2> inline __device__ T2 atomicAdd(T &ref, T2 val) {
  return atomicAdd(&ref, val);
}

template <class T, class T2>
inline __device__ T2 atomicAdd(thrust::tuple<T &, T &> &&refs, T2 val) {
  T2 newval;
  if (val.x)
    newval.x = atomicAdd(&thrust::get<0>(refs), val.x);
  if (val.y)
    newval.y = atomicAdd(&thrust::get<1>(refs), val.y);
  return newval;
}

template <class T, class T2>
inline __device__ T2 atomicAdd(thrust::tuple<T &, T &, T &> &&refs, T2 val) {
  T2 newval;
  if (val.x)
    newval.x = atomicAdd(&thrust::get<0>(refs), val.x);
  if (val.y)
    newval.y = atomicAdd(&thrust::get<1>(refs), val.y);
  if (val.z)
    newval.z = atomicAdd(&thrust::get<2>(refs), val.z);
  return newval;
}

template <class T, class T2>
inline __device__ T2 atomicAdd(thrust::tuple<T &, T &, T &, T &> &&refs,
                               T2 val) {
  T2 newval;
  if (val.x)
    newval.x = atomicAdd(&thrust::get<0>(refs), val.x);
  if (val.y)
    newval.y = atomicAdd(&thrust::get<1>(refs), val.y);
  if (val.z)
    newval.z = atomicAdd(&thrust::get<2>(refs), val.z);
  if (val.w)
    newval.w = atomicAdd(&thrust::get<3>(refs), val.w);
  return newval;
}

} // namespace uammd

#endif
