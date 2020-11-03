#ifndef UAMMD_ATOMICS
#define UAMMD_ATOMICS
#include "global/defines.h"
namespace uammd{
  template<class T>
  inline __device__ T atomicAdd(T* address, T val){ return ::atomicAdd(address, val);}

#ifndef SINGLE_PRECISION
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
  inline __device__ double atomicAdd(double* address, double val){
    unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
		      __double_as_longlong(val +
					   __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }
#endif
#endif

  inline __device__ real3 atomicAdd(real3* address, real3 val){
    real3 newval;
    if(val.x) newval.x = atomicAdd(&(*address).x, val.x);
    if(val.y) newval.y = atomicAdd(&(*address).y, val.y);
    if(val.z) newval.z = atomicAdd(&(*address).z, val.z);
    return newval;
  }

  inline __device__ real2 atomicAdd(real2* address, real2 val){
    real2 newval;
    if(val.x) newval.x = atomicAdd(&(*address).x, val.x);
    if(val.y) newval.y = atomicAdd(&(*address).y, val.y);
    return newval;
  }
}

#endif
