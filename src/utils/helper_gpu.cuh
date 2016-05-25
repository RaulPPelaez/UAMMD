#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#include<stdio.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}


#define cudaCheckErrors(msg) \
  do { \
  cudaError_t __err = cudaGetLastError(); \
  if (__err != cudaSuccess) { \
  fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
    msg, cudaGetErrorString(__err), \
	  __FILE__, __LINE__); \
  fprintf(stderr, "*** FAILED - ABORTING\n"); \
  exit(1); \
  } \
  } while (0)

#endif
