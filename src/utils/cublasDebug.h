#ifndef CUBLAS_DEBUG_H
#define CUBLAS_DEBUG_H
#include"debugTools.cuh"
#ifdef CUDA_ERROR_CHECK
#define CUBLAS_ERROR_CHECK
#endif

#define CublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)



const char* cublasGetErrorString(cublasStatus_t status){
  switch(status){
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

inline void __cublasSafeCall( cublasStatus_t err, const char *file, const int line )
{
  #ifdef CUBLAS_ERROR_CHECK
  if ( CUBLAS_STATUS_SUCCESS != err )
    {
      fprintf( stderr, "cublasSafeCall() failed at %s:%i : %s - code: %i\n",
	       file, line, cublasGetErrorString( err ), err);
      exit( -1 );
    }
  #endif

  return;
}

#endif
