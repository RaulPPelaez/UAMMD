#ifndef CUSOLVER_DEBUG_H
#define CUSOLVER_DEBUG_H


#include"debugTools.cuh"

#ifdef CUDA_ERROR_CHECK
#define CUSOLVER_ERROR_CHECK
#endif

#define CusolverSafeCall(err) __cusolverSafeCall(err, __FILE__, __LINE__)



const char* cusolverGetErrorString(cusolverStatus_t status){
  switch(status){
  case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
  case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
  case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
  case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
  case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
  case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
  default: return "UNKNOWN_ERROR";
  }
}

inline void __cusolverSafeCall( cusolverStatus_t err, const char *file, const int line )
{
  #ifdef CUSOLVER_ERROR_CHECK
  if ( CUSOLVER_STATUS_SUCCESS != err )
    {
      fprintf( stderr, "cusolverSafeCall() failed at %s:%i : %s - code: %i\n",
	       file, line, cusolverGetErrorString( err ), err);
      exit( -1 );
    }
  #endif

  return;
}

#endif
