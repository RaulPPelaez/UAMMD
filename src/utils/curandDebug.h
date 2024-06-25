#ifndef CURAND_DEBUG_H
#define CURAND_DEBUG_H
#include"debugTools.h"
#ifdef CUDA_ERROR_CHECK
#define CURAND_ERROR_CHECK
#endif


#define CurandSafeCall(err) __curandSafeCall(err, __FILE__, __LINE__)

inline
const char* curandGetErrorString(curandStatus_t err){
  switch (err) {
  case CURAND_STATUS_VERSION_MISMATCH:    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:     return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:   return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:          return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:        return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:  return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:      return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:      return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:       return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:      return "CURAND_STATUS_INTERNAL_ERROR";
  default: return "CURAND Unknown error code\n";
  }
}

inline void __curandSafeCall( curandStatus_t err, const char *file, const int line ){
  #ifdef CURAND_ERROR_CHECK
  if ( CURAND_STATUS_SUCCESS != err )
    {
      fprintf( stderr, "curandSafeCall() failed at %s:%i : %s - code: %i\n",
	       file, line, curandGetErrorString( err ), err);
      exit( -1 );
    }
  #endif

  return;
}

#endif
