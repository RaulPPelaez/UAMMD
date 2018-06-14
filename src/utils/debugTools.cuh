/*Raul P. Pelaez 2017. Some utilities for debugging GPU code

 */
#ifndef DEBUGTOOLS_CUH
#define DEBUGTOOLS_CUH


//#define USE_NVTX
#define CUDA_ERROR_CHECK
#ifdef UAMMD_DEBUG
#define CUDA_ERROR_CHECK_SYNC
#endif

#ifdef USE_NVTX


#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
  }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
  #ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s - code: %i\n",
	       file, line, cudaGetErrorString( err ), err);
      exit( -1 );
    }
  #endif

  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
  #ifdef CUDA_ERROR_CHECK_SYNC
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	       file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
	       file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  #endif

  return;
}

#endif
