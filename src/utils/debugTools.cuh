/*Raul P. Pelaez 2019. Some utilities for debugging GPU code

 */
#ifndef DEBUGTOOLS_CUH
#define DEBUGTOOLS_CUH

#define CUDA_ERROR_CHECK

#ifdef UAMMD_DEBUG
#define CUDA_ERROR_CHECK_SYNC
#endif

#include"utils/exception.h"

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

namespace uammd{

  class cuda_generic_error: public std::runtime_error{
    cudaError_t error_code;
  public:
    cuda_generic_error(std::string msg, cudaError_t err):
      std::runtime_error(msg + ": " + cudaGetErrorString(err) + " - code: " + std::to_string(err)),
      error_code(err){}

    cudaError_t code(){return error_code;}
  };

}

inline void __cudaSafeCall(cudaError err, const char *file, const int line){
  #ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err){
    cudaGetLastError(); //Reset CUDA error status    
    throw uammd::cuda_generic_error("CudaSafeCall() failed at "+
				    std::string(file) + ":" + std::to_string(line), err);
  }
  #endif
}

inline void __cudaCheckError(const char *file, const int line){
#ifdef CUDA_ERROR_CHECK_SYNC
  err = cudaDeviceSynchronize();
  if(cudaSuccess != err){
    throw uammd::cuda_generic_error("CudaCheckError() with sync failed at "+
				    std::string(file) + ":" + std::to_string(line), err);
  }
#endif
  cudaError err = cudaGetLastError();
  if(cudaSuccess != err){
    throw uammd::cuda_generic_error("CudaCheckError() failed at "+
				    std::string(file) + ":" + std::to_string(line), err);
  }
}

#endif
