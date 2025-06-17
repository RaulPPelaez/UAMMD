

#ifndef DEVICE_BLAS_H
#define DEVICE_BLAS_H
#include "global/defines.h"
#include "utils/cublasDebug.h"
#include "utils/cuda_lib_defines.h"
namespace uammd {
namespace lanczos {
template <class... T>
auto device_gemv(cublasHandle_t cublas_handle, T... args) {
  CublasSafeCall(cublasgemv(cublas_handle, CUBLAS_OP_N, args...));
}
template <class... T>
auto device_nrm2(cublasHandle_t cublas_handle, T... args) {
  CublasSafeCall(cublasnrm2(cublas_handle, args...));
}
template <class... T>
auto device_axpy(cublasHandle_t cublas_handle, T... args) {
  CublasSafeCall(cublasaxpy(cublas_handle, args...));
}
template <class... T> auto device_dot(cublasHandle_t cublas_handle, T... args) {
  CublasSafeCall(cublasdot(cublas_handle, args...));
}
template <class... T>
auto device_scal(cublasHandle_t cublas_handle, T... args) {
  CublasSafeCall(cublasscal(cublas_handle, args...));
}
} // namespace lanczos
} // namespace uammd
#endif
