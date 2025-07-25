#pragma once
#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "utils/cufftPrecisionAgnostic.h"
#include <global/defines.h>
#include <thrust/complex.h>
#include <vector>
namespace uammd {
namespace BVP {
template <class T> struct LapackeUAMMD;
template <> struct LapackeUAMMD<float> {
  template <class... T> static int getrf(T... args) {
    return LAPACKE_sgetrf(args...);
  }
  template <class... T> static int getri(T... args) {
    return LAPACKE_sgetri(args...);
  }
};
template <> struct LapackeUAMMD<double> {
  template <class... T> static int getrf(T... args) {
    return LAPACKE_dgetrf(args...);
  }
  template <class... T> static int getri(T... args) {
    return LAPACKE_dgetri(args...);
  }
};

template <> struct LapackeUAMMD<lapack_complex_float> {
  template <class... T> static int getrf(T... args) {
    return LAPACKE_cgetrf(args...);
  }
  template <class... T> static int getri(T... args) {
    return LAPACKE_cgetri(args...);
  }
};
template <> struct LapackeUAMMD<lapack_complex_double> {
  template <class... T> static int getrf(T... args) {
    return LAPACKE_zgetrf(args...);
  }
  template <class... T> static int getri(T... args) {
    return LAPACKE_zgetri(args...);
  }
};
template <> struct LapackeUAMMD<cufftComplex_t<float>> {
  static int getrf(int matrix_layout, lapack_int n, lapack_int m,
                   cufftComplex_t<float> *a, lapack_int lda, lapack_int *ipiv) {
    return LAPACKE_cgetrf(matrix_layout, n, m, (lapack_complex_float *)(a), lda,
                          ipiv);
  }
  static int getri(int matrix_layout, lapack_int n, cufftComplex_t<float> *a,
                   lapack_int lda, lapack_int *ipiv) {
    return LAPACKE_cgetri(matrix_layout, n, (lapack_complex_float *)(a), lda,
                          ipiv);
  }
};
template <> struct LapackeUAMMD<cufftComplex_t<double>> {
  static int getrf(int matrix_layout, lapack_int n, lapack_int m,
                   cufftComplex_t<double> *a, lapack_int lda,
                   lapack_int *ipiv) {
    return LAPACKE_zgetrf(matrix_layout, n, m, (lapack_complex_double *)(a),
                          lda, ipiv);
  }
  static int getri(int matrix_layout, lapack_int n, cufftComplex_t<double> *a,
                   lapack_int lda, lapack_int *ipiv) {
    return LAPACKE_zgetri(matrix_layout, n, (lapack_complex_double *)(a), lda,
                          ipiv);
  }
};
template <> struct LapackeUAMMD<thrust::complex<float>> {
  static int getrf(int matrix_layout, lapack_int n, lapack_int m,
                   thrust::complex<float> *a, lapack_int lda,
                   lapack_int *ipiv) {
    return LAPACKE_cgetrf(matrix_layout, n, m, (lapack_complex_float *)(a), lda,
                          ipiv);
  }
  static int getri(int matrix_layout, lapack_int n, thrust::complex<float> *a,
                   lapack_int lda, lapack_int *ipiv) {
    return LAPACKE_cgetri(matrix_layout, n, (lapack_complex_float *)(a), lda,
                          ipiv);
  }
};
template <> struct LapackeUAMMD<thrust::complex<double>> {
  static int getrf(int matrix_layout, lapack_int n, lapack_int m,
                   thrust::complex<double> *a, lapack_int lda,
                   lapack_int *ipiv) {
    return LAPACKE_zgetrf(matrix_layout, n, m, (lapack_complex_double *)(a),
                          lda, ipiv);
  }
  static int getri(int matrix_layout, lapack_int n, thrust::complex<double> *a,
                   lapack_int lda, lapack_int *ipiv) {
    return LAPACKE_zgetri(matrix_layout, n, (lapack_complex_double *)(a), lda,
                          ipiv);
  }
};

template <class T>
std::vector<T> invertSquareMatrix(const std::vector<T> &A, lapack_int N) {
  lapack_int pivotArray[N];
  int errorHandler;
  auto invA = A;
  errorHandler = LapackeUAMMD<T>::getrf(LAPACK_ROW_MAJOR, N, N, invA.data(), N,
                                        pivotArray);
  if (errorHandler) {
    throw std::runtime_error("Lapacke getrf failed with error code: " +
                             std::to_string(errorHandler));
  }
  errorHandler =
      LapackeUAMMD<T>::getri(LAPACK_ROW_MAJOR, N, invA.data(), N, pivotArray);
  if (errorHandler) {
    throw std::runtime_error("Lapacke getri failed with error code: " +
                             std::to_string(errorHandler));
  }
  return invA;
}
} // namespace BVP
} // namespace uammd
