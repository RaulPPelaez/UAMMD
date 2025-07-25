/*Raul P. Pelaez 2019-2021. Boundary Value Problem Matrix algebra utilities
 */
#ifndef BVP_MATRIX_UTILS_H
#define BVP_MATRIX_UTILS_H
#include <thrust/pair.h>
#include <utils/exception.h>
#include <vector>
#ifdef USE_EIGEN
#include "MatrixInverseEigen.h"
#else
#include "MatrixInvertLapacke.h"
#endif
#include <thrust/complex.h>
namespace uammd {
namespace BVP {
using complex = thrust::complex<real>;

template <class T, class T2>
auto matmul(const T &A, int ncol_a, int nrow_a, const T2 &B, int ncol_b,
            int nrow_b) {
  using type = typename T::value_type;
  std::vector<type> C;
  C.resize(ncol_b * nrow_a);
  for (int i = 0; i < nrow_a; i++) {
    for (int j = 0; j < ncol_b; j++) {
      type tmp{};
      for (int k = 0; k < ncol_a; k++) {
        tmp += A[k + ncol_a * i] * B[j + ncol_b * k];
      }
      C[j + ncol_b * i] = tmp;
    }
  }
  return C;
}
// Solves the linear system A*x = b
// Given A (2x2 matrix) and b as two numbers of an arbitrary type (could be
// complex, real...)
template <class T, class R>
__device__ thrust::pair<T, T> solve2x2System(thrust::complex<R> A[4],
                                             thrust::pair<T, T> b) {
  const auto det = A[0] * A[3] - A[1] * A[2];
  const T c0 = (b.first * A[3] - A[1] * b.second) / det;
  const T d0 = (b.second * A[0] - b.first * A[2]) / det;
  return thrust::make_pair(c0, d0);
}

template <class T>
__device__ thrust::pair<T, T> solve2x2System(real4 A, thrust::pair<T, T> b) {
  const real det = A.x * A.w - A.y * A.z;
  const T c0 = (b.first * A.w - A.y * b.second) / det;
  const T d0 = (b.second * A.x - b.first * A.z) / det;
  return thrust::make_pair(c0, d0);
}

template <class T>
__device__ thrust::pair<T, T> solve2x2System(real A[4], thrust::pair<T, T> b) {
  real4 A_r4 = {A[0], A[1], A[2], A[3]};
  return solve2x2System(A_r4, b);
}
} // namespace BVP
} // namespace uammd
#endif
