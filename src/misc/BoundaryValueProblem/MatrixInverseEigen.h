#pragma once
#include <Eigen/Dense>
#include <complex>
#include <global/defines.h>
#include <type_traits>
#include <vector>

namespace uammd {
namespace BVP {
template <class T>
std::vector<T> invertSquareMatrix(const std::vector<T> &A, int N) {
  auto invA = A; // copy
  using Scalar = std::conditional_t<
      std::is_same_v<T, float2>, std::complex<float>,
      std::conditional_t<std::is_same_v<T, double2>, std::complex<double>, T>>;
  Scalar *dataPtr = reinterpret_cast<Scalar *>(invA.data());
  Eigen::Map<
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      M(dataPtr, N, N);
  M = M.inverse();
  return invA;
}
} // namespace BVP
} // namespace uammd
