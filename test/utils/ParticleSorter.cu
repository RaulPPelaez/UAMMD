/*Raul P. Pelaez 2022. Tests for the Lanczos algorithm.
  Tests the result of sqrt(M)*v for increasingly complex matrices and vectors of
  several sizes.
 */
#include "utils/ParticleSorter.cuh"
#include "utils/container.h"
#include "utils/execution_policy.cuh"
#include "gmock/gmock.h"
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <random>
using namespace uammd;

template <class T> using cached_vector = uninitialized_cached_vector<T>;

TEST(Sorter, CanBeCreated) { auto sorter = std::make_shared<ParticleSorter>(); }

struct TrivialHash {
  inline __host__ __device__ uint operator()(uint i) const { return i; }
};

TEST(Sorter, SortsCorrectly) {
  auto sorter = std::make_shared<ParticleSorter>();
  int n = 163840;
  cached_vector<uint> vec(n);
  auto svec = vec;
  thrust::sequence(vec.rbegin(), vec.rend(), 0);
  cudaStream_t st;
  cudaStreamCreate(&st);
  thrust::sequence(uammd::cached_device_execution_policy.on(st), vec.rbegin(),
                   vec.rend(), 0);
  uint max_hash = n; // Optional parameter, increases performance if known
  // Transforms the values into hashes
  auto hash = thrust::make_transform_iterator(vec.begin(), TrivialHash());
  // Make sorter aware of the latest hashes
  sorter->updateOrderWithCustomHash(hash, n, max_hash, st);
  // Reorder a given vector with the latest order
  sorter->applyCurrentOrder(vec.begin(), svec.begin(), n, st);
  std::vector<int> hvec, htheo;
  thrust::copy(svec.begin(), svec.end(), std::back_inserter(hvec));
  thrust::copy(vec.begin(), vec.end(), std::back_inserter(htheo));
  for (int i = 0; i < n; i++) {
    ASSERT_EQ(hvec[i], htheo.rbegin()[i]);
  }
}
