/*Tests for the CellList NeighbourContainer.
  Compares the number of neighbours found traversing the cell list against a
  brute force search for every combination of box periodicity.
 */
#include "Interactor/NeighbourList/CellList/CellListBase.cuh"
#include "Interactor/NeighbourList/CellList/NeighbourContainer.cuh"
#include "gmock/gmock.h"
#include <gtest/gtest.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <tuple>
using namespace uammd;

__global__ void countNeighboursCellList(CellList_ns::NeighbourContainer nc,
                                        Box box, real rc, int numberParticles,
                                        int *count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numberParticles)
    return;
  nc.set(i);
  const real3 pi = make_real3(nc.getSortedPositions()[i]);
  int nneigh = 0;
  for (auto it = nc.begin(); it;) {
    auto neigh = *it++;
    const real3 rij = box.apply_pbc(make_real3(neigh.getPos()) - pi);
    if (dot(rij, rij) < rc * rc)
      nneigh++;
  }
  count[nc.getGroupIndexes()[i]] = nneigh;
}

std::vector<int> countNeighboursBruteForce(const thrust::host_vector<real4> &pos,
                                           Box box, real rc) {
  const int numberParticles = pos.size();
  std::vector<int> count(numberParticles, 0);
  for (int i = 0; i < numberParticles; i++) {
    for (int j = 0; j < numberParticles; j++) {
      const real3 rij = box.apply_pbc(make_real3(pos[j]) - make_real3(pos[i]));
      if (dot(rij, rij) < rc * rc)
        count[i]++;
    }
  }
  return count;
}

thrust::host_vector<real4> randomPositionsInBox(int numberParticles, real L) {
  std::mt19937 gen(1234);
  std::uniform_real_distribution<real> dist(-0.5 * L, 0.5 * L);
  thrust::host_vector<real4> pos(numberParticles);
  for (auto &p : pos)
    p = make_real4(dist(gen), dist(gen), dist(gen), 0);
  return pos;
}

class CellListPeriodicity
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {};

TEST_P(CellListPeriodicity, FindsSameNeighboursAsBruteForce) {
  const auto [px, py, pz] = GetParam();
  const int numberParticles = 4000;
  const real L = 16;
  const real rc = 2;
  Box box(make_real3(L));
  box.setPeriodicity(px, py, pz);
  Grid grid(box, make_int3(L / rc));
  auto h_pos = randomPositionsInBox(numberParticles, L);
  thrust::device_vector<real4> pos = h_pos;
  CellListBase cl;
  cl.update(thrust::raw_pointer_cast(pos.data()), numberParticles, grid);
  thrust::device_vector<int> d_count(numberParticles, -1);
  CellList_ns::NeighbourContainer nc(cl.getCellList());
  countNeighboursCellList<<<(numberParticles + 127) / 128, 128>>>(
      nc, box, rc, numberParticles, thrust::raw_pointer_cast(d_count.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<int> count = d_count;
  auto expected = countNeighboursBruteForce(h_pos, box, rc);
  for (int i = 0; i < numberParticles; i++) {
    ASSERT_EQ(count[i], expected[i])
        << "Particle " << i << " at " << h_pos[i].x << " " << h_pos[i].y << " "
        << h_pos[i].z;
  }
}

INSTANTIATE_TEST_SUITE_P(AllPeriodicities, CellListPeriodicity,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                            ::testing::Bool()));
