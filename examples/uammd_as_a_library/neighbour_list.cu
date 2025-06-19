/*Raul P. Pelaez 2021. An example on how to use a neighbour list outside the
  UAMMD ecosystem You can copy paste parts of this example to accelerate your
  code.

  This example will generate some random positions inside a cubic periodic box,
    then create a neighbour list with them and finally traverse that list.
 */
#include "Interactor/NeighbourList/BasicList/BasicListBase.cuh"
#include "uammd.cuh"
#include <thrust/extrema.h>
#include <thrust/random.h>
using namespace uammd;
template <class T> using gpu_container = thrust::device_vector<T>;

// A thrust functor to generate random numbers
struct RNG {
  real L;
  RNG(real L) : L(L) {}
  __device__ real4 operator()(int i) {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<real> dist(-L * 0.5, L * 0.5);
    rng.discard(i);
    return {dist(rng), dist(rng), dist(rng), 0};
  }
};

// Creates and returns a vector with random positions inside a cubic box of side
// L (always the same random positions)
gpu_container<real4> generateRandomPositions(real L, int numberParticles) {
  gpu_container<real4> positions(numberParticles);
  auto it = thrust::make_counting_iterator<int>(0);
  thrust::transform(it, it + numberParticles, positions.begin(), RNG(L));
  return positions;
}

// This kernel takes the basic neighbour list and uses it to traverse the
// neighbours of each particle It is a template to hide the ugly long name of
// the structure This type could also be found with using NeighbourList =
// decltype(&BasicNeighbourListBase::getBasicNeighbourList);
template <class NeighbourList>
__global__ void useNeighbourList(NeighbourList nl, real4 *positions,
                                 int numberParticles) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberParticles)
    return;
  // Number of neighbours of particle in index tid
  int nneigh = nl.numberNeighbours[tid];
  // Index of first neighbour of particle tid in neighbourList
  int offset = nl.particleStride[tid];
  // Position of particle in index tid, using the internal copy of the positions
  // yields better performance
  real4 pos_i = nl.sortPos[tid];
  // Equivalent alternative, but with slower access pattern
  // real4 pos_i = positions[nl.groupIndex[tid]];
  for (int j = 0; j < nneigh; j++) {
    // Index of neighbour particle in sortPos
    int index_j = nl.neighbourList[offset + j];
    // Position of neighbour, using the internal copy of the positions yields
    // better performance
    real4 pos_j = nl.sortPos[index_j];
    // Equivalent alternative, but with slower access pattern:
    // real4 pos = positions[nl.groupIndex[index_j]];
    // Do something with both neighbours:
    //...
    // Simply use positions to avoid "unused variable" compiler warning
    pos_i + pos_j;
  }
}

// Prepares and launches the kernel above with one thread per particle
// Notice that a block per particle would probably be more performant, but
// results in slightly more complex code.
template <class NeighbourList>
void useListWithCUDAKernel(gpu_container<real4> &positions,
                           NeighbourList listData) {
  const int numberParticles = positions.size();
  constexpr int blockSize = 128;
  const int numberBlocks = numberParticles / blockSize + 1;
  auto pos_ptr = thrust::raw_pointer_cast(positions.data());
  useNeighbourList<<<numberBlocks, blockSize>>>(listData, pos_ptr,
                                                numberParticles);
  cudaDeviceSynchronize();
  CudaCheckError();
}

// Traverses the list (equivalently to the function useListWithCUDAKernel) but
// using thrust::for_each instead of a CUDA kernel The main benefit o this
// version is that thrust will handle
template <class NeighbourList>
void useListWithThrust(gpu_container<real4> &positions, NeighbourList nl) {
  const int numberParticles = positions.size();
  auto cit = thrust::make_counting_iterator<int>(0);
  thrust::for_each(cit, cit + numberParticles, [=] __device__(int tid) {
    int nneigh = nl.numberNeighbours[tid];
    int offset = nl.particleStride[tid];
    real4 pos_i = nl.sortPos[tid];
    for (int j = 0; j < nneigh; j++) {
      int index_j = nl.neighbourList[offset + j];
      real4 pos_j = nl.sortPos[index_j];
      // Do something with both neighbours:
      //...
      // Simply use positions to avoid "unused variable" compiler warning
      pos_i + pos_j;
    }
  });
}

struct NlistCPU {
  // Neighbour j of particle i (of a total of nneigh[i] neighbours):
  // list[stride*i + j]
  std::vector<int> nneigh;
  int stride;
  std::vector<int> list;
};

// Lets change the layout of the list to something more CPU friendly and return
// it
template <class NeighbourList>
NlistCPU downloadList(int numberParticles, NeighbourList nl) {
  NlistCPU h_nl;
  // The CPU neighbour list will store a list of size "stride" for each
  // particle.
  thrust::device_ptr<const int> d_nneigh(nl.numberNeighbours);
  // stride is the maximum number of neighbours a particle has
  int stride = *thrust::max_element(d_nneigh, d_nneigh + numberParticles);
  h_nl.stride = stride;
  h_nl.nneigh.resize(numberParticles);
  thrust::copy(d_nneigh, d_nneigh + numberParticles, h_nl.nneigh.begin());
  // Lets fill a gpu version of the cpu list with the following layout:
  // Neighbour j of particle i (of a total of nneigh[i] neighbours):
  // list[stride*i + j]
  thrust::device_vector<int> d_list(numberParticles * h_nl.stride);
  auto d_list_ptr = thrust::raw_pointer_cast(d_list.data());
  auto cit = thrust::make_counting_iterator<int>(0);
  thrust::for_each(cit, cit + numberParticles, [=] __device__(int tid) {
    int nneigh = nl.numberNeighbours[tid];
    int stride_i = nl.particleStride[tid];
    int i = nl.groupIndex[tid];
    for (int j = 0; j < nneigh; j++) {
      // We are using the internal format of the BasicList, which stores the
      // neighbors like this
      int index_j = nl.groupIndex[nl.neighbourList[stride_i * j + tid]];
      d_list_ptr[stride * i + j] = index_j;
    }
  });
  // Once the gpu verison is filled we copy it to the cpu.
  h_nl.list.resize(h_nl.stride * numberParticles);
  thrust::copy(d_list.begin(), d_list.end(), h_nl.list.begin());
  return h_nl;
}

int main(int argc, char *argv[]) {
  // Some arbitrary parameters
  const int numberParticles = 1e5;
  const real L = 128;
  const real rcut = 2.5;
  // Fill a vector with random positions
  auto positions = generateRandomPositions(L, numberParticles);
  BasicNeighbourListBase nl;
  // A cubic box of size L
  Box box({L, L, L});
  // You can make the box aperiodic in some direction with this:
  // box.setPeriodicity(0,0,0); //Completely aperiodic box
  // Update/create the neighbour list
  nl.update(positions.begin(), numberParticles, box, rcut);
  // Fetch the list (with the positions given at the last update)
  auto listDataGPU = nl.getBasicNeighbourList();
  // Use the list by writing a CUDA kernel, currently it just does nothing.
  useListWithCUDAKernel(positions, listDataGPU);
  // Use the list with thrust, without the need to write a CUDA kernel
  useListWithThrust(positions, listDataGPU);
  // Change the layout of the list and download it to the CPU
  auto listCPU = downloadList(numberParticles, listDataGPU);
  return 0;
}
