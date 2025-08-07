/*Raul P. Pelaez 2018-2020. Adapted from Pablo Ibanez Freire's MonteCarlo
Anderson code.

  This module implements Anderson's Monte Carlo NVT GPU algorithm [1].
  The algorithm is presented for hard spheres in two dimensions, but suggests it
should be valid for any interparticle potential or dimensionality.

  Works by partitioning the system into a checkerboard like pattern in which the
cells of a given subgrid are separated a distance greater or equal than the cut
off distance of the potential. This allows for the cells of a subgrid to be
updated independently. In 3D there are 8 subgrids (4 in 2D). Each one of the 8
subgrids is processed sequentially.

  The algorithm can be summarized as follows:


  1- The checkerboard is placed with a random origin
  2- For each subgrid perform a certain prefixed number of trials on the
particles inside each cell

Certain details must be taken into account to ensure detailed-balance:
  1- Any origin in the simulation box must be equally probable
  2- The different subgrids must be processed in a random order
  3- The particles in each cell must be processed in a random order*
  4- Any particle attempt to leave the cell is directly rejected


  * We believe that detailed balance is mantained even if the particles are
selected randomly (as opposed to traverse a random permutation of the particle
list) References:

[1] Massively parallel Monte Carlo for many-particle simulations on GPUs. Joshua
A. Anderson et. al. https://arxiv.org/pdf/1211.1646.pdf

TODO:
100- Optimize kernel launch parameters and MCStepKernel.
80- Get rid of the CellList, this algorithm can be implemented without
reconstructing a cell list from scratch eac step. Although the bottleneck is
probably the traversal MCStepKernel.
 */
#include "Anderson.cuh"
#include "Interactor/NeighbourList/CellList/NeighbourContainer.cuh"
#include "Interactor/NeighbourList/common.cuh"
#include "utils/TransverserUtils.cuh"
#include <third_party/saruprng.cuh>
#include <thrust/iterator/permutation_iterator.h>
namespace uammd {
namespace MC_NVT {

namespace Anderson_ns {

Grid createGrid(Box box, real cutOff) {
  int3 cellDim = make_int3(box.boxSize / cutOff);
  // I need an even number of cells
  if (cellDim.x % 2 != 0)
    cellDim.x -= 1;
  if (cellDim.y % 2 != 0)
    cellDim.y -= 1;
  if (cellDim.z % 2 != 0)
    cellDim.z -= 1;
  if (box.boxSize.z == 0) {
    cellDim.z = 1;
  }
  return Grid(box, cellDim);
}

bool checkGridValidity(Grid grid) {
  auto cellDim = grid.cellDim;
  if (cellDim.x < 3 or cellDim.y < 3 or cellDim.z == 2) {
    return false;
  }
  return true;
}
} // namespace Anderson_ns

template <class Pot>
Anderson<Pot>::Anderson(shared_ptr<ParticleData> pd, shared_ptr<Pot> pot,
                        Parameters in_par)
    : Integrator(pd, "MonteCarlo::Anderson"), pot(pot), steps(0), par(in_par),
      is2D(false), jumpSize(par.initialJumpSize) {
  sys->log<System::MESSAGE>("[MC_NVT::Anderson] Created");
  sys->log<System::MESSAGE>("[MC_NVT::Anderson] Temperature: %e",
                            par.temperature);
  if (par.temperature < real(0.0)) {
    sys->log<System::ERROR>("[MC_NVT::Anderson] Please specify a temperature!");
    throw std::invalid_argument("Negative temperature detected");
  }
  cl = std::make_shared<CellListBase>();
  if (par.box.boxSize.z == real(0.0)) {
    this->is2D = true;
  }
  this->updateSimulationBox(par.box);
  sys->log<System::MESSAGE>("[MC_NVT::Anderson] Box size: %e %e %e",
                            grid.box.boxSize.x, grid.box.boxSize.y,
                            grid.box.boxSize.z);
  sys->log<System::MESSAGE>("[MC_NVT::Anderson] Grid dimensions: %d %d %d",
                            grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
  this->seed = par.seed;
  if (par.seed == 0) {
    this->seed = sys->rng().next32();
  }
  CudaSafeCall(cudaStreamCreate(&st));
  CudaCheckError();
  this->currentAcceptanceRatio = 0;
}

template <class Pot> void Anderson<Pot>::updateSimulationBox(Box box) {
  for (auto updatable : updatables) {
    updatable->updateBox(par.box);
  }
  real rcut = pot->getCutOff();
  this->grid = Anderson_ns::createGrid(box, rcut);
  if (not Anderson_ns::checkGridValidity(grid)) {
    sys->log<System::ERROR>("[MC_NVT::Anderson] I cannot work with such a "
                            "large cut off (%e) in this box (%e)!",
                            rcut, box.boxSize.x);
    std::invalid_argument("Cut off is too large");
  }
  this->maxOriginDisplacement = 0.5 * grid.box.boxSize.x;
  int ncells = grid.getNumberCells();
  triedChanges.resize(ncells);
  acceptedChanges.resize(ncells);
  resetAcceptanceCounters();
}

template <class Pot> void Anderson<Pot>::updateAcceptanceRatio() {
  uint naccepted =
      thrust::reduce(acceptedChanges.begin(), acceptedChanges.end(), 0);
  uint ntries = thrust::reduce(triedChanges.begin(), triedChanges.end(), 0);
  this->currentAcceptanceRatio = real(naccepted) / ntries;
}

template <class Pot> void Anderson<Pot>::updateJumpSize() {
  const real3 maxJump = grid.cellSize;
  const real minJumpSize = grid.cellSize.x / 100000;
  if (currentAcceptanceRatio < par.acceptanceRatio) {
    jumpSize *= 0.9;
    if (jumpSize <= minJumpSize) {
      jumpSize = minJumpSize;
    }
  } else if (currentAcceptanceRatio > par.acceptanceRatio) {
    jumpSize *= 1.02;
    jumpSize = std::min({jumpSize, maxJump.x, maxJump.y});
    if (!is2D) {
      jumpSize = std::min(jumpSize, maxJump.z);
    }
  }
  sys->log<System::DEBUG>("[MC_NVT::Anderson] Current acceptance ratio: %e",
                          currentAcceptanceRatio);
  sys->log<System::DEBUG>(
      "[MC_NVT::Anderson] Current step size: %e, %e*cellSize", jumpSize,
      jumpSize / grid.cellSize.x);
}

template <class Pot> void Anderson<Pot>::forwardTime() {
  sys->log<System::DEBUG>(
      "[MC_NVT::Anderson] Performing Monte Carlo Parallel step: %d", steps);
  if (steps == 0) {
    for (auto updatable : updatables) {
      updatable->updateTemperature(par.temperature);
    }
  }
  steps++;
  updateOrigin();
  updateListWithCurrentOrigin();
  storeCurrentSortedPositions();
  performStep();
  updateGlobalPositions();
  if (steps % par.tuneSteps == 0 and steps > 1) {
    updateAcceptanceRatio();
    this->resetAcceptanceCounters();
    updateJumpSize();
  }
  CudaCheckError();
}

template <class Pot> void Anderson<Pot>::updateOrigin() {
  this->currentOrigin =
      make_real3(sys->rng().uniform3(-1.0, 1.0) * maxOriginDisplacement);
  if (is2D) {
    currentOrigin.z = 0;
  }
  sys->log<System::DEBUG1>("[MC_NVT::Anderson] Current origin: %e %e %e",
                           currentOrigin.x, currentOrigin.y, currentOrigin.z);
}

namespace Anderson_ns {
struct ShiftPositionTransform {
  real3 origin;

  ShiftPositionTransform(real3 ori) : origin(ori) {}

  __device__ real4 operator()(real4 pos) { return pos + make_real4(origin, 0); }
};
} // namespace Anderson_ns

template <class Pot> void Anderson<Pot>::updateListWithCurrentOrigin() {
  int numberParticles = pg->getNumberParticles();
  auto pos = pd->getPos(access::location::gpu, access::mode::read);
  auto posGroup = pg->getPropertyIterator(pos);
  auto origin_tr = Anderson_ns::ShiftPositionTransform(currentOrigin);
  auto shiftedPositions = thrust::make_transform_iterator(posGroup, origin_tr);
  cl->update(shiftedPositions, numberParticles, grid);
  CudaCheckError();
}

template <class Pot> void Anderson<Pot>::storeCurrentSortedPositions() {
  int numberParticles = pg->getNumberParticles();
  sortPos.resize(numberParticles);
  auto clData = cl->getCellList();
  auto origin_tr =
      Anderson_ns::ShiftPositionTransform(real(-1.0) * currentOrigin);
  thrust::transform(thrust::cuda::par, clData.sortPos,
                    clData.sortPos + numberParticles, sortPos.begin(),
                    origin_tr);
  CudaCheckError();
}

template <class Pot> void Anderson<Pot>::performStep() {
  const int numberSubGrids = is2D ? 4 : 8;
  std::array<int, 8> shuffled_indexes = {0, 1, 2, 3, 4, 5, 6, 7};
  fori(0, numberSubGrids - 1) {
    int j = i + (sys->rng().next() % (numberSubGrids - i));
    std::swap(shuffled_indexes[i], shuffled_indexes[j]);
  }
  fori(0, numberSubGrids) {
    int3 subgrid = offset3D[shuffled_indexes[i]];
    updateParticlesInSubgrid(subgrid);
  }
  CudaCheckError();
}

namespace Anderson_ns {
__device__ int3 computeCellFromThreadId(int tid, int3 cellDim, int3 offset) {
  int3 celli;
  const int3 cd = cellDim / 2;
  celli.x = 2 * (tid % cd.x) + offset.x;
  celli.y = 2 * ((tid / cd.x) % cd.y) + offset.y;
  celli.z = 2 * (tid / (cd.x * cd.y)) + offset.z;
  if (cd.z <= 1) {
    celli.z = 0;
  }
  return celli;
}

__device__ real3 randomDisplacement(Saru &rng, real jumpSize) {
  real3 displacement;
  displacement.x = jumpSize * (real(2.0) * rng.f() - real(1.0));
  displacement.y = jumpSize * (real(2.0) * rng.f() - real(1.0));
  displacement.z = jumpSize * (real(2.0) * rng.f() - real(1.0));
  return displacement;
}

__device__ bool checkIfPositionIsInCell(real3 pos, int3 cell, Grid grid) {
  const int3 newCell = grid.getCell(pos);
  if (cell.x != newCell.x or cell.y != newCell.y or cell.z != newCell.z) {
    return false;
  }
  return true;
}

template <class CellList, class PotentialTransverser>
__device__ real computeEnergyDifference(int group_i, real4 newPos, real4 oldPos,
                                        real4 *pos, CellList cl,
                                        PotentialTransverser pot, int3 celli) {
  using Adaptor = SFINAE::TransverserAdaptor<PotentialTransverser>;
  Adaptor adaptor;
  auto quantity = Adaptor::zero(pot);
  adaptor.getInfo(pot, group_i);
  ForceEnergyVirial oldE{};
  ForceEnergyVirial newE{};
  const bool is2D = cl.grid.cellDim.z == 1;
  const int numberNeighbourCells = is2D ? 9 : 27;
  for (int i = 0; i < numberNeighbourCells; i++) {
    int3 cellj = celli;
    cellj.x += i % 3 - 1;
    cellj.y += (i / 3) % 3 - 1;
    cellj.z += is2D ? 0 : (i / 9 - 1);
    cellj = cl.grid.pbc_cell(cellj);
    const int icellj = cl.grid.getCellIndex(cellj);
    const int firstParticle = cl.cellStart[icellj] - cl.VALID_CELL;
    if (cl.cellStart[icellj] < cl.VALID_CELL) {
      continue;
    }
    const int lastParticle = cl.cellEnd[icellj];
    const int nincell = lastParticle - firstParticle;
    for (int j = firstParticle; j < nincell + firstParticle; j++) {
      auto pos_j = pos[j];
      const int group_j = cl.groupIndex[j];
      Adaptor::accumulate(pot, oldE,
                          adaptor.compute(pot, group_j, oldPos, pos_j));
      if (group_i == group_j) {
        pos_j = newPos;
      }
      Adaptor::accumulate(pot, newE,
                          adaptor.compute(pot, group_j, newPos, pos_j));
    }
  }
  return newE.energy - oldE.energy;
}

template <class CellList, class PotentialTransverser>
__global__ void MCStepKernel(PotentialTransverser pot, CellList cl, real4 *pos,
                             real3 origin, int3 subgridOffset, int triesPerCell,
                             real beta, real jumpSize, int step, int seed,
                             uint *triedCounter, uint *acceptedCounter) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto grid = cl.grid;
  const bool is2D = grid.cellDim.z == 1;
  if (tid >= grid.getNumberCells()) {
    return;
  }
  const int3 celli = computeCellFromThreadId(tid, grid.cellDim, subgridOffset);
  const int icell = grid.getCellIndex(celli);
  const int firstParticle = cl.cellStart[icell] - cl.VALID_CELL;
  if (cl.cellStart[icell] < cl.VALID_CELL) {
    return;
  }
  const int nincell = cl.cellEnd[icell] - firstParticle;
  Saru rng(seed, step, icell);
  for (int attempt = 0; attempt < triesPerCell; attempt++) {
    if (triedCounter) {
      triedCounter[icell]++;
    }
    const int i = firstParticle + int(rng.f() * nincell);
    const int group_i = cl.groupIndex[i];
    const real4 oldPos = pos[i];
    real4 newPos = oldPos + make_real4(randomDisplacement(rng, jumpSize), 0);
    if (is2D) {
      newPos.z = oldPos.z;
    }
    const bool isNewPositionInOldCell =
        checkIfPositionIsInCell(make_real3(newPos) + origin, celli, grid);
    if (not isNewPositionInOldCell) {
      continue;
    }
    const real dH =
        computeEnergyDifference(group_i, newPos, oldPos, pos, cl, pot, celli);
    // Metropolis acceptance rule
    const real Z = rng.f();
    const real acceptanceProbabilty = thrust::min(real(1.0), exp(-beta * dH));
    if (Z <= acceptanceProbabilty) {
      pos[i] = newPos;
      if (acceptedCounter) {
        acceptedCounter[icell]++;
      }
    }
  }
}
} // namespace Anderson_ns

template <class Pot>
void Anderson<Pot>::updateParticlesInSubgrid(int3 subgrid) {
  uint *triedChanges_ptr = thrust::raw_pointer_cast(triedChanges.data());
  uint *acceptedChanges_ptr = thrust::raw_pointer_cast(acceptedChanges.data());
  real beta = 1.0 / par.temperature;
  auto clData = cl->getCellList();
  auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
  int Nthreads = 128;
  int Nblocks = (grid.getNumberCells() / 8) / Nthreads +
                (((grid.getNumberCells() / 8) % Nthreads) ? 1 : 0);
  auto pot_tr = pot->getTransverser({false, true, false}, grid.box, pd);
  size_t shMemorySize =
      SFINAE::SharedMemorySizeDelegator<decltype(pot_tr)>().getSharedMemorySize(
          pot_tr);
  Anderson_ns::MCStepKernel<<<Nblocks, Nthreads, shMemorySize, st>>>(
      pot_tr, clData, sortPos_ptr, currentOrigin, subgrid, par.triesPerCell,
      beta, jumpSize, steps, seed, triedChanges_ptr, acceptedChanges_ptr);
  CudaCheckError();
}

template <class Pot> void Anderson<Pot>::resetAcceptanceCounters() {
  thrust::fill(triedChanges.begin(), triedChanges.end(), 0);
  thrust::fill(acceptedChanges.begin(), acceptedChanges.end(), 0);
}

template <class Pot> real Anderson<Pot>::sumEnergy() {
  currentOrigin = real3();
  updateListWithCurrentOrigin();
  const int numberParticles = pg->getNumberParticles();
  {
    auto energy = pd->getEnergy(access::location::gpu, access::mode::write);
    auto energyGroup = pg->getPropertyIterator(energy);
    thrust::fill(thrust::cuda::par, energyGroup, energyGroup + numberParticles,
                 real());
  }
  int Nthreads = 128;
  int Nblocks =
      numberParticles / Nthreads + ((numberParticles % Nthreads) ? 1 : 0);
  auto globalIndex = pg->getIndexIterator(access::location::gpu);
  auto tr = pot->getTransverser({false, true, false}, grid.box, pd);
  size_t shMemorySize =
      SFINAE::SharedMemorySizeDelegator<decltype(tr)>().getSharedMemorySize(tr);
  auto ni = CellList_ns::NeighbourContainer(cl->getCellList());
  NeighbourList_ns::
      transverseWithNeighbourContainer<<<Nblocks, Nthreads, shMemorySize, 0>>>(
          tr, globalIndex, ni, numberParticles);
  CudaCheckError();
  return 0;
}
} // namespace MC_NVT
} // namespace uammd
