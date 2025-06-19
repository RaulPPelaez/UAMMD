/*Raul P. Pelaez 2019-2021. NeighbourContainer interface example.

  This file computes a LJ liquid simulation, similar to benchmark.cu.

  The difference is that instead of using PairForces, this code gets a
  NeighbourContainer from CellList and processes the neighbours manually (see
  processNeighbours kernel).

  It is an example on how to interface with a NeighbourList without writing a
  Transverser and without necesarily constructing an explicit neighbour list.

 */
#include "Interactor/NeighbourList/CellList.cuh"
#include "uammd.cuh"
// #include"Interactor/NeighbourList/VerletList.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Interactor/Interactor.cuh"
#include "utils/InitialConditions.cuh"
#include "utils/InputFile.h"
#include <fstream>

using namespace uammd;
using std::endl;
using std::make_shared;

real3 boxSize;
real dt;
std::string outputFile;
int numberParticles;
int numberSteps, printSteps;
real temperature, friction, rcut;

__device__ real lj_force(real r2) {
  const real invr2 = real(1.0) / r2;
  const real invr6 = invr2 * invr2 * invr2;
  const real fmod = (real(-48.0) * invr6 + real(24.0)) * invr6 * invr2;
  return fmod;
}

__device__ real lj_energy(real r2) {
  const real invr2 = real(1.0) / r2;
  const real invr6 = invr2 * invr2 * invr2;
  return real(4.0) * invr6 * (invr6 - real(1.0));
}

// A new way of using a neighbour list
template <class NeighbourContainer>
__global__ void processNeighbours(
    NeighbourContainer ni, // Provides iterator with neighbours of a particle
    int numberParticles, Box box,
    real4 *force, // Forces in group indexing
    real *energy, // Energies in group indexing
    real *virial  // Virial in group indexing
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numberParticles)
    return;
  // Set ni to provide iterators for particle i
  ni.set(i);
  const real3 pi =
      make_real3(cub::ThreadLoad<cub::LOAD_LDG>(ni.getSortedPositions() + i));
  real3 f = real3();
  real e = 0;
  real v = 0;
  // for(auto neigh: ni){ //This is equivalent to the while loop, although a tad
  // slower
  auto it = ni.begin(); // Iterator to the first neighbour of particle i
  // Note that ni.end() is not a pointer to the last neighbour, it just
  // represents "no more neighbours" and
  //  should not be dereferenced
  while (it) { // it will cast to false when there are no more neighbours
    auto neigh = *it++; // The iterator can only be advanced and dereferenced
    // int j = neigh.getGroupIndex();
    const real3 pj = make_real3(neigh.getPos());
    const real3 rij = box.apply_pbc(pj - pi);
    const real r2 = dot(rij, rij);
    if (r2 > 0 and r2 < (real(6.25))) {
      const real3 fmod = (force or virial) ? lj_force(r2) * rij : real3();
      if (force)
        f += fmod;
      if (energy)
        e += lj_energy(r2);
      if (virial)
        v += dot(fmod, rij);
    }
  }
  const int global_index = ni.getGroupIndexes()[i];
  if (force)
    force[global_index] += make_real4(f);
  if (energy)
    energy[global_index] += e;
  if (virial)
    virial[global_index] += v;
}

// A small interactor that computes LJ force using a CellList
class myInteractor : public Interactor {
  // Uncomment to select a different neighbour list strategy
  // using NeighbourList = VerletList;
  using NeighbourList = CellList;
  std::shared_ptr<NeighbourList> cl;

public:
  myInteractor(std::shared_ptr<ParticleData> pd) : Interactor(pd, "Custom") {
    cl = make_shared<NeighbourList>(pd);
  }

  void sum(Computables comp, cudaStream_t st) override {
    Box box(boxSize);
    cl->update(box, rcut, st);
    // NeighbourContainer can provide forward iterators with the neighbours of
    // each particle The drawback of it being a forward iterator is that it can
    // only be advanced, once you have asked for the next neighbour there is no
    // going back without starting from the first. With it=ni.begin() you can
    // only do it++, etc, there is no operator[] nor it--
    auto ni = cl->getNeighbourContainer();
    auto force =
        comp.force
            ? pd->getForce(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;
    auto energy = comp.energy ? pd->getEnergy(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    auto virial = comp.virial ? pd->getVirial(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    processNeighbours<<<numberParticles / 128 + 1, 128, 0, st>>>(
        ni, numberParticles, box, force, energy, virial);
  }
};

void readParameters(std::shared_ptr<System> sys, std::string file);

int main(int argc, char *argv[]) {
  auto sys = std::make_shared<System>(argc, argv);
  readParameters(sys, "data.main.neighbourIterator");
  auto pd = std::make_shared<ParticleData>(numberParticles, sys);
  Box box(boxSize);
  { // Initialize positions
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto initial = initLattice(box.boxSize, numberParticles, fcc);
    std::transform(initial.begin(), initial.end(), pos.begin(), [&](real4 p) {
      p.w = 0;
      return p;
    });
  }
  using NVT = VerletNVT::GronbechJensen;
  NVT::Parameters par;
  par.temperature = temperature;
  par.dt = dt;
  par.friction = friction;
  auto verlet = make_shared<NVT>(pd, par);
  auto inter = std::make_shared<myInteractor>(pd);
  verlet->addInteractor(inter);
  std::ofstream out(outputFile);
  Timer tim;
  tim.tic();
  forj(0, numberSteps) {
    verlet->forwardTime();
    if (printSteps > 0 and j % printSteps == 0) {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int *sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out << "#Lx=" << 0.5 * box.boxSize.x << ";Ly=" << 0.5 * box.boxSize.y
          << ";Lz=" << 0.5 * box.boxSize.z << ";" << endl;
      fori(0, numberParticles) {
        real4 pc = pos[sortedIndex[i]];
        real3 p = box.apply_pbc(make_real3(pc));
        out << p << " " << 0.5 << " " << 0 << "\n";
      }
    }
    if (j % 500 == 0) {
      pd->sortParticles();
    }
  }
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps / totalTime);
  sys->finish();
  return 0;
}

void generateDefaultParameters(std::string file) {
  std::ofstream default_options(file);
  default_options << "boxSize 32 32 32" << std::endl;
  default_options << "numberParticles 16384" << std::endl;
  default_options << "outputFile /dev/stdout" << std::endl;
  default_options << "rcut 2.5" << std::endl;
  default_options << "dt 0.01" << std::endl;
  default_options << "numberSteps 500" << std::endl;
  default_options << "printSteps -1" << std::endl;
  default_options << "temperature 1.0" << std::endl;
  default_options << "friction 1" << std::endl;
}

void readParameters(std::shared_ptr<System> sys, std::string file) {
  if (!std::ifstream(file).good()) {
    generateDefaultParameters(file);
  }
  InputFile in(file, sys);
  in.getOption("boxSize", InputFile::Required) >> boxSize.x >> boxSize.y >>
      boxSize.z;
  in.getOption("numberParticles", InputFile::Required) >> numberParticles;
  in.getOption("outputFile", InputFile::Required) >> outputFile;
  in.getOption("rcut", InputFile::Required) >> rcut;
  in.getOption("numberSteps", InputFile::Required) >> numberSteps;
  in.getOption("printSteps", InputFile::Required) >> printSteps;
  in.getOption("dt", InputFile::Required) >> dt;
  in.getOption("temperature", InputFile::Required) >> temperature;
  in.getOption("friction", InputFile::Required) >> friction;
}
