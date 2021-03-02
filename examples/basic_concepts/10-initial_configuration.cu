/* Raul P. Pelaez 2021
   Initializing positions.
   You can generate the initial state of the particles by reading it from a file or making it random, but in this tutorial
    we will learn about the InitialConditions.cuh header, which provices a convenient function that places positions in a lattice.

 */

#include<uammd.cuh>
#include"utils/InitialConditions.cuh"

using namespace uammd;

//I like to place these basic UAMMD objects in a struct so it is easy to pass them around
struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<System> sys;
};

//Creates and returns a UAMMD struct with the basics that we have covered in previous tutorials
UAMMD initializeUAMMD(int argc, char *argv[]){
  UAMMD sim;
  //Initialize System and ParticleData
  sim.sys = std::make_shared<System>(argc, argv);
  constexpr int numberParticles = 16384; //Lets just hardcode a number of particles
  sim.pd = std::make_shared<ParticleData>(sim.sys, numberParticles);
  return sim;
}

//This function will place ParticleData positions in a lattice 
void initializePositionsInALattice(UAMMD sim){
  //initLattice will return a vector with numberParticles positions inside a box of size L following a certain lattice.
  //In this case we chose an FCC lattice, but there are others, as sc (simple cubic), bcc, hcp,...
  real3 L = make_real3(32,32,32); //Just an arbitrary box size
  int numberParticles = sim.pd->getNumParticles();
  auto lattice_positions = initLattice(L, numberParticles, fcc);
  //We simply copy it to ParticleData positions
  auto pos = sim.pd->getPos(access::cpu, access::write);
  std::copy(lattice_positions.begin(), lattice_positions.end(), pos.begin());
}


int main(int argc, char* argv[]){
  //Lets initialize UAMMD as always
  auto sim = initializeUAMMD(argc, argv);
  //And then initialize positions in a lattice
  initializePositionsInALattice(sim);
  //Doing this might be useful to start simulations with hard potentials that cannot handle overlapping particles
  //Destroy the UAMMD environment and exit
  sim.sys->finish();
  return 0;
}
