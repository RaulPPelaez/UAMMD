/* P. Palacios-Alonso 2024
   Test of the implementation of SPHIntegrator, it will simulate a LJ gas
   using the new SPHIntegrator and a VerletNVE integrator with the SPH potential and check
   that the reuslts are the same  
 */
#include <fstream>
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include "utils/ParticleSorter.cuh"
#include "utils/container.h"
#include "utils/execution_policy.cuh"
#include <iterator>
#include<random>
#include "uammd.cuh"
#include "Integrator/SPH.cuh"
#include "Interactor/SPH.cuh"
#include "Interactor/Potential/Potential.cuh"
#include "Interactor/PairForces.cuh"
#include "utils/InitialConditions.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"
#include "Interactor/NeighbourList/CellList.cuh"


using namespace uammd;
using std::make_shared;

struct Parameters{
  real3 boxSize;
  real dt;
  int numberParticles;
  int numberSteps, printSteps;
  real restDensity;
  real support;
  real viscosity;
  real gasStiffness;
  real radius;
  
};

//Let us group the UAMMD simulation in this struct that we can pass around
struct UAMMD{
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
  Parameters par;
};

//Creates the SPH integrator
auto createIntegratorSPH(UAMMD sim){
  SPHIntegrator::Parameters par;
  par.support        = sim.par.support;
  par.viscosity      = sim.par.viscosity;
  par.gasStiffness   = sim.par.gasStiffness;
  par.restDensity    = sim.par.restDensity;
  par.dt             = sim.par.dt;
  par.box            = Box(sim.par.boxSize);
  par.initVelocities = false;
  auto sph = make_shared<SPHIntegrator>(sim.pd,  par);
  return sph;
}

//Creates a VerletNVE integrator and adds the SPH potential
auto createIntegratorVerletNVEWithSPH(UAMMD sim){
  SPH::Parameters par;
  par.support      = sim.par.support;
  par.viscosity    = sim.par.viscosity;
  par.gasStiffness = sim.par.gasStiffness;
  par.restDensity  = sim.par.restDensity;
  par.box          = Box(sim.par.boxSize);
  auto sph         = make_shared<SPH>(sim.pd, par);

  using NVE = VerletNVE;
  NVE::Parameters params;
  params.dt             = sim.par.dt;
  params.initVelocities = false;
  auto verlet = make_shared<NVE>(sim.pd,  params);
  verlet->addInteractor(sph);
  return verlet;
}

//Set the initial positons for the particles in a fcc lattice
void initializeParticles(UAMMD sim){
  auto pos = sim.pd->getPos(access::cpu, access::write);
  auto initial =  initLattice(sim.par.boxSize, sim.par.numberParticles, fcc);
  std::copy(initial.begin(), initial.end(), pos.begin());
}

//Creates the LJ interactor
auto createLJInteractor(UAMMD sim){
  auto pot = make_shared<Potential::LJ>();
  Potential::LJ::InputPairParameters par;
  par.epsilon = 1.0;
  par.shift = false;
  par.sigma  = 2*sim.par.radius;
  par.cutOff = 2.5*par.sigma;
  pot->setPotParameters(0, 0, par);

  using PairForces  = PairForces<Potential::LJ, VerletList>;
  PairForces::Parameters params;
  params.box = Box(sim.par.boxSize);  //Box to work on
  auto pairforceslj = make_shared<PairForces>(sim.pd, params, pot);
  return pairforceslj;
}

bool compareReal3(real3 v1, real3 v2){
  return v1.x == v2.x and v1.y == v2.y and v1.z == v2.z;
}
  
bool checkSamePositions(UAMMD sim1, UAMMD sim2){
  auto pos1 = sim1.pd->getPos(access::cpu, access::read);
  auto pos2 = sim2.pd->getPos(access::cpu, access::read);
  bool samePositions = true;
  fori(0, pos1.size()){
    samePositions = compareReal3(make_real3(pos1[i]), make_real3(pos2[i]));
  }
  return samePositions;  
}

TEST(SPH, PositionsConsistentAcrossMethods){
  auto sys = make_shared<System>();
  Parameters par;
  par.support         = 2.4;   //Cut off distance for the SPH kernel
  par.viscosity       = 1.0;   //Environment viscosity
  par.gasStiffness    = 1.0;
  par.restDensity     = 1.0;
  par.radius          = 1.0;
  par.boxSize         = make_real3(32,32,32);
  par.dt              = 0.001;
  par.numberParticles = 1000;
  par.numberSteps     = 10000;
  par.printSteps      = 100;
  //Prepare the simulation that runs the VerletNVE integrator with SPH potential
  UAMMD sim1;
  sim1.sys = sys;
  sim1.par = par;
  sim1.pd = make_shared<ParticleData>(par.numberParticles, sim1.sys);
  initializeParticles(sim1);
  auto integrator1 = createIntegratorVerletNVEWithSPH(sim1);
  auto lj1         = createLJInteractor(sim1);
  integrator1->addInteractor(lj1);

  //Prepare the simulation that runs the SPH integrator
  UAMMD sim2;
  sim2.sys = sys;
  sim2.par = par;  
  sim2.pd = make_shared<ParticleData>(sim2.par.numberParticles, sim2.sys);  
  initializeParticles(sim2);  
  auto integrator2 = createIntegratorSPH(sim2);
  auto lj2         = createLJInteractor(sim2);  
  integrator2->addInteractor(lj2);
  //Run both simulations and check they move the particles in the same way
  fori(0, par.numberSteps){
    integrator1->forwardTime();
    integrator2->forwardTime();
    if(i%par.printSteps == 0){
      ASSERT_TRUE(checkSamePositions(sim1, sim2));
    }
  }
  sys->finish();
}
