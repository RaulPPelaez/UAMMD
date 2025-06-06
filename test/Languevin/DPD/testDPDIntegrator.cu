/* P. Palacios-Alonso 2024
   Test of the implementation of DPDIntegrator, it will simulate a LJ gas
   using the new DPDIntegrator and a VerletNVE integrator with the DPD potential and check
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
#include "Integrator/DPD.cuh"
#include "Interactor/Potential/Potential.cuh"
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
  real temperature, friction;
  real strength;
  real cutOff;
  real radius;
};

//Let us group the UAMMD simulation in this struct that we can pass around
struct UAMMD{
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
  Parameters par;
};

//Creates the DPD integrator
auto createIntegratorDPD(UAMMD sim){
  DPDIntegrator::Parameters par;
  par.temperature = sim.par.temperature;
  par.cutOff = sim.par.cutOff;
  par.gamma = sim.par.friction;
  par.A = sim.par.strength;
  par.dt = sim.par.dt;
  par.box = Box(sim.par.boxSize);
  auto dpd = make_shared<DPDIntegrator>(sim.pd,  par);
  return dpd;
}

//Creates a VerletNVE integrator and adds the DPD potential
auto createIntegratorVerletNVEWithDPD(UAMMD sim){
  Potential::DPD::Parameters par;
  par.temperature = sim.par.temperature;
  par.cutOff = sim.par.cutOff;
  par.gamma = sim.par.friction;
  par.A = sim.par.strength;
  par.dt = sim.par.dt;
  auto dpd = make_shared<Potential::DPD>(par);

  using PFDPD = PairForces<Potential::DPD>;
  typename PFDPD::Parameters paramsdpd;
  paramsdpd.box = Box(sim.par.boxSize);
  auto pairforces = make_shared<PFDPD>(sim.pd, paramsdpd, dpd);

  using NVE = VerletNVE;
  NVE::Parameters params;
  params.dt = par.dt;
  params.initVelocities=false;
  auto verlet = make_shared<NVE>(sim.pd,  params);
  verlet->addInteractor(pairforces);
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

TEST(DPD, PositionsConsistentAcrossMethods){
  auto sys = make_shared<System>();
  Parameters par;
  par.temperature     = 1.25;
  par.radius          = 1;
  par.cutOff          = 2*2.5*par.radius;
  par.friction        = 1.3;
  par.strength        = 1.5;
  par.dt              = 0.001;
  par.boxSize         = make_real3(25,25,25);
  par.numberParticles = 1000;
  par.numberSteps     = 100;


  //Prepare the simulation that runs the VerletNVE integrator with DPD potential
  UAMMD sim1;
  sim1.sys = sys;
  sim1.par = par;
  sim1.pd = make_shared<ParticleData>(par.numberParticles, sim1.sys);
  initializeParticles(sim1);
  auto integrator1 = createIntegratorVerletNVEWithDPD(sim1);
  auto lj1         = createLJInteractor(sim1);
  integrator1->addInteractor(lj1);

  //Prepare the simulation that runs the DPD integrator
  UAMMD sim2;
  sim2.sys = sys;
  sim2.par = par;  
  sim2.pd = make_shared<ParticleData>(sim2.par.numberParticles, sim2.sys);  
  initializeParticles(sim2);  
  auto integrator2 = createIntegratorDPD(sim2);
  auto lj2         = createLJInteractor(sim2);  
  integrator2->addInteractor(lj2);

  //Run both simulations and check they move the particles in the same way
  fori(0, par.numberSteps){
    integrator1->forwardTime();
    integrator2->forwardTime();
    ASSERT_TRUE(checkSamePositions(sim1, sim2));
  }
  sys->finish();
}
