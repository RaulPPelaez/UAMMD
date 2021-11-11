/* Raul P. Pelaez 2021

   This code contains a collection of functions that create and return an instance of the different UAMMD integrators.
   The code in itself does not do much, rather it is intended  to serve as a copy pastable collection of snippets.

 */

#include<uammd.cuh>


using namespace uammd;

//I like to place these basic UAMMD objects in a struct so it is easy to pass them around
struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<System> sys;
  //Parameters par; //For this example parameters will be hardcoded
};

//Creates and returns a UAMMD struct with the basics that we have covered in previous tutorials
UAMMD initializeUAMMD(int argc, char *argv[], int numberParticles){
  UAMMD sim;
  //Initialize System and ParticleData
  sim.sys = std::make_shared<System>(argc, argv);
  sim.pd = std::make_shared<ParticleData>(sim.sys, numberParticles);
  return sim;
}

#include<Integrator/BrownianDynamics.cuh>
//There are several BD integrators
using BDMethod = BD::EulerMaruyama;
// using BDMethod = BD::MidPoint;
// using BDMethod = BD::AdamsBashforth;
// using BDMethod = BD::Leimkuhler;
std::shared_ptr<Integrator> createIntegratorBD(UAMMD sim){
  typename BDMethod::Parameters par;
  par.temperature = 1.0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0; //Self diffusion coefficient will be D = T*M = T/(6*pi*vis*hydrodynamicRadius)
  par.dt = 0.001;
  //Optionally you can place a shear matrix, dX = M*F*dt + sqrt(2*D*dt)*dW + K*R
  //par.K = {{1,2,3},{1,2,3},{1,2,3}};
  //or
  //par.K[0] = {1,2,3};
  //or
  //par.K[1].x = 1;
  //All K elements start being zero.
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<BDMethod>(sim.pd, pg, sim.sys, par);
}

#include "Integrator/VerletNVT.cuh"
using Verlet = VerletNVT::GronbechJensen;
//using Verlet = VerletNVT::Basic; //A velocity reescaling algorithm
std::shared_ptr<Integrator> createIntegratorVerletNVT(UAMMD sim){
  typename Verlet::Parameters par;
  par.temperature = 1.0;
  par.friction = 1.0;
  par.dt = 0.1;
  //If present, all particles will have this mass, otherwise the individual particle masses in ParticleData will be used
  //If those masses have not been set then the default mass is 1.0.
  //par.mass = 1.0;
  //If set to false particle velocities will be left untouched during initialization
  //If true (default) velocities will be sampled from the equilibrium configuration
  //par.initVelocities = false;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<Verlet>(sim.pd, pg, sim.sys, par);
}

#include "Integrator/VerletNVE.cuh"
std::shared_ptr<Integrator> createIntegratorVerletNVE(UAMMD sim){
  typename VerletNVE::Parameters par;
  par.dt = 0.1;
  par.energy = 1; //Optionally a target energy can be passed that VerletNVE will set according to velocities keep constant
  //par.initVelocities = false; //If true, velocities will be initialized by the module to ensure the desired energy
  //Note that it does not make sense to pass an energy and prevent VerletNVE from initializing velocities to match it.
  //If present, all particles will have this mass, otherwise the individual particle masses in ParticleData will be used
  //If those masses have not been set then the default mass is 1.0.
  //par.mass = 1.0;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<VerletNVE>(sim.pd, pg, sim.sys, par);
}

#include "Integrator/VerletNVE.cuh"
#include "Interactor/Potential/DPD.cuh"
#include"Interactor/PairForces.cuh"
//DPD is handled by UAMMD as a VerletNVE integrator with a special short range interaction
std::shared_ptr<Integrator> createIntegratorDPD(UAMMD sim){
  using NVE = VerletNVE;
  NVE::Parameters par;
  par.dt = 1.0;
  par.initVelocities = false;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  auto verlet = std::make_shared<NVE>(sim.pd, pg, sim.sys, par);
  using DPD = PairForces<Potential::DPD>;
  Potential::DPD::Parameters dpd_params;
  dpd_params.cutOff = 1.0;
  dpd_params.temperature = 1.0;
  dpd_params.gamma = 1.0;
  dpd_params.A = 1.0;
  dpd_params.dt = 0.1;
  auto pot = std::make_shared<Potential::DPD>(sim.sys, dpd_params);
  DPD::Parameters params;
  real3 L = make_real3(32,32,32);
  params.box = Box(L);
  auto pairforces = std::make_shared<DPD>(sim.pd, pg, sim.sys, params, pot);
  verlet->addInteractor(pairforces);
  return verlet;
}

#include "Integrator/VerletNVE.cuh"
#include "Interactor/SPH.cuh"
//SPH is handled by UAMMD as a VerletNVE integrator with a special interaction
std::shared_ptr<Integrator> createIntegratorSPH(UAMMD sim){
  using NVE = VerletNVE;
  NVE::Parameters par;
  par.dt = 0.1;
  par.initVelocities = false;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  auto verlet = std::make_shared<NVE>(sim.pd, pg, sim.sys, par);
  SPH::Parameters params;
  real3 L = make_real3(32,32,32);
  params.box = Box(L);
  //Pressure for a given particle "i" in SPH will be computed as gasStiffness·(density_i - restDensity)
  //Where density is computed as a function of the masses of the surroinding particles
  //Particle mass starts as 1, but you can change this in customizations.cuh
  params.support = 2.4;   //Cut off distance for the SPH kernel
  params.viscosity = 1.0;   //Environment viscosity
  params.gasStiffness = 1.0;
  params.restDensity = 1.0;
  auto sph = std::make_shared<SPH>(sim.pd, pg, sim.sys, params);
  verlet->addInteractor(sph);
  return verlet;
}

#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_PSE.cuh"
#include"Integrator/BDHI/BDHI_FCM.cuh"
//Creates a triply periodic Brownian Dynamics with Hydrodynamic Interactions integration module
std::shared_ptr<Integrator> createIntegratorBDHI(UAMMD sim){
  //There are several hydrodynamics modules, we choose between Positively Split Ewald (PSE) or Force Coupling Method (FCM) here
  // mainly for performance reasons. FCM is faster for small and/or dense systems, but it is limited in the system size by memory.
  // PSE can be slower it temperature>0, but does not have that system size constraints.
  //FCM scales linearly with system size (so doubling the box size in the three dimensions makes it 8 times slower) and number of particles
  //PSE scales linearly with the number of particles, independently of system size. But the "psi" parameter must be tweaked to find the optimal performance for each case.
  //See the wiki for more information about these modules
  real3 L = make_real3(32,32,32);
  real hydrodynamicRadius = 1.0;
  real maxL = std::max({L.x, L.y, L.z});
  int maxcells = maxL/hydrodynamicRadius;
  //In both modules, particle self diffusion coefficient will be T/(6*pi*viscosity*hydrodynamicRadius) or close to it
  if(maxcells >= 128){
    using Scheme = BDHI::PSE;
    Scheme::Parameters par;
    par.box = Box(L);
    par.temperature = 1.0;
    par.viscosity = 1.0;
    par.dt = 0.1;
    par.hydrodynamicRadius = hydrodynamicRadius;
    par.tolerance = 1e-4;
    //Balances the load of the algorithm, low values work best for dilute and/or big systems.
    // Higher values will work best for dense and/or small systems.
    par.psi = 1.0/par.hydrodynamicRadius;
    auto bdhi = std::make_shared<BDHI::EulerMaruyama<Scheme>>(sim.pd, sim.sys, par);
    return bdhi;
  }
  else{
    using Scheme = BDHI::FCM;
    Scheme::Parameters par;
    par.box = Box(L);
    par.temperature = 1.0;
    par.viscosity = 1.0;
    par.dt = 0.1;
    par.hydrodynamicRadius = hydrodynamicRadius;
    par.tolerance = 1e-4;
    auto bdhi = std::make_shared<BDHI::EulerMaruyama<Scheme>>(sim.pd, sim.sys, par);
    return bdhi;
  }
}


std::shared_ptr<Integrator> createIntegratorFCM(UAMMD sim){
  //The FCM module also works as an standalone Integrator.
  //In this mode, FCM can also compute angular displacements due to torques acting on the particles
  //See the wiki for more information about these modules
  real3 L = make_real3(32,32,32);
  real hydrodynamicRadius = 1.0;
  BDHI::FCMIntegrator::Parameters par;
  par.box = Box(L);
  par.temperature = 1.0;
  par.viscosity = 1.0;
  par.dt = 0.1;
  par.hydrodynamicRadius = hydrodynamicRadius;
  par.tolerance = 1e-4;
  auto bdhi = std::make_shared<BDHI::FCMIntegrator>(sim.pd, par);
  return bdhi;
}
int main(int argc, char* argv[]){
  int N = 16384;
  auto sim = initializeUAMMD(argc, argv, N);
  auto integrator = createIntegratorBD(sim);
  for(int i= 0; i<100; i++){
    integrator->forwardTime();
  }
  sim.sys->finish();
  return 0;
}
