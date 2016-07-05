#include "SimulationConfig.h"

SimulationConfig::SimulationConfig(): Driver(){

  gcnf.T = 0.0f;
  gcnf.L = 64;
  gcnf.N = pow(2,14);
  gcnf.dt = 0.01f;
  
  gcnf.nsteps = 10000;
  gcnf.print_steps = 100;
  
  /*Start in a cubic lattice*/
  cubicLattice(pos.data, 32, gcnf.N);
  pos.upload();
     
  /****Initialize the modules*******/
  /*This is the simulation construction, where you choose integrator and force evaluators*/
  auto interactor =  make_shared<PairForces>(LJ);
  //auto interactor2 =  make_shared<ExternalForces>();
  //auto interactor3 = make_shared<BondedForces>("one.bond");

  //integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>();
  //integrator = make_shared<VerletNVE>();
  integrator = make_shared<VerletNVT>();

  /*You can add several interactors to an integrator as such*/
  integrator->addInteractor(interactor);
  //integrator->addInteractor(interactor2);
  //integrator->addInteractor(interactor3);
  
  //measurables.push_back(make_shared<EnergyMeasure>(integrator->getInteractors(), integrator, N, L));
}


