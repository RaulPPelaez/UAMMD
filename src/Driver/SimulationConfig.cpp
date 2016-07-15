#include "SimulationConfig.h"

SimulationConfig::SimulationConfig(): Driver(){

  gcnf.E = 0.0f;
  gcnf.T = 10.0f;
  gcnf.L = 128;

  gcnf.dt = 0.01f;
  
  //gcnf.rcut = 1.12246204830937f; //WCA
  
  gcnf.nsteps = 100;
  gcnf.print_steps = -1;
  gcnf.measure_steps = -1;
  gcnf.relaxation_steps = 0;


  readFile(pos, "gluco.pos");

  gcnf.N = pos.size();
  uint N = gcnf.N;

  fori(0,N){
    pos[i].x -= 32;
  }
  // pos = Vector4(N, true);
  // pos.fill_with(make_float4(0.0f));

  
  
  force = Vector4(N);
  force.fill_with(make_float4(0.0f));
  force.upload();
  
  /*Start in a cubic lattice*/
  //cubicLattice(pos.data, 32, gcnf.N);  
  pos.upload();
  
  /****Initialize the modules*******/
  /*This is the simulation construction, where you choose integrator and force evaluators*/
  //auto interactor =  make_shared<PairForces>(LJ);
  //auto interactor2 =  make_shared<ExternalForces>();
  auto interactor3 = make_shared<BondedForces>("gluco.bond");

  //integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>();
  //integrator = make_shared<VerletNVE>();
  integrator = make_shared<VerletNVT>();
  /*You can add several interactors to an integrator as such*/
  //integrator->addInteractor(interactor);
  //integrator->addInteractor(interactor2);
  integrator->addInteractor(interactor3);


  measurables.push_back(make_shared<EnergyMeasure>(integrator->getInteractors(), integrator, gcnf.N, gcnf.L));


}


