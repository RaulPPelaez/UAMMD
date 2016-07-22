#include "SimulationConfig.h"

SimulationConfig::SimulationConfig(): Driver(){

  gcnf.E = 0.0f;
  gcnf.T = 0.2f;
  gcnf.L = 64;

  gcnf.N = 1000;
  gcnf.dt = 0.01f;
  
  gcnf.rcut = 2.5f;// 1.12246204830937f; //WCA
  
  gcnf.nsteps = 10000;
  gcnf.print_steps = 500;
  gcnf.measure_steps = -1;
  gcnf.relaxation_steps = 0;


  gcnf.seed = 13446714043708551611ULL;

  //readFile(pos, "glucos.pos");

  //  gcnf.N = pos.size();
  uint N = gcnf.N;

  // fori(0,N){
  //   pos[i].x -= 32;
  // }
  
  pos = Vector4(N);
  pos.fill_with(make_float4(0.0f));

  
  
  force = Vector4(N);
  force.fill_with(make_float4(0.0f));
  force.upload();
  
  /*Start in a cubic lattice*/
  cubicLattice(pos.data, 32, gcnf.N);
  //  std::random_shuffle(pos.begin(), pos.end());
  pos.upload();

  /****Initialize the modules*******/
  /*This is the simulation construction, where you choose integrator and force evaluators*/
  //auto interactor =  make_shared<PairForcesDPD>();
  //auto interactor2 =  make_shared<ExternalForces>();
  //auto interactor3 = make_shared<BondedForces>("glucos.bond");

  Vector4 D(4), K;
  D.fill_with(make_float4(0.0f));
  K = D;

  D[0].x= 1.0f;
  D[1].y= 1.0f;
  D[2].z= 1.0f;
  
  integrator = make_shared<BrownianEulerMaruyama>(D, K);
  //integrator = make_shared<VerletNVE>();
  //integrator = make_shared<VerletNVE>();
  /*You can add several interactors to an integrator as such*/
  //integrator->addInteractor(interactor);
  //integrator->addInteractor(interactor2);
  //integrator->addInteractor(interactor3);


  measurables.push_back(make_shared<EnergyMeasure>(integrator->getInteractors(), integrator, gcnf.N, gcnf.L));


}


