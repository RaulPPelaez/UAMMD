#include "SimulationConfig.h"
#include<random>
#include<iomanip>
SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
  Timer tim; tim.tic();
  /***********************Set all the parameters needed**********************/
  gcnf.E = 0.0f;
  gcnf.T = 0.01f;
  gcnf.gamma = 1.0f;
  
  
  gcnf.N = pow(2,14);
  //float dens = 1; //stod(argv[2], NULL);
  gcnf.L = 32;//cbrt(gcnf.N/dens);
  gcnf.dt = 0.001f;
  
  gcnf.rcut = 2.5f;// 1.12246204830937f; //WCA //rcut = 2.5*sigma -> biggest sigma in the system
  
  gcnf.nsteps1 = 5000;
  gcnf.nsteps2 = 5000;
  gcnf.print_steps = 100;
  gcnf.measure_steps = -1;
  
  gcnf.relaxation_steps = -1;
  
  gcnf.seed = 0xffaffbfDEADBULL;
  
  ///*Read a configuration from file*/
  // pos = readFile("init.particles");
  // gcnf.N = pos.size();
  
  /*Call this after all parameters are set.
    Do not modify pos, force or vel before calling this function, 
    unless when using readFile
    and do not change any parameter afterwards*/
  setParameters();
  
  cerr<<"Box size: "<<gcnf.L<<endl;
  cerr<<"Number of particles: "<<gcnf.N<<endl;
  
  /********************************Set initial conditions*******************/
  /*Start in a cubic lattice*/
  cubicLattice(pos.data, gcnf.L, gcnf.N);
  //std::random_shuffle(pos.begin(), pos.end());
  pos.upload();
  
  /*********************************Initialize the modules*****************************/  
  /*This is the simulation construction, where you choose integrator and force evaluators*/
  auto interactor = make_shared<PairForces>();
  //auto interactor2 =  make_shared<ExternalForces>();
  //auto interactor3 = make_shared<ThreeBondedForces>("agar.3bonds");
  //auto interactor4 = make_shared<BondedForces>("kk.bonds");
  
  // Matrixf D(3,3), K(3,3);
  
  // D.fill_with(0.0f);
  // fori(0,3) D[i][i] = gcnf.gamma*gcnf.T;
  // K.fill_with(0.0f);
  
  //integrator = make_shared<BrownianEulerMaruyama>(D, K);
  integrator = make_shared<VerletNVE>();
  /*You can add several interactors to an integrator as such*/
  integrator->addInteractor(interactor);
  //integrator->addInteractor(interactor2);
  //integrator->addInteractor(interactor3);
  //integrator->addInteractor(interactor4);
  measurables.push_back(// /*You can measure energy coming from any source, in this case all the interactors in integrator*/
        make_shared<EnergyMeasure>(integrator->getInteractors(), integrator)
       );
  
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  

  tim.tic();  
  /***************************Start of the simulation***************************/


  run(gcnf.nsteps1, true); //Relaxation, without printing or measuring

  cerr<<"\n\nChanging Integrator...\n\n"<<endl;
  integrator = make_shared<VerletNVT>();
  integrator->addInteractor(interactor);

  
  run(gcnf.nsteps2);

  




  
  /*********************************End of simulation*****************************/
  float total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(float)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nTotal time: "<<setprecision(5)<<total_time<<"s"<<endl;
  
}
