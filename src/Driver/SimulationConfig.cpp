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
  gcnf.L = make_float3(96.0f, 16.0f, 16.0f);
  gcnf.dt = 0.01f;
  
  gcnf.rcut = 2.5f;// 1.12246204830937f; //WCA //rcut = 2.5*sigma -> biggest sigma in the system
  
  gcnf.nsteps1 = 40000;
  gcnf.nsteps2 = 50000;
  gcnf.print_steps = 500;
  gcnf.measure_steps = 100;
  
  
  gcnf.seed = 0xffaffbfDEADBULL;


  /********************************Set initial conditions*******************/
  ///*Read a configuration from file*/
  // pos = readFile("init.particles");
  // gcnf.N = pos.size();

  /*Start in a cubic lattice*/
  pos = cubicLattice(gcnf.L, gcnf.N);

  cerr<<"Box size: "<<gcnf.L<<endl;
  cerr<<"Number of particles: "<<gcnf.N<<endl;

  /*Call this after all parameters are set.
    and do not change any parameter afterwards*/
  /*This function initializes pos if it is not initialized yet. You can set the initial positions
    before or after calling this*/  
  setParameters();

  /*Dont forget to upload the positions to GPU!*/
  pos.upload();
  /*********************************Initialize the modules*****************************/
  /*See Driver.h for a list of all available modules!*/
  /*This is the simulation construction, where you choose integrator and force evaluators*/
  auto interactor = make_shared<PairForces>();
  
  integrator = make_shared<VerletNVE>();
  /*You can add several interactors to an integrator as such*/
  integrator->addInteractor(interactor);
  
  measurables.push_back(// /*You can measure energy coming from any source, in this case all the interactors in integrator*/
	       make_shared<EnergyMeasure>(integrator->getInteractors(), integrator)
               );
  
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  
  tim.tic();  
  /***************************Start of the simulation***************************/

  /**Run nsteps1 relaxation steps, no writing to disk, no measuring**/
  run(gcnf.nsteps1); //Relaxation, without printing or measuring
  
  /*You can change the integrator at any time between calls to run*/
  /*But remember to clear the measurables before,
    so you are not measuring the energy from an incorrect place!*/
  cerr<<"\n\nChanging Integrator...\n\n"<<endl;
  measurables.clear();

  /*Set a new Integrator like before*/
  integrator = make_shared<VerletNVT>();
  integrator->addInteractor(interactor);

  /*Add it to the measrables vector, which is now empty!*/
  measurables.push_back(make_shared<EnergyMeasure>(integrator->getInteractors(), integrator));
  
  /**Run nsteps2 with the second integrator, no relaxation**/
  run(gcnf.nsteps2);
  
  
  
  
  /*********************************End of simulation*****************************/
  float total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(float)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nTotal time: "<<setprecision(5)<<total_time<<"s"<<endl;  
}
