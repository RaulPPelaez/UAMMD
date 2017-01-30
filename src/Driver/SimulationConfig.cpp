/* Raul P.Pelaez 2016- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration and constructs the simulation.
 */

#include "SimulationConfig.h"
#include<random>
#include<iomanip>


SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
  
  Timer tim; tim.tic();
  /***********************Set all the parameters needed**********************/
  /*See globals.h for a list of parameters*/
  /*If you set a parameter that the modules you use do not need, It will just be ignored. i.e. Setting gcnf.E and using VerletNVT*/
  
  gcnf.T = 1.0;
  
  gcnf.L = make_real3(128);


  gcnf.N = pow(2,20);
  gcnf.dt = 0.001;
  
  /*Remember, rcut should be according to the longest range potential in pairforces!
    i.e the biggest sigma in LJ*/
  gcnf.rcut = 2.5;//  //rcut = 2.5*sigma -> biggest sigma in the system;
  
  gcnf.nsteps1 = 1e4;
  gcnf.nsteps2 = 0;
  gcnf.print_steps = 500;
  gcnf.measure_steps = -1;


  /*Seed of the global CPU RNG, the GPU seed is selected using grng*/
  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);

  /********************************Set initial conditions*************************/
  /*Start in a lattice, see available lattices in utils.cpp*/
  pos = initLattice(gcnf.L, gcnf.N, sc); //Start in a simple cubic lattice  

  /*Call this after all parameters are set.
    and do not change any parameter afterwards*/
  /*This function initializes pos if it is not initialized yet. You can set the initial positions
    before or after calling this*/
  
  setParameters();

  /*Dont forget to upload the positions to GPU once you are done changing it!*/  
  pos.upload();
  
  /*********************************Initialize the modules*****************************/
  /*See Modules.h for a list of all available modules!*/
  /*This is the simulation construction, where you choose integrator and force evaluators*/
  
  /*Create an Interactor (AKA Force evaluator) like this, later you will have to add it to the integrator*/
  
  auto interactor = make_shared<PairForces>();

  /*FOR PAIR FORCES*/
  /*If there are many particle types, you have to specify the potential parameters as such*/
  /*Currently this feature only works for potentials with two parameters:
    the first one reescales the distance and the second is a factor to the force modulus*/
  /*For LJ this are sigma and epsilon, by default all the parameters are 1*/
  // interactor->setPotParam(0,0, make_real2(1, 1));
  // interactor->setPotParam(0,1, make_real2(1.5, 1));
  // interactor->setPotParam(1,1, make_real2(2, 1));


  /*Create an Integrator like this*/
  integrator = make_shared<VerletNVT>();
  
  /*And inform it of the Force evaluators like this*/
  /*You can add several interactors to an integrator like so*/
  integrator->addInteractor(interactor);

  
  /*Create any desired measurable modules by adding them to the measurables vector like this*/
   // measurables.push_back(/*You can measure energy coming from any source, in this case all the interactors in integrator and the integrator itself*/
   // 			make_shared<EnergyMeasure>(integrator->getInteractors(), integrator)
   // 			);

  
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  
  tim.tic();
  /***************************Start of the simulation***************************/
  /**Run nsteps1,
     passing true to this function will be considered as a relaxation run, and wont write to disk nor measure anything**/
  //run(1000, true);  
  run(gcnf.nsteps1);  

  // /*You can change the integrator at any time between calls to run*/
  // /*But remember to clear the measurables before,
  //   so you are not measuring the energy from an incorrect place!*/
  // cerr<<"\n\nChanging Integrator...\n\n"<<endl;
  // measurables.clear();  
  // /*Set a new Integrator like before*/
  //integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(D,K, LANCZOS, MATRIXFREE, 7);
  // integrator->addInteractor(interactor);
  // integrator->addInteractor(interactor2);
  
  // /*Add it to the measrables vector, which is now empty!*/
  // measurables.push_back(make_shared<EnergyMeasure>(integrator->getInteractors(), integrator));
  
  // /**Run nsteps2 with the second integrator, no relaxation**/
  // run(gcnf.nsteps2);
  
  
  /*********************************End of simulation*****************************/
  double total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;  
}
