/* Raul P.Pelaez 2016- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration, constructs the simulation and runs it.
 */

#include "SimulationConfig.h"
#include<random>
#include<iomanip>


SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
  
  Timer tim; tim.tic();
  /***********************Set all the parameters needed**********************/
  /*See globals.h for a list of parameters*/
  /*If you set a parameter that the modules you use do not need, It will just be ignored. i.e. Setting gcnf.E and using VerletNVT*/
  gcnf.E = 0.0;
  gcnf.T = 0.01;
  gcnf.gamma = 1.0;
  
  gcnf.N = pow(2,13);
  gcnf.L = make_real3(18.3);
  gcnf.dt = 0.01f;
  /*Remember, rcut should be according to the longest range potential in pairforces!
    i.e the biggest sigma in LJ*/
  gcnf.rcut = 2.5;  //rcut = 2.5*sigma -> biggest sigma in the system; pow(2, 1/6.); //WCA
  
  gcnf.nsteps1 = 400000;
  gcnf.nsteps2 = 0;
  gcnf.print_steps = 500;
  gcnf.measure_steps = 10;
  
  gcnf.seed = 0xffaffbfDEADBULL;

  /********************************Set initial conditions*************************/
  /*Read a configuration from file*/
  // pos = readFile("helix.pos_fix");
  // gcnf.N = pos.size();
  /*You can create or modify the initial configuration just by modifying the pos array*/
  // fori(0,gcnf.N){
  //   pos[i].z -= 40;
  // }
  
  /*Start in a lattice, see available lattices in utils.cpp*/
  pos = initLattice(gcnf.L, gcnf.N, fcc); //Start in a simple cubic lattice  
  
  /*Call this after all parameters are set.
    and do not change any parameter afterwards*/
  /*This function initializes pos if it is not initialized yet. You can set the initial positions
    before or after calling this*/
  
  setParameters();
  /*Set random types, 0 or 1*/
  // fori(0,gcnf.N)
  //   pos[i].w = grng.uniform(0,1)>0.15?0:1;
  
  /*Dont forget to upload the positions to GPU once you are done changing it!*/
  pos.upload();
  
  /*********************************Initialize the modules*****************************/
  /*See Driver.h for a list of all available modules!*/
  /*This is the simulation construction, where you choose integrator and force evaluators*/
  /*Create an Interactor (AKA Force evaluator) like this, later you will have to add it to the other necessary modules*/
  auto interactor = make_shared<PairForces>();
  /*If there are many particle types, you have to specify the potential parameters as such*/
  /*Currently this feature only works for potentials with two parameters:
    the first one reescales the distance and the second is a factor to the force modulus*/
  /*For LJ this are sigma and epsilon, by default all the parameters are 1*/
  // interactor->setPotParam(0,0, make_real2(1, 1));
  // interactor->setPotParam(0,1, make_real2(1.5, 1));
  // interactor->setPotParam(1,1, make_real2(2, 1));
  // Matrixf D(3,3), K(3,3);
  // D.fill_with(0.0f);
  // fori(0,3) D[i][i] = gcnf.gamma*gcnf.T;  
  // K.fill_with(0.0f);                             
    
  /*Create an Integrator like this*/
  //integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(D,K);
  integrator = make_shared<VerletNVE>();
  /*And inform it of the Force evaluators like this*/
  /*You can add several interactors to an integrator as such*/
  integrator->addInteractor(interactor);
  //integrator->addInteractor(interactor2);
  
  /*Create any desired measurable modules by adding them to the measurables vector like this*/
   measurables.push_back(/*You can measure energy coming from any source, in this case all the interactors in integrator and the integrator itself*/
   			make_shared<EnergyMeasure>(integrator->getInteractors(), integrator)
   			);

  
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  
  tim.tic();
  /***************************Start of the simulation***************************/
  /**Run nsteps1,
     passing true to this function will be considered as a relaxation run, and wont write to disk nor measure anything**/
  run(gcnf.nsteps1);
  
  // /*You can change the integrator at any time between calls to run*/
  // /*But remember to clear the measurables before,
  //   so you are not measuring the energy from an incorrect place!*/
  // cerr<<"\n\nChanging Integrator...\n\n"<<endl;
  // measurables.clear();
  
  // /*Set a new Integrator like before*/
  // integrator = make_shared<VerletNVT>();
  // integrator->addInteractor(interactor);
  
  // /*Add it to the measrables vector, which is now empty!*/
  // measurables.push_back(make_shared<EnergyMeasure>(integrator->getInteractors(), integrator));
  
  // /**Run nsteps2 with the second integrator, no relaxation**/
  // run(gcnf.nsteps2);
  
  
  /*********************************End of simulation*****************************/
  double total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;  
}
