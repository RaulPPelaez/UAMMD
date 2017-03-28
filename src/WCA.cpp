/* Raul P.Pelaez 2016- Simulation Config

      Implementation of the SimulationConfig class.

         The constructor of this class sets all the parameters, creates the initial configuration, constructs and runs the simulation.


	    Currently configured to create a Brownian Dynamics with hydrodynamic interactions using the Cholesky decomposition. With WCA particles in a periodic box (HI is in an infinite box).

	    References:
	    [1] https://github.com/RaulPPelaez/UAMMD/wiki
	    [2] https://github.com/RaulPPelaez/UAMMD/wiki/PairForces 

You can check the gdr with tools/gdr. Run it to see the input.
*/

#include "SimulationConfig.h"


SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
  Timer tim; tim.tic();
  /***********************Set all the parameters needed**********************/
  /*See globals.h for a list of global parameters*/
  /*Most modules will get their parameters from there if no arguments are supplied to their constructors*/
  /*If you set a parameter that the modules you use do not need, It will just be ignored. i.e. Setting gcnf.E and using VerletNVT*/

  gcnf.T = 1.0;

  gcnf.L = make_real3(40, 40, 0);

  gcnf.rcut = pow(2, 1/6.0); //cut off radius for WCA

  gcnf.N = pow(2,10);
  gcnf.dt = 0.005;


  gcnf.nsteps1 = 100000;
  gcnf.nsteps2 = 74000;
  gcnf.print_steps = 500;  


  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);
  pos = initLattice(gcnf.L, gcnf.N, sq); //Start in a square lattice

  setParameters();

  //fori(0,gcnf.N) pos[i] = make_real4(make_real3(pos[i])+10, 0);

  /*A random initial configuration*/
  // fori(0,gcnf.N){
  //   pos[i] = make_real4(grng.uniform3(-gcnf.L.x/2.0, gcnf.L.x/2.0), 0);
  //   pos[i].z = 0;
  // }

  /*Upload positions to GPU once the initial conditions are set*/
  pos.upload();

  /*Short range forces, with LJ potential if other is not provided*/
  auto interactor2 = make_shared<PairForces<CellList>>();


  /*Termalization integrator*/
  integrator = make_shared<VerletNVT>();
  integrator->addInteractor(interactor2);
  


  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;

  tim.tic();
  /***************************Start of the simulation***************************/
  /**Run nsteps1,
     passing true to this function will be considered as a relaxation run, and wont write to disk nor measure anything**/
  /*Run termalization*/
  run(gcnf.nsteps1, true);
  

  /*Change the integrator and run the simulation*/
  real vis = 1;
  real rh = 1.0;
  Matrixf K(3,3);
  K.fill_with(0);
  //int niter = 4; //Number of Lanczos iterations
  integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(K, vis, rh, CHOLESKY);//PSE); //LANCZOS, niter);

  integrator->addInteractor(interactor2);
  
  /**Run nsteps2 with the second integrator, no relaxation**/
  run(gcnf.nsteps2);


  /*********************************End of simulation*****************************/
  double total_time = tim.toc();

  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;

}
/*Change the output format here, in this function, the updated positions will be available 
 in CPU. Writing is done in parallel.
*/
void Writer::write_concurrent(){
  real3 L = gcnf.L;
  uint N = gcnf.N;
  real4 *posdata = pos.data;
  cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
  fori(0,N){
    uint type = (uint)(posdata[i].w+0.5);
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<.5f<<" "<<type<<"\n";
  }
  cout<<flush;
}

