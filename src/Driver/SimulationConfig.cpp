/* Raul P.Pelaez 2016- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration, constructs and runs the simulation.   


   Currently configured to create a Brownian Dynamics with hydrodynamic interactions using the Cholesky decomposition. With ideal particles trapped in a harmonic box.

References:
[1] https://github.com/RaulPPelaez/UAMMD/wiki
[2] https://github.com/RaulPPelaez/UAMMD/wiki/External-Forces 
 */

#include "SimulationConfig.h"

/*This is a functor to be used with the ExternalForces module. See [1]
  The operator (pos, i) will be called for each particle, and will have available
  any private member.
  The result will be added to the global force array: force[i] = tr(pos, i);
  It is intended for computing a quantity for each particle that depends only on information about that particle.
 */
struct Externaltor{
  Externaltor(real3 L, real k): L(L), k(k){}
  inline __device__ real4 operator()(const real4 &pos, int i){
    real4 f = make_real4(0);

    /*A simple harmonic wall*/
    real *F = &(f.x);
    const real *r = &(pos.x);
    for(int i=0; i<3; i++){
      if(r[i]> L.x*0.5f)
     	F[i] -= k*(r[i]-L.x*0.5f);
      else if(r[i]<-L.x*0.5f)
     	F[i] -= k*(r[i]+L.x*0.5f);
    }                                 

    return f;
  }
  real k;
  real3 L;
};
  
SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){    
  Timer tim; tim.tic();
  /***********************Set all the parameters needed**********************/
  /*See globals.h for a list of parameters*/
  /*If you set a parameter that the modules you use do not need, It will just be ignored. i.e. Setting gcnf.E and using VerletNVT*/

  gcnf.T = 0.5;
  
  gcnf.L = make_real3(32, 32, 0);

  
  gcnf.N = pow(2,10);
  gcnf.dt = 0.05;
  
    
  gcnf.nsteps1 = 74000;
  gcnf.print_steps = 200;
  gcnf.measure_steps = -1;


  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);
  //pos = initLattice(gcnf.L, gcnf.N, sc); //Start in a simple cubic lattice
  //pos = readFile("armadillo.inipos");
  
  
  setParameters();
  
  //fori(0,gcnf.N) pos[i] = make_real4(make_real3(pos[i])+10, 0);
  
  /*A random initial configuration*/
  fori(0,gcnf.N){
    pos[i] = make_real4(grng.uniform3(-gcnf.L.x/2.0, gcnf.L.x/2.0), 0);
    pos[i].z = 0;
  }

  /*Upload positions to GPU once the initial conditions are set*/
  pos.upload();
  
  //integrator = make_shared<VerletNVT>();
  real vis = 1;
  real rh = 1.0;
  Matrixf K(3,3);
  K.fill_with(0);
  //int niter = 4; //Number of Lanczos iterations
  integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(K, vis, rh, CHOLESKY);//PSE); //LANCZOS, niter);

  /*Short range forces, with LJ potential if other is not provided*/
  // auto interactor = make_shared<PairForces<CellList>>();

  /*ExternalForces module specialized for the ExternalTor above*/
  Externaltor tr(gcnf.L, 50.0f);
  auto interactor = make_shared<ExternalForces<Externaltor>>(tr);

  /*Add it to the integrator*/
  integrator->addInteractor(interactor);
  
  //integrator->addInteractor(interactor2);
  //integrator->addInteractor(interactor3);
  
  
  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  
  tim.tic();
  /***************************Start of the simulation***************************/
  /**Run nsteps1,
     passing true to this function will be considered as a relaxation run, and wont write to disk nor measure anything**/
  //  run(1000, true);  
  run(gcnf.nsteps1);  

  // /*You can change the integrator at any time between calls to run*/
  // /*But remember to clear the measurables before,
  //   so you are not measuring the energy from an incorrect place!*/
  // cerr<<"\n\nChanging Integrator...\n\n"<<endl;
  // measurables.clear();  
  // /*Set a new Integrator like before*/
  // integrator = make_shared<VerletNVT>();
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
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<1.0f<<" "<<type<<"\n";
  }
  cout<<flush;
}
