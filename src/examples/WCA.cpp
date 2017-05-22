/* Raul P.Pelaez 2016- Simulation Config

      Implementation of the SimulationConfig class.

      The constructor of this class sets all the parameters, creates the initial configuration, constructs and runs the simulation.


      A system of WCA particles evolving with a NVT MD integrator.
      The particles are trapped in a box in the center via an external force called HarmonicTrap.


      References:
      [1] https://github.com/RaulPPelaez/UAMMD/wiki
      [2] https://github.com/RaulPPelaez/UAMMD/wiki/PairForces 
      [3] https://github.com/RaulPPelaez/UAMMD/wiki/ExternalForces 

*/

#include "SimulationConfig.h"

/*This is a functor to be used with the ExternalForces module. See [1]
  The operator (pos, i) will be called for each particle, and will have available
  any private member.
  The result will be added to the global force array: force[i] = tr(pos, i);
  It is intended for computing a quantity for each particle that depends only on information about that particle.
 */
struct HarmonicTrap{
  HarmonicTrap(real3 L, real k): L(L), k(k){}
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
  /*See globals.h for a list of global parameters*/
  /*Most modules will get their parameters from there if no arguments are supplied to their constructors*/
  /*If you set a parameter that the modules you use do not need, It will just be ignored. i.e. Setting gcnf.E and using VerletNVT*/
  
  gcnf.T = 1.0;

  gcnf.L = make_real3(64);
 

  gcnf.N = pow(2,14);

  /*Writing is done in parallel, 
    so its better to have a low dt and a large print_steps that the other way around 
    (when print_steps is so low that it starts to decrease performance)*/
  gcnf.dt = 0.0001;


  gcnf.nsteps1 = 1e5;
  gcnf.print_steps = 1000;  


  gcnf.seed = 0xffaffbfDEADBULL;
  pos = initLattice(gcnf.L/2, gcnf.N, fcc); //Start in a square lattice

  setParameters();


  /*Upload positions to GPU once the initial conditions are set*/
  pos.upload();



  /*Termalization integrator*/
  integrator = make_shared<VerletNVT>();

  /*Short range forces, with LJ potential if other is not provided*/
  auto lj = make_shared<PairForces<CellList, Potential::LJ>>(pow(2, 1/6.0f));
  lj->setPotParams(0,0,
		   {1 /*epsilon*/, 1 /*sigma*/, powf(2, 1/6.) /*rcut*/, false /*shift?*/});

  integrator->addInteractor(lj);
  
  /*ExternalForces module specialized for the HarmonicTrap above*/
  auto harmonic_trap = make_shared<ExternalForces<HarmonicTrap>>(HarmonicTrap(gcnf.L/2.0f, 0.5f /*Kspring*/));
  /*Add it to the integrator*/
  integrator->addInteractor(harmonic_trap);


  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;

  tim.tic();
  /***************************Start of the simulation***************************/
  /**Run nsteps1,
     passing true to this function will be considered as a relaxation run, and wont write to disk nor measure anything**/
  /*Run termalization*/
  run(gcnf.nsteps1); 
   
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

