/* Raul P.Pelaez 2016- Simulation Config

   Implementation of the SimulationConfig class.
   
   The constructor of this class sets all the parameters, creates the initial configuration, constructs and runs the simulation.
   
   
   UAMMD input file example:
    
    2D ideal brownian walkers starting in a random configuration.

      
      The simulation starts with the particles randomly distributed. 
      An NVT MD integrator is used to move the particles for a few steps. Then the integrator is switched to a BD one and the simulation continues.
      
      As there is no interaction, gcnf.L is not needed, and the box is infinite.

   References:
   [1] https://github.com/RaulPPelaez/UAMMD/wiki
   [2] https://github.com/RaulPPelaez/UAMMD/wiki/PairForces 

   You can check the gdr with tools/gdr. Run it to see the input.
*/

#include "SimulationConfig.h"


SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
  Timer tim; tim.tic();

  
  
  gcnf.T = 1.0;

 
  gcnf.N = 100000;
  gcnf.dt = 0.01;

  gcnf.L = make_real3(800,800,0);
  
  gcnf.nsteps1 = 1000;
  gcnf.nsteps2 = 1000000;
  gcnf.print_steps = 10000;  


  gcnf.seed = 0xffaffbfDEADBULL;




  pos = Vector4(gcnf.N);
  /*A random initial configuration*/
  fori(0,gcnf.N){
    pos[i] = make_real4(grng.uniform3(-400, 400), 0);
    pos[i].z = 0;
  }

  setParameters();
  /*Upload positions to GPU once the initial conditions are set*/
  pos.upload();

  /*Termalization integrator*/
  integrator = make_shared<VerletNVT>();


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
  Matrixf K(3,3), M(3,3);
  K.fill_with(0); /*Shear matrix*/
  M.fill_with(0); /*Mobility matrix*/
  fori(0,3) M[i][i] = 1/(6*M_PI*vis*rh);
  integrator = make_shared<BrownianEulerMaruyama>(M, K);
  

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
  uint N = gcnf.N;
  real4 *posdata = pos.data;


  cout<<"#Lx="<<0<<";Ly="<<0<<";Lz="<<0<<";\n";
  fori(0,N){
    //uint type = (uint)(posdata[i].w+0.5);
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" 0.5f 0\n";// "<<.5f<<" "<<type<<"\n";
  }
  cout<<flush;
}

