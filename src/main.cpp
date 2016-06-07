/*
Raul P. Pealez 2016. Launcher for Driver class.

Checkout Driver class to understand usage of this branch as a module.


NOTES:
The idea is to use either Integrator or Interactor in another project as a module.

Once initialized this classes will perform a single task very fast as black boxes:

Integrator uploads the positions according to the velocities, forces and current positions.
Interactor computes the pair forces using the current positions according to the selected potential

The idea is for Integrator to control the positions and velocities and for Interactor to control the forces. Communicating each variable when needed. So if you need the vel. in the force computing you can pass it when computing the force and modify the force function accordingly.
-------
To check physics try N= 2^14 and L = 32 and uncomment write calls

Or try N= 324 and L = 30 and uncomment write calls, in Driver put initial conditions in a cubicLattice2D with L=20
------
Current benchmark:
GTX980 CUDA-7.5
N = 2^20
L = 128
dt = 0.001f
PairForces with rcut = 2.5 and no energy measure
TwoStepVelverlet, no writing to disk
---------------------HIGHSCORE-----------------------
Number of cells: 51 51 51; Total cells: 132651
Initializing...	DONE!!
Initialization time: 0.16063s
Computing step: 10000   
Mean step time: 123.1 FPS

Total time: 81.233s

real	1m21.763s
user	0m55.792s
sys	0m25.972s
---------------------------------------------------


*/
#include<iomanip>
#include"Driver/Driver.h"

int main(){

  Timer tim;
  uint N = 324;//pow(2,20);
  tim.tic();
  Driver psystem(N, 30, 2.5f, 0.001f);

  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;  
  int nsteps = 1000000;

  tim.tic();
  fori(0,nsteps){
    psystem.update(); //Prints timing information
    if(i%500==0) psystem.write(); //Writing is done in parallel, is practically free if the interval is big enough
  }
  float total_time = tim.toc();
  cerr<<"\nMean step time: "<<setprecision(5)<<(float)nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nTotal time: "<<setprecision(5)<<total_time<<"s"<<endl;
  //psystem.write(true);
  cudaDeviceReset();
  return 0;
}


