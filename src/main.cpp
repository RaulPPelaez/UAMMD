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

*/
#include<iomanip>
#include"Driver/Driver.h"

int main(){

  Timer tim;
  uint N = pow(2,20);
  tim.tic();
  Driver psystem(N, 128, 2.5f, 0.001f);

  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;  
  int nsteps = 10000;

  tim.tic();
  fori(0,nsteps){
    psystem.update(); //Prints timing information
    //if(i%500==0) psystem.write(); //Writing is done in parallel, is practically free if the interval is big enough
  }
  float total_time = tim.toc();
  cerr<<"\nMean step time: "<<setprecision(5)<<(float)nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nTotal time: "<<setprecision(5)<<total_time<<"s"<<endl;
  psystem.write(true);
  cudaDeviceReset();
  return 0;
}


