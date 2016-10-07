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
Current benchmark:
GTX980 CUDA-7.5
N = 2^20
L = 128
dt = 0.001f
1e4 steps
PairForces with rcut = 2.5 and no energy measure
TwoStepVelverlet, no writing to disk, Tini = 0.03
Starting in a cubicLattice
---------------------HIGHSCORE-----------------------
Number of cells: 51 51 51; Total cells: 132651
Initializing...	DONE!!
Initialization time: 0.15172s
Computing step: 10000   
Mean step time: 127.33 FPS

Total time: 78.535s

real	1m19.039s
user	0m53.772s
sys	0m25.212s
---------------------------------------------------
TODO:
100- Read and construct simulation configuration from script
*/
#include<iomanip>
#include"globals/globals.h"
#include"Driver/SimulationConfig.h"


/*Declaration of extern variables in globals.h*/
GlobalConfig gcnf;
Vector4 pos, force;
Vector3 vel;
Xorshift128plus grng;

int main(int argc, char *argv[]){

  fori(1,argc)
    if(strcmp("--device", argv[i])==0)
      cudaSetDevice(atoi(argv[i+1]));

  /*To increase the size of the printf buffer inside kernels, default is 4096 lines I think*/
  // size_t size;
  // cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
  // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size*1000);

  /*The simulation handler*/
  SimulationConfig psystem(argc, argv);

  
  
  if(gcnf.print_steps>0) psystem.write(true);

  cudaDeviceSynchronize();
  
  return 0;
}


