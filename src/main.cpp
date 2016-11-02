/*
Raul P. Pealez 2016. Launcher for Driver class.

NOTES:
The idea is to construct a simulation in SimulationConfig.cpp using the existing modules.

Once initialized these modules will perform a single task very fast as black boxes:

Integrator uploads the positions according to the forces and current positions (and any other internal state necessary, as the velocity .i.e).

Interactor computes the forces using the current positions according to the selected force module (Nbody or short range pair forces i.e)

-------
Current benchmark:
GTX980 CUDA-7.5
N = 2^20
L = 128
dt = 0.001
T = 0.03
1e4 steps
PairForces with rcut = 2.5 and no energy measure
VerletNVT, no writing to disk
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
#include"globals/defines.h"
#include<chrono>
#include<ctime>
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
  {
    fori(0,60) cerr<<"━";
    cerr<<"┓"<<endl;
    string line1 = "\033[94m╻\033[0m \033[94m╻┏━┓┏┳┓┏┳┓╺┳┓\033[0m";
    string line2 = "\033[94m┃\033[0m \033[94m┃┣━┫┃┃\033[34m┃┃┃┃\033[0m \033[34m┃┃\033[0m";
    string line3 = "\033[34m┗━┛╹\033[0m \033[34m╹╹\033[0m \033[34m╹╹\033[0m \033[34m╹╺┻┛\033[0m";
    cerr<<line1<<endl;
    cerr<<line2<<" Version: "<<UAMMD_VERSION<<endl;
    cerr<<line3<<endl;
  
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    cerr<<"Computation started at "<<std::ctime(&time)<<endl;
  }
  /*To increase the size of the printf buffer inside kernels, default is 4096 lines I think*/
  // size_t size;
  // cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
  // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size*1000);

  /*The simulation handler*/
  SimulationConfig psystem(argc, argv);
  /*The constructor does all the work*/
  /*Wait for any unfinished GPU job and get out*/
  cudaDeviceSynchronize();
  {
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    cerr<<"Computation finished at "<<std::ctime(&time)<<endl;
    fori(0,60) cerr<<"━";
    cerr<<"┛"<<endl;

  }
  return 0;
}


