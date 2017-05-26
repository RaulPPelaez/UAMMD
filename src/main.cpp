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
#include"Driver/SimulationScript.h"
#include<cuda.h>

/*Declaration of extern variables in globals.h*/
GlobalConfig gcnf;
uint current_step = 0;
SystemInfo sysInfo;
Vector4 pos, force;
Vector3 vel;
Xorshift128plus grng;


int main(int argc, char *argv[]){
  int dev = -1;
  {
    int arg_dev = checkFlag(argc, argv, "--device");
    if(arg_dev!=-1)
      dev = atoi(argv[arg_dev+1]);
  }
  {
    fori(0,60) cerr<<"━";
    cerr<<"┓"<<endl;
    string line1 = "\033[94m╻\033[0m \033[94m╻┏━┓┏┳┓┏┳┓╺┳┓\033[0m";
    string line2 = "\033[94m┃\033[0m \033[94m┃┣━┫┃┃\033[34m┃┃┃┃\033[0m \033[34m┃┃\033[0m";
    string line3 = "\033[34m┗━┛╹\033[0m \033[34m╹╹\033[0m \033[34m╹╹\033[0m \033[34m╹╺┻┛\033[0m";
    cerr<<line1<<endl;
    cerr<<line2<<" Version: "<<UAMMD_VERSION<<endl;
    cerr<<line3<<endl;
    cerr<<"Compiled at: "<<__DATE__<<" "<<__TIME__<<endl;
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    cerr<<"Computation started at "<<std::ctime(&time)<<endl;
  }
  /*Get device information*/
  
  cudaFree(0);/*This forces CUDAs lazy initialization to create a context, so a GPU is available to cudaGetDevice*/
  if(dev<0)
    cudaGetDevice(&dev);  
  cudaFree(0);
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  cerr<<"Using device: "<<deviceProp.name<<" with id: "<<dev<<endl;

  sysInfo.dev = dev;
  sysInfo.cuda_arch = 100*deviceProp.major + 10*deviceProp.minor;
  cerr<<"Compute capability of the device: "<<deviceProp.major<<"."<<deviceProp.minor<<endl;
  cerr<<"  ";
  fori(0,29) cerr<<"━ ";
  cerr<<endl;
  if(sysInfo.cuda_arch<200){
    cerr<<"\tERROR: Unsupported configuration, the GPU must have at least compute capability 2.0"<<endl;
    exit(1);
  }
  /*To increase the size of the printf buffer inside kernels, default is 4096 lines I think*/
   size_t size;
  cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size*10);

  /*Check for a simulation script*/
  int arg_script = checkFlag(argc, argv, "--script");
  
  /*The simulation handler*/
  /*The constructor does all the work*/

  if(arg_script!=-1)   /*Simulation is script driven*/
    SimulationScript psystem(argc, argv, argv[arg_script+1]);
  else /*Simulation is hardcoded in SimulationConfig.cpp*/
    SimulationConfig psystem(argc, argv);

  
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


