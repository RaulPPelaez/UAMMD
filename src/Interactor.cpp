/*
Raul P. Pelaez 2016. Interactor class GPU wrapper, to use from CPU, handles GPU calls, integration and interaction of particles. Also keeps track of all the variables and writes/reads simulation to disk.

The idea is for this class to be use to compute only the forces, and integrate elsewhere.
Like give me a pointer to positions and get a pointer to forces

TODO:
100-Separate integrator and Interactor, should be done separatedly.
100-Allow force function to be change from outside.
90- Colors, this almost done, color can simply be encoded in pos.w
90- Springs. Use Floren's algorithm from fluam.
90- Non cubic boxes, almost done, just be carefull in the constructor and use vector types.

NOTES:
Use update isntead of update_development to increase performance (update doesnt record time and doesnt sync device)

*/
#include"Interactor.h"
#include<thread>
#include<cmath>
#include<iostream>
#include<fstream>
#include<iomanip>
using namespace std;


Interactor::Interactor(int N, float L, float rcut, float dt, forceType fs):
  N(N), L(L), rcut(rcut), dt(dt){
  /**Put parameters in Param struct**/
  params.rcut = rcut;

  int xcells = int(L/rcut+0.5);
  int ycells = xcells, zcells= xcells;

  params.L = L;

  params.cellSize = L/(float)xcells;

  params.xcells = xcells;
  params.ycells = ycells;
  params.zcells = zcells;

  
  ncells = xcells*ycells*zcells;
  params.ncells = ncells;
  cerr<<"Number of cells: "<<xcells<<" "<<ycells<<" "<<zcells<<"; Total cells: "<<ncells<<endl;
  
  forceSelector = fs;

  init();

}

Interactor::~Interactor(){
  writeThread->join();
}
//Force between two particles, depending on square distance between them
// this function is only called on construction, so it doesnt need to be optimized at all
float forceLJ(float r2){
  float invr2 = 1.0f/(r2);
  float invr6 = invr2*invr2*invr2;		 
  float invr8 = invr6*invr2;		 
  
  float fmod = -48.0f*invr8*invr6+24.0f*invr8;

  return fmod;
}
float nullForce(float r2){return 0.0f;}

//Initialize variables and upload them to GPU, init CUDA
void Interactor::init(){
  cerr<<"Initializing...";

  switch(forceSelector){
  case LJ:
    /*Compute potential, using LJ function*/
    pot = Potential(forceLJ, 4096, params.rcut);
    break;
  case NONE:
    pot = Potential(nullForce, 4096, params.rcut);
    break;
  default:
    cerr<<"NON RECOGNIZED POTENTIAL SELECTED!!"<<endl;
    exit(1);
  }

  /*pos and force will be float4, vel float3*/
  pos   = Vector<float>(4*N, true);  sortPos   = Vector<float>(4*N); 
  vel   = Vector<float>(3*N); 
  force = Vector<float>(4*N); 
  

  sortPos.fill_with(0.0f);
  /*Start with a cubic lattice, later one can read a configuration from file*/
  cubicLattice(pos.data, L, N);   
  

  vel.fill_with(0.0f);
  force.fill_with(0.0f);

  /*Random initial velocities*/
  fori(0,3*N) vel[i] = 0.1f*(2.0f*(rand()/(float)RAND_MAX)-1.0f);


  pos.upload();
  vel.upload();
  force.upload();

  sortPos.upload(); 

  cellIndex = Vector<uint>(N+1); cellIndex.upload();
  particleIndex= Vector<uint>(N+1); particleIndex.upload();
  cellStart        = Vector<uint>(ncells); cellStart.upload();
  cellEnd          = Vector<uint>(ncells); cellEnd.upload();

  int memneed = (2*4+3*3)*N*sizeof(float)+(2*ncells +2*N)*sizeof(uint);
  cerr<<"\nMemory needed: "<<setprecision(2)<<memneed/float(1024*1024)<<" mb"<<endl;

  initGPU(params, pot.getData(), pot.getSize(), cellStart, cellEnd, particleIndex, ncells, sortPos, N);

  cerr<<"\tDONE!!"<<endl;
}
/*Perform an integration step*/
void Interactor::update(){
  static int steps = 0;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";
 
  /*** UPDATE POSITIONS***/
  integrate(pos, vel, force, dt, N, 1);
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  /*Compute cell id of each particle*/
  calcCellIndex(pos, cellIndex, particleIndex, N);

  /*Sort the particle indices by hash (cell index)*/
  sortCellIndex(cellIndex, particleIndex, N);
  /*Reorder positions by cell index and construct cellStart and cellEnd*/
  reorderAndFind(sortPos,
		 cellIndex, particleIndex,
		 cellStart, cellEnd, params.ncells,
		 pos, N); 
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computeForce(sortPos,
	       force, 
	       cellStart, cellEnd, 
	       particleIndex,
	       N);
  /***UPDATE VELOCITIES***/
  integrate(pos, vel, force, dt, N, 2);//, steps%10 ==0 && steps<10000 && steps>1000);
  steps++;
}
//Performs an integration step like update, but syncs the GPU after each substep to measure time
//Use this for debug and development
void Interactor::update_development(){
  static int steps = 0;
  static Timer tim;
  static float tima[] = {0,0,0,0,0,0};

  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";
 
  /*** UPDATE POSITIONS***/
  tim.tic();
  integrate(pos, vel, force, dt, N, 1);
  cudaDeviceSynchronize();
  tima[0] += tim.toc();

  /*** CONSTRUCT NEIGHBOUR LIST ***/
  /*Compute cell id of each particle*/
  tim.tic();
  calcCellIndex(pos, cellIndex, particleIndex, N);
  cudaDeviceSynchronize();
  tima[4] += tim.toc();
  /*Sort the particle indices by hash (cell index)*/
  tim.tic();
  sortCellIndex(cellIndex, particleIndex, N);
  cudaDeviceSynchronize();
  tima[5] += tim.toc();
  /*Reorder positions by cell index and construct cellStart and cellEnd*/
  tim.tic();
  reorderAndFind(sortPos,
		 cellIndex, particleIndex,
		 cellStart, cellEnd, params.ncells,
		 pos, N); 
  cudaDeviceSynchronize();
  tima[1] += tim.toc();

  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  tim.tic();
  computeForce(sortPos,
	       force, 
	       cellStart, cellEnd, 
	       particleIndex,
	       N);
  cudaDeviceSynchronize();
  tima[2] += tim.toc();

  /***UPDATE VELOCITIES***/
  tim.tic();
  integrate(pos, vel, force, dt, N, 2);//, steps%10 ==0 && steps<10000 && steps>1000);
  cudaDeviceSynchronize();
  tima[3] += tim.toc();

  /*** PRINT INFORMATION***/
    int nprint= 500;

    if(steps%nprint==0){
      cerr<<endl;
      cerr<<"Integrate 1 "<<tima[0]/(float)nprint<<endl;
      cerr<<"Cell find calc hash "<<tima[4]/(float)nprint<<endl;
      cerr<<"Cell find sort hash "<<tima[5]/(float)nprint<<endl;
      cerr<<"Cell find reorder pos "<<tima[1]/(float)nprint<<endl;
      cerr<<"Compute Forces "<<tima[2]/(float)nprint<<endl;
      cerr<<"Integrate 2 "<<tima[3]/(float)nprint<<endl<<endl;
      forj(0,6) tima[j] = 0;
    }
  steps++;
}

//Write a step to disk using a separate thread
void Interactor::write(bool block){
  /*Wait for the last write operation to finish*/
  if(writeThread) writeThread->join();
  /*Bring pos from GPU*/
  pos.download();
  /*Wait for copy to finish*/
  cudaDeviceSynchronize();
  /*Query the write operation to another thread*/
  writeThread = new std::thread(write_concurrent, pos.data, L, N);
  if(block) writeThread->join();
}

//Read an initial configuratio nfrom fileName, TODO
void Interactor::read(const char *fileName){
  ifstream in(fileName);
  float r,c,l;
  in>>l;
  fori(0,N){
    in>>pos[4*i]>>pos[4*i+1]>>pos[4*i+2]>>r>>c;
  }
  in.close();
  pos.upload();
}
//This function writes a step to disk
void write_concurrent(float *pos, float L, uint N){
  cout<<"#L="<<L*0.5<<";\n";
  fori(0,N){
    cout<<pos[4*i]<<" "<<pos[4*i+1]<<" "<<pos[4*i+2]<<" 0.56 1\n";
  }
}
