/*
Raul P. Pealez 2016. Launcher for Interactor.

Serves as a benchmark.
-------------------------------------------
Current benchmark (best time in a GTX980 non Ti):
 N = 2^20, nsteps = 1e4, initial in a cube of L=128, dt = 0.003
         , random initial vel with amp. 0.1

 Computing step: 9500   
 Integrate 1 0.00047091
 Cell find calc hash 0.00015472
 Cell find sort hash 0.0010706
 Cell find reorder pos 0.00033744
 Compute Forces 0.0068873
 Integrate 2 0.00026054

 Mean step time: 116.5 FPS

 Total time: 85.835s
--------------------------------------------
To check physics try N= 2^14 and L = 32 and uncomment write calls


*/
#include"Interactor.h"
#include<iomanip>
Interactor *psystem;
int main(){
  Timer tim;
  uint N = pow(2,20);
  tim.tic();
  psystem = new Interactor(N,128, 2.5, 0.003f);
  //  psystem->read("1Minit.pos");

  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;  
  //psystem->write(true);
  int nsteps = 10000;
  tim.tic();
  //  fori(0,nsteps) psystem->update();
  fori(0,nsteps){
    psystem->update();
    // if(i%100==0) psystem->write();
  }
  float total_time = tim.toc();
  cerr<<"\nMean step time: "<<setprecision(5)<<(float)nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nTotal time: "<<setprecision(5)<<total_time<<"s"<<endl;
  //  psystem->write(true);
  cudaDeviceReset();
  return 0;
}


