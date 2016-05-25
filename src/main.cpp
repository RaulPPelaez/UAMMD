/*
Raul P. Pealez 2016. Launcher for Interactor.

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
    psystem->update_development(); //Prints timing information
    //if(i%500==0) psystem->write(); //Writing is done in parallel, is practically free if the interval is big enough
  }
  float total_time = tim.toc();
  cerr<<"\nMean step time: "<<setprecision(5)<<(float)nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nTotal time: "<<setprecision(5)<<total_time<<"s"<<endl;
  //psystem->write(true);
  cudaDeviceReset();
  return 0;
}


