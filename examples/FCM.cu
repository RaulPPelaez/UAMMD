/*Raul P. Pelaez 2018. FCM example
*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_FCM.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>

using namespace uammd;
using namespace std;

using uammd::real;

int main(int argc, char *argv[]){

  int N = pow(2,14);
  auto sys = make_shared<System>();
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(N, sys);

  Box box(32);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    
    auto initial =  initLattice(box.boxSize, N, fcc);
    fori(0,N){
      pos.raw()[i] = initial[i];
      pos.raw()[i].w = 0;
    }    
  }
  

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  double rh =  1;
  ofstream out("kk");
  BDHI::FCM::Parameters par;
  par.temperature = 1.0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius =  rh;
  par.dt = 0.001;
  par.box = box;
  
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::FCM>>(pd, pg, sys, par);
   
  
  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  Timer tim;
  tim.tic();
  int nsteps = 1000;
  int printSteps = 10;

  forj(0,nsteps){

    bdhi->forwardTime();
    if(j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");

      auto pos = pd->getPos(access::location::cpu, access::mode::read);

      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);      
      out<<"#"<<endl;
      real3 p;
      fori(0,N){
	real4 pc = pos.raw()[sortedIndex[i]];
	p = make_real3(pc);
	int type = pc.w;
	out<<p<<" "<<rh<<" "<<type<<endl;
      }
    }
    if(j%500 == 0){
      pd->sortParticles();
    }
  }
  
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
  sys->finish();

  return 0;
}


