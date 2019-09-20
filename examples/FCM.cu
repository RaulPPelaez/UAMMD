/*Raul P. Pelaez 2018. FCM/PSE example
  Non interacting particles inside a box starting in an FCC lattice fluctuate via the periodic FCM/RPY hydrodynamic kernel with the Force coupling method.

You can visualize the output with superpunto.
*/

#include"uammd.cuh"
#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_FCM.cuh"
#include"Integrator/BDHI/BDHI_PSE.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>

using namespace uammd;

using std::make_shared;
using std::endl;

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
    std::copy(initial.begin(), initial.end(), pos.begin());
  }


  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  using Method  = BDHI::PSE;
  //using Method  = BDHI::FCM;
  double rh =  1;
  std::ofstream out("pos.fcm");
  Method::Parameters par;
  par.temperature = 1.0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius =  rh;
  par.dt = 0.005;
  par.box = box;
  par.tolerance = 1e-2;

  auto bdhi = make_shared<BDHI::EulerMaruyama<Method>>(pd, pg, sys, par);


  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  Timer tim;
  tim.tic();
  int nsteps = 10000;
  int printSteps = 20;

  forj(0,nsteps){

    bdhi->forwardTime();
    if(j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");

      auto pos = pd->getPos(access::location::cpu, access::mode::read);

      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#"<<endl;
      real3 p;
      fori(0,N){
	real4 pc = pos[sortedIndex[i]];
	p = make_real3(pc);
	int type = pc.w;
	out<<p<<" "<<rh<<" "<<type<<"\n";
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


