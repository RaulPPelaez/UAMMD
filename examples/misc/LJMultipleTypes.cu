/*Raul P. Pelaez 2019. A short range forces example.

A LJ liquid simulation in a periodic box.

This example showcases the performance when using a "large" number of particle types.
It does this by setting particles types randomly from 0 to an input number and then configuring the potential
to handle each particle pair. In this case all particle pairs have the same parameters, but the potential is oblivious to this.

You can visualize the reuslts with superpunto

 */

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/VerletNVT.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>


using namespace uammd;
using namespace std;


int main(int argc, char *argv[]){
  int N = 16384;
  auto sys = make_shared<System>(argc, argv);
  ullint seed = 0xf31337Bada55D00dULL;
  sys->rng().setSeed(seed);
  auto pd = make_shared<ParticleData>(N, sys);
  Box box(32);
  int ntypes = 100;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto initial =  initLattice(box.boxSize, N, fcc);
    std::transform(initial.begin(), initial.end(), pos.begin(),
		   [&](real4 p){p.w = sys->rng().next32()%ntypes; return p;});
  }
  using NVT = VerletNVT::GronbechJensen;
  NVT::Parameters par;
  par.temperature = 0.1;
  par.dt = 0.01;
  par.friction = 1.0/(6*M_PI);
  auto verlet = make_shared<NVT>(pd, par);
  {
    using PairForces = PairForces<Potential::LJ>;
    auto pot = make_shared<Potential::LJ>();
    {
      Potential::LJ::InputPairParameters par;
      par.epsilon = 1.0;
      par.shift = false;
      par.sigma = 1;
      par.cutOff = 2.5*par.sigma;
      //All interaction parameters are the same, though the potential does not know this.
      fori(0,ntypes){
	for(int j = ntypes-1; j>=0; j--){
	  pot->setPotParameters(i, j, par);
	}
      }
    }

    PairForces::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = make_shared<PairForces>(pd, params, pot);
    verlet->addInteractor(pairforces);
  }


  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  Timer tim;
  tim.tic();
  int nsteps = 10000;

  forj(0, nsteps){
    verlet->forwardTime();
    if(j%500 == 0) pd->sortParticles();
  }

  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
  sys->finish();

  return 0;
}
