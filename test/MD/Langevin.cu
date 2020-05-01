/*Raul P. Pelaez 2018. Langevin test.

  Performs a NVT MD simulation of a LJ liquid.
  It uses the VerletNVT::GronbechJensen integrator.
  It allows for testing of the LJ potential and the integrator by using it to reproduce the eq of state.

*/

#include"uammd.cuh"
#include"Integrator/VerletNVT.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>


using namespace uammd;


real3 boxSize;
int numberSteps;
int printSteps;
int relaxSteps;
real dt;
int numberParticles;
real sigma;
real epsilon;
real cutOff;
real temperature;
bool shift;
std::string outfile, energyOutfile;
using namespace std;
void readParameters(std::string datamain, shared_ptr<System> sys);
int main(int argc, char *argv[]){
  auto sys = make_shared<System>();
  readParameters("data.main", sys);

  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(numberParticles, sys);

  Box box(boxSize);
  sys->log<System::MESSAGE>("[System] Box size: %e %e %e", boxSize.x, boxSize.y, boxSize.z);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto energy = pd->getEnergy(access::location::cpu, access::mode::readwrite);
    auto initial =  initLattice(box.boxSize, numberParticles, sc);
    fori(0,numberParticles){
      pos.raw()[i] = initial[i];
      pos.raw()[i].w = 0;
      energy.raw()[i] = 0;
    }
  }

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");


  using NVT = VerletNVT::GronbechJensen;
  NVT::Parameters par;
  par.temperature = temperature;
  par.dt = dt;
  par.viscosity = 1.0/(M_PI);
  auto verlet = make_shared<NVT>(pd, pg, sys, par);

  using PairForces = PairForces<Potential::LJ>;
  if(shift)
    sys->log<System::MESSAGE>("[System] LJ Parameters: sigma %e, epsilon %e, cutOff %e·sigma, shift: truncated and shifted", sigma, epsilon, cutOff);
  else
    sys->log<System::MESSAGE>("[System] LJ Parameters: sigma %e, epsilon %e, cutOff %e·sigma, shift: truncated", sigma, epsilon, cutOff);

  auto pot = make_shared<Potential::LJ>(sys);
  {

    Potential::LJ::InputPairParameters par;
    par.epsilon = epsilon;
    par.shift = shift;

    par.sigma = sigma;
    par.cutOff = cutOff*par.sigma;
    pot->setPotParameters(0, 0, par);
  }

  PairForces::Parameters params;
  params.box = box;
  auto pairforces = make_shared<PairForces>(pd, pg, sys, params, pot);

  verlet->addInteractor(pairforces);

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();



  std::ofstream out(outfile);
  std::ofstream eout(energyOutfile);
  eout.precision(2*sizeof(uammd::real));
  out.precision(2*sizeof(uammd::real));
  Timer tim;
  tim.tic();


  forj(0,relaxSteps) verlet->forwardTime();

  forj(0,numberSteps){

    verlet->forwardTime();

    if(printSteps > 0 && j%printSteps==1)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      {
	auto pos = pd->getPos(access::location::cpu, access::mode::read);
	const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
	out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
	real3 p;
	fori(0,numberParticles){
	  real4 pc = pos.raw()[sortedIndex[i]];
	  //pos.raw()[sortedIndex[i]] = make_real4(box.apply_pbc(make_real3(pc)), 0);
	  //p = box.apply_pbc(make_real3(pc));
	  p = make_real3(pc);
	  int type = pc.w;
	  out<<p<<" "<<0.5*(type==1?2:1)<<" "<<type<<endl;
	}
      }
      {
	pairforces->sumEnergy();
	auto energy = pd->getEnergy(access::location::cpu, access::mode::readwrite);
	auto vel = pd->getVel(access::location::cpu, access::mode::read);
	double U = 0.0;
	double K = 0.0;
	fori(0, numberParticles){
	  U += energy.raw()[i];
	  K+=0.5*dot(vel.raw()[i],vel.raw()[i]);
	  energy.raw()[i] = 0;
	}
	eout<<(0.5*U)/numberParticles<<" "<<(K)/numberParticles<<endl;

      }
    }

    if(j%500 == 0){
      pd->sortParticles();
    }
  }

  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps/totalTime);
  sys->finish();
  return 0;
}

void readParameters(std::string datamain, shared_ptr<System> sys){


  InputFile in(datamain, sys);

  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("relaxSteps", InputFile::Required)>>relaxSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("sigma", InputFile::Required)>>sigma;
  in.getOption("epsilon", InputFile::Required)>>epsilon;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("shiftLJ", InputFile::Required)>>shift;
  in.getOption("outfile", InputFile::Required)>>outfile;
  in.getOption("energyOutfile", InputFile::Required)>>energyOutfile;
  in.getOption("cutOff", InputFile::Required)>>cutOff;

}
