/*Raul P. Pelaez 2018. MonteCarlo NVT test.

  Performs a NVT MC simulation of a LJ liquid.
  It uses the MC_NVT::Anderson integrator.
  It allows for testing of the LJ potential and the integrator by using it to reproduce the eq of state.
  
*/

#include"uammd.cuh"
#include"Integrator/MonteCarlo/NVT/Anderson.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>


using namespace uammd;
using namespace std;


real3 boxSize;
int numberSteps;
int printSteps;
int relaxSteps;

int numberParticles;
real sigma;
real epsilon;
real cutOff;
real temperature;
bool shift;
std::string outfile, energyOutfile;


int attempsPerCell;
real initialJumpSize;
int tuneSteps;
int thermalizationSteps;
real desiredAcceptanceRatio;
real acceptanceRatioRate;




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

  using NVT = MC_NVT::Anderson<Potential::LJ>;
  NVT::Parameters par;
  par.box = box;
  par.kT = temperature;
  par.attempsPerCell          = attempsPerCell;
  par.initialJumpSize		= initialJumpSize;
  
  par.thermalizationSteps	= thermalizationSteps;
  par.desiredAcceptanceRatio	= desiredAcceptanceRatio;
  par.acceptanceRatioRate	= acceptanceRatioRate;
  par.tuneSteps = tuneSteps;


  auto mc = make_shared<NVT>(pd, pg, sys, pot, par);

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();



  std::ofstream out(outfile);
  std::ofstream eout(energyOutfile);
  Timer tim;
  tim.tic();


  forj(0,relaxSteps) mc->forwardTime();

  forj(0,numberSteps){

    mc->forwardTime();

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
	double U = mc->computeInternalEnergy(true);
	double K = 1.5*temperature;
	eout<<U<<" "<<K<<endl;
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

  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("sigma", InputFile::Required)>>sigma;
  in.getOption("epsilon", InputFile::Required)>>epsilon;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("shiftLJ", InputFile::Required)>>shift;
  in.getOption("outfile", InputFile::Required)>>outfile;
  in.getOption("energyOutfile", InputFile::Required)>>energyOutfile;
  in.getOption("cutOff", InputFile::Required)>>cutOff;  
  in.getOption("attempsPerCell", InputFile::Required)>>attempsPerCell;
  in.getOption("initialJumpSize", InputFile::Required)>>initialJumpSize;
  in.getOption("tuneSteps", InputFile::Required)>>tuneSteps;
  in.getOption("thermalizationSteps", InputFile::Required)>>thermalizationSteps;
  in.getOption("desiredAcceptanceRatio", InputFile::Required)>>desiredAcceptanceRatio;
  in.getOption("acceptanceRatioRate", InputFile::Required)>>acceptanceRatioRate;

  
}
