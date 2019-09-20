/*Raul P. Pelaez 2018. Fluctuating Immerse Boundary for Brownian Dynamics with Hydrodynamic Interactions demo.

See BDHI/FIB.cuh for more info.
 */

#include"uammd.cuh"
#include"Integrator/BDHI/FIB.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>


using namespace uammd;
using std::make_shared;
using std::endl;


real3 boxSize;
real dt;
std::string outputFile;
int numberParticles;
int numberSteps, printSteps;
real temperature, viscosity;
real hydrodynamicRadius;

void readParameters(std::shared_ptr<System> sys, std::string file);

int main(int argc, char *argv[]){

  auto sys = make_shared<System>(argc, argv);
  readParameters(sys, "data.main.fib");

  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(numberParticles, sys);

  Box box(boxSize);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);

    auto initial =  initLattice(box.boxSize, numberParticles, sc);
    std::copy(initial.begin(), initial.end(), pos.begin());
  }


  auto pg = make_shared<ParticleGroup>(pd, sys, "All");




  BDHI::FIB::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius =  hydrodynamicRadius;
  par.dt = dt;
  par.box = box;
  par.scheme=BDHI::FIB::MIDPOINT; //Use simple midpoint scheme

  auto bdhi = make_shared<BDHI::FIB>(pd, sys, par);

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  std::ofstream out(outputFile);
  bdhi->forwardTime();
  bdhi->forwardTime();
  Timer tim;
  tim.tic();

  forj(0, numberSteps){

    bdhi->forwardTime();
    if(printSteps>0 and j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

      out<<"#"<<endl;
      real3 p;
      fori(0, numberParticles){
	real4 pc = pos[sortedIndex[i]];
	p = make_real3(pc);
	int type = pc.w;
	out<<p<<" "<<hydrodynamicRadius<<" "<<type<<"\n";
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


void readParameters(std::shared_ptr<System> sys, std::string file){

  {
    if(!std::ifstream(file).good()){
      real visco = 1/(6*M_PI);
      std::ofstream default_options(file);
      default_options<<"boxSize 25 25 25"<<std::endl;
      default_options<<"numberParticles 16384"<<std::endl;
      default_options<<"dt 0.005"<<std::endl;
      default_options<<"numberSteps 20000"<<std::endl;
      default_options<<"printSteps 100"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"temperature 1.0"<<std::endl;
      default_options<<"viscosity "<<visco<<std::endl;
      default_options<<"hydrodynamicRadius 1.0"<<std::endl;
    }
  }

  InputFile in(file, sys);

  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("outputFile", InputFile::Required)>>outputFile;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>hydrodynamicRadius;
}