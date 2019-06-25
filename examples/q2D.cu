

#include"uammd.cuh"
#include"Integrator/Hydro/BDHI_quasi2D.cuh"

#include"utils/InputFile.h"
#include<fstream>
#include<cstring>

using namespace uammd;

using std::make_shared;
using std::endl;
using std::cerr;


int N;
real3 boxSize = make_real3(0);

int2 cells = {-1, -1};
std::string initFile, outFile("/dev/stdout");

real viscosity, temperature, dt, tolerance=1e-3, hydrodynamicRadius;

int numberSteps, printSteps, relaxSteps;
bool hydro;


void readParameters(shared_ptr<System> sys);
int main(int argc, char *argv[]){    

  auto sys = make_shared<System>(argc, argv);

  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);
  readParameters(sys);

  
  Box box(boxSize);
  
  std::ifstream in(initFile);
  if(!in) sys->log<System::CRITICAL>("Could not read from %s!", initFile.c_str());
  in>>N;
  
  auto pd = make_shared<ParticleData>(N, sys);

  {  
    auto pos = pd->getPos(access::location::cpu, access::mode::write); 
    fori(0,N){
      in>>pos[i].x>>pos[i].y>>pos[i].z;
      pos[i].w = 0;
    }
  }
  
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  std::ofstream out(outFile);

  std::shared_ptr<Integrator> bdhi;
  {
    using Scheme = BDHI::Quasi2D;
    Scheme::Parameters par;
    par.temperature = temperature;
    par.viscosity = viscosity;
    par.dt = dt;
    par.hydrodynamicRadius = hydrodynamicRadius;
    par.cells = cells;
    par.box = box;
    
    bdhi = make_shared<Scheme>(pd, pg, sys, par);    
  }
  sys->log<System::MESSAGE>("RUNNING!!!");

  //pd->sortParticles();
        
  Timer tim;
  tim.tic();
  int nsteps = numberSteps;  

  forj(0,relaxSteps){
    bdhi->forwardTime();
  }
  forj(0,nsteps){

    bdhi->forwardTime();

    if(printSteps and j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");   
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);      
      out<<"#"<<"\n";
      fori(0,N){
	real4 pc = pos[sortedIndex[i]];
	real3 p = make_real3(pc);
	int type = pc.w;
	real radius = hydrodynamicRadius;
	out<<p<<" "<<radius<<" "<<type<<"\n";
      }
      out<<std::flush;
    }
  
    //if(j%500 == 0)   pd->sortParticles();
    
  }
  
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
  sys->finish();
  
  return 0;
}



  
void readParameters(shared_ptr<System> sys){
  std::string fileName = sys->getargv()[1];
  InputFile in(fileName, sys);
  
  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y;
  in.getOption("cells", InputFile::Optional)>>cells.x>>cells.y;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("relaxSteps", InputFile::Required)>>relaxSteps;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>hydrodynamicRadius;
  in.getOption("initcoords", InputFile::Required)>>initFile;
  in.getOption("output", InputFile::Optional)>>outFile;  
  in.getOption("tolerance", InputFile::Optional)>>tolerance;
  in.getOption("hydro", InputFile::Required)>>hydro;    
}
