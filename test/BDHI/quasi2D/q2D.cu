/* Raul P. Pelaez 2020. q2D test

 */
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
std::string testSelection;
real viscosity, temperature, dt, tolerance=1e-3, hydrodynamicRadius;
real F = 0;
int numberSteps, printSteps, relaxSteps;
std::string scheme;
int in_numberParticles;

class miniInteractor: public Interactor{
public:
  using Interactor::Interactor;
  
  real2 F = real2();
  
  void sumForce(cudaStream_t st) override{
    auto force = pd->getForce(access::location::cpu, access::mode::write);
    force[0] = make_real4(F.x, F.y,0,0);
    if(pg->getNumberParticles() > 1)
      force[1] = make_real4(-F.x,-F.y, 0,0);
  }
  
  real sumEnergy() override{return 0;}
};

template<class Scheme>
void selfMobilityTest(shared_ptr<System> sys, int dir){
  int N = 1;
  Box box(boxSize);
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  typename Scheme::Parameters par;
  par.temperature = 0;
  par.viscosity = viscosity;
  par.dt = dt;
  par.hydrodynamicRadius = hydrodynamicRadius;
  par.cells = cells;
  par.box = box;
  auto bdhi = make_shared<Scheme>(pd, pg, sys, par); 
  auto inter= make_shared<miniInteractor>(pd, sys, "puller");
  if(dir==0)
    inter->F.x = F;
  else
    inter->F.y = F;
  bdhi->addInteractor(inter);
  real3 M = real3();
  int ntest = 1000;
  fori(0, ntest){
    real3 oldPos;
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::write);
      pos[0] = make_real4(make_real3(sys->rng().uniform3(-boxSize.x*0.5, boxSize.x*0.5)));
      pos[0].z = 0;
      // pos[0].x = 0.4;
      // pos[0].y = 0.4;
      oldPos = make_real3(pos[0]);
    }
    bdhi->forwardTime();
    real3 newPos;
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      newPos = make_real3(pos[0]);
    }
    real3 dx = box.apply_pbc(newPos - oldPos);
    M += dx/(dt*F);
  }
  std::ofstream out(outFile);
  out.precision(sizeof(real)*2); 
  out<<(M/double(ntest))<<std::endl;
}

template<class Scheme>
void selfDiffusionTest(shared_ptr<System> sys){
  int N = in_numberParticles;
  Box box(boxSize);
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  typename Scheme::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.dt = dt;
  par.hydrodynamicRadius = hydrodynamicRadius;
  par.cells = cells;
  par.box = box;
  auto bdhi = make_shared<Scheme>(pd, pg, sys, par);  
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    fori(0, N){
      pos[i] = make_real4(make_real3(sys->rng().uniform3(-boxSize.x*0.5, boxSize.x*0.5)));
      pos[i].z = 0;
    }
  }
  std::ofstream out(outFile);
  out.precision(sizeof(real)*2);
  fori(0, numberSteps){
    bdhi->forwardTime();
    if(printSteps and i%printSteps == 0){
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      out<<"#"<<endl;
      fori(0, N){
	out<<pos[i]<<"\n";
      }
    }
  }
}

void runSelfMobilityTest(shared_ptr<System> sys){
  fori(0,1){
    if(scheme.compare("true2D") == 0){
      selfMobilityTest<BDHI::True2D>(sys, i);
    }
    else if(scheme.compare("quasi2D") == 0){
      selfMobilityTest<BDHI::Quasi2D>(sys, i);
    }
  }
}

void runSelfDiffusionTest(shared_ptr<System> sys){
  if(scheme.compare("true2D") == 0){
    selfDiffusionTest<BDHI::True2D>(sys);
  }
  else if(scheme.compare("quasi2D") == 0){
    selfDiffusionTest<BDHI::Quasi2D>(sys);
  }
}

void readParameters(shared_ptr<System> sys);
int main(int argc, char *argv[]){    
  auto sys = make_shared<System>(argc, argv);
  readParameters(sys);
  if(testSelection.compare("selfMobility") == 0){
    runSelfMobilityTest(sys);
  }
  else if(testSelection.compare("selfDiffusion") == 0){    
    runSelfDiffusionTest(sys);
  }    
  sys->finish(); 
  return 0;
}

void readParameters(shared_ptr<System> sys){
  std::string fileName = sys->getargv()[1];
  InputFile in(fileName, sys);
  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y;
  in.getOption("cells", InputFile::Optional)>>cells.x>>cells.y;
  in.getOption("numberSteps", InputFile::Optional)>>numberSteps;
  in.getOption("printSteps", InputFile::Optional)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("relaxSteps", InputFile::Optional)>>relaxSteps;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>hydrodynamicRadius;
  in.getOption("output", InputFile::Optional)>>outFile;  
  in.getOption("tolerance", InputFile::Optional)>>tolerance;
  in.getOption("scheme", InputFile::Required)>>scheme;
  in.getOption("test", InputFile::Required)>>testSelection;
  in.getOption("initcoords", InputFile::Optional)>>initFile;
  in.getOption("F", InputFile::Optional)>>F;
  in.getOption("numberParticles", InputFile::Optional)>>in_numberParticles;
}
