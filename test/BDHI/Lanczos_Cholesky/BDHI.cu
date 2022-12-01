#include"uammd.cuh"
#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_Cholesky.cuh"
#include"Integrator/BDHI/BDHI_Lanczos.cuh"
#include"utils/InputFile.h"
#include<iomanip>
#include<fstream>


using namespace uammd;

using std::make_shared;
using std::endl;
using std::cerr;

int N;
ullint seed = 0xf31337Bada55D00dULL^time(NULL);
real3 boxSize;
real radius_min = 1;
real radius_max = 2;
std::string outfile;
real temperature;
real viscosity;
real dt;
real tolerance = 1e-6;

int nsteps;
int printSteps;

std::string mode = "Lanczos";


class miniInteractor: public Interactor{
public:
  using Interactor::Interactor;
  real3 F = make_real3(0);
  void sum(Computables comp, cudaStream_t st) override{
    auto force = pd->getForce(access::location::cpu, access::mode::write);
    const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
    force.raw()[sortedIndex[0]] = make_real4(F,0);
  }
};


void readParameters(shared_ptr<System> sys);
int main(int argc, char *argv[]){
  auto sys = make_shared<System>(argc, argv);
  readParameters(sys);
  sys->rng().setSeed(seed);
  auto pd = make_shared<ParticleData>(N, sys);
  Box box(boxSize);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto radius = pd->getRadius(access::location::cpu, access::mode::write);
    fori(0,N){
      pos.raw()[i] = make_real4(sys->rng().uniform3(-boxSize.x*0.5, boxSize.x*0.5),0);
      radius.raw()[i] = radius_min;
    }
    pos.raw()[0] = make_real4(0);
    radius.raw()[0] = radius_max;
  }

  std::ofstream out(outfile);
  BDHI::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.dt = dt;
  par.tolerance = tolerance;
 
  shared_ptr<Integrator> bdhi;
  if(mode == "Lanczos")
    bdhi = make_shared<BDHI::EulerMaruyama<BDHI::Lanczos>>(pd, par);
  else if(mode =="Cholesky")
    bdhi = make_shared<BDHI::EulerMaruyama<BDHI::Cholesky>>(pd, par);
  else
    sys->log<System::CRITICAL>("[System] Unrecognized mode! choose Lanczos or Cholesky");

  auto inter = make_shared<miniInteractor>(pd, "puller");
  inter->F.x = 1;
  bdhi->addInteractor(inter);

  //pd->sortParticles();

  Timer tim;
  tim.tic();
  forj(0,nsteps){
    if(printSteps>0 and j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      auto radius = pd->getRadius(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

      out<<"#"<<endl;
      real3 p;
      fori(0,N){
	real4 pc = pos.raw()[sortedIndex[i]];
	p = make_real3(pc);
	int type = int(radius.raw()[sortedIndex[i]]);
	out<<std::setprecision(17)<<p<<" "<<radius.raw()[sortedIndex[i]]<<" "<<type<<"\n";
      }
    }
    bdhi->forwardTime();
    //if(j%500 == 0) pd->sortParticles();
  }

  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
  sys->finish();
  return 0;
}

void readParameters(shared_ptr<System> sys){
  InputFile in("data.main", sys);
  in.getOption("N", InputFile::Required)>>N;
  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("radius_min", InputFile::Required)>>radius_min;
  in.getOption("radius_max", InputFile::Required)>>radius_max;
  in.getOption("outfile", InputFile::Required)>>outfile;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("tolerance", InputFile::Optional)>>tolerance;
  in.getOption("nsteps", InputFile::Required)>>nsteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("mode", InputFile::Required)>>mode;
  in.getOption("seed", InputFile::Optional)>>seed;
}
