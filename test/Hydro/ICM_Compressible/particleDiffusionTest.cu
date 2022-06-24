/* Raul P. Pelaez 2022. Compressible ICM particle diffusion test code.

   Places a group of particles randomly and lets them evolve in the fluid.
   Reads parameters from a data.main.

 */
#include "Integrator/Integrator.cuh"
#include "uammd.cuh"
#include "computeStructureFactor.cuh"
#include "Integrator/Hydro/ICM_Compressible.cuh"
#include "utils/container.h"
#include "utils/InputFile.h"
#include <cstdint>
#include <memory>
#include <stdexcept>

using namespace uammd;

using ICM = Hydro::ICM_Compressible;


struct Parameters{
  real dt = 0.1;
  real3 boxSize = make_real3(32,32,32)*100;
  int3 cellDim = {30,30,30};
  real bulkViscosity = 127.05;
  real speedOfSound = 14.67;
  real shearViscosity = 53.71;
  real temperature = 1;
  real initialDensity = 0.632;

  real relaxTime = 500;
  real simulationTime = -1;
  real printTime = 0;

  int numberParticles;
};

auto createICMIntegratorCompressible(std::shared_ptr<ParticleData> pd, Parameters ipar){
  ICM::Parameters par;
  par.dt = ipar.dt;
  par.boxSize = ipar.boxSize;
  //par.hydrodynamicRadius = 0.5;
  par.cellDim = ipar.cellDim;
  par.bulkViscosity = ipar.bulkViscosity;
  par.speedOfSound = ipar.speedOfSound;
  par.shearViscosity = ipar.shearViscosity;
  par.temperature = ipar.temperature;
  par.initialDensity = [=](real3 r){return ipar.initialDensity;};
  //par.initialVelocityX = [=](real3 r){return 0.001*ipar.boxSize.x/(ipar.cellDim.x);};
  return std::make_shared<ICM>(pd, par);
}

Parameters readParameters(std::string file){
  InputFile in(file);
  Parameters par;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("boxSize", InputFile::Required)>>par.boxSize.x>>par.boxSize.y>>par.boxSize.z;
  in.getOption("cellDim", InputFile::Required)>>par.cellDim.x>>par.cellDim.y>>par.cellDim.z;
  in.getOption("bulkViscosity", InputFile::Required)>>par.bulkViscosity;
  in.getOption("shearViscosity", InputFile::Required)>>par.shearViscosity;
  in.getOption("speedOfSound", InputFile::Required)>>par.speedOfSound;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("initialDensity", InputFile::Required)>>par.initialDensity;
  in.getOption("relaxTime", InputFile::Optional)>>par.relaxTime;
  in.getOption("simulationTime", InputFile::Required)>>par.simulationTime;
  in.getOption("printTime", InputFile::Required)>>par.printTime;
  in.getOption("numberParticles", InputFile::Required)>>par.numberParticles;
  return par;
}

void writePositions(std::shared_ptr<ParticleData> pd){
  auto pos = pd->getPos(access::cpu, access::read);
  std::cout<<"#"<<std::endl;
  for(auto p: pos) std::cout<<make_real3(p)<<"\n";
}

class Random{
  uint s1,s2;
  real L;
public:

  Random(uint s1, uint s2, real L):s1(s1), s2(s2),L(L){}

  __device__ real4 operator()(int i){
    Saru rng(s1, s2, i);
    return (make_real4(rng.f(), rng.f(), rng.f(), 0)-0.5)*L;
  }

};
auto initializeParticles(Parameters par, std::shared_ptr<System> sys){
  int numberParticles = par.numberParticles;
  auto pd = std::make_shared<ParticleData>(numberParticles, sys);
  auto pos = pd->getPos(access::gpu, access::write);
  auto cit = thrust::make_counting_iterator(0);
  thrust::transform(thrust::cuda::par,
		    cit, cit + numberParticles,
		    pos.begin(),
		    Random(sys->rng().next32(), sys->rng().next32(), par.boxSize.x));
  return pd;
}

int main(int argc, char *argv[]){
  auto sys = std::make_shared<System>(argc, argv);
  auto par = readParameters(argv[1]);
  auto pd = initializeParticles(par, sys);
  auto icm = createICMIntegratorCompressible(pd, par);
  {
    int relaxSteps =par.relaxTime/par.dt+1;
    fori(0, relaxSteps) icm->forwardTime();
  }
  int ntimes = par.simulationTime/par.dt;
  int sampleSteps = par.printTime/par.dt;
  fori(0, ntimes){
    icm->forwardTime();
    if(i%sampleSteps == 0){
      writePositions(pd);
      // writeFluidVelocity(icm);
      // writeFluidDensity(icm);
    }
  }
  System::log<System::MESSAGE>("Processing and writing results");
  return 0;
}
