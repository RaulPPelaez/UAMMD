#include"uammd.cuh"
#include"utils/quaternion.cuh"
#include"Integrator/BDHI/BDHI_FCM.cuh"
//#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"utils/InitialOrientations.cuh"
#include"utils/InitialConditions.cuh"

using namespace uammd;
using std::make_shared;

struct inputParams {
  int nParticles;
  int nsteps;
  int nsave;
  real lbox;
  real dt;
  real radius;
  real viscosity;
  real temperature;
  real tolerance = 0.001;
  bool rot; // 1 if we want to include the rotational movement, 0 if not
};

#include"utils/InputFile.h"
inputParams readData(std::shared_ptr<System> sys, std::string file){

  inputParams par;
  InputFile in(file, sys);
  in.getOption("nParticles", InputFile::Required)>>par.nParticles;
  in.getOption("nsteps", InputFile::Required)>>par.nsteps;
  in.getOption("nsave", InputFile::Required)>>par.nsave;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("radius", InputFile::Required)>>par.radius;
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("lbox", InputFile::Required)>>par.lbox;
  in.getOption("rot", InputFile::Required)>>par.rot;
  par.viscosity/=(8*M_PI);
  return par;
}

void saveData(std::shared_ptr<uammd::ParticleData> pd, int N, real r, std::ofstream &file){
  auto pos = pd -> getPos(access::location::cpu, access::mode::read);
  auto dir = pd -> getDirIfAllocated(access::location::cpu, access::mode::read);
  const int * index2id = pd->getIdOrderedIndices(access::location::cpu);
  auto save = [&](int id){    
    real3 pi = make_real3(pos[id]);
    file<<pi<<" "<<r<<" 0\n";
    if (dir.raw()){
      Quat diri = dir[id];
      real3 vx = diri.getVx();
      real3 vy = diri.getVy();
      real3 vz = diri.getVz();
      file<<pi+r*vx<<" "<<r/5.0<<" 1\n";
      file<<pi+r*vy<<" "<<r/5.0<<" 1\n";
      file<<pi+r*vz<<" "<<r/5.0<<" 1\n"; 
    }
  };
  file<<"#\n";
  std::for_each(index2id,index2id+pos.size(),save);
}



int main(int argc, char* argv[]){

  auto sys = make_shared<System> (argc, argv);
  auto par = readData(sys, "Params.data");
  Box box(par.lbox);
  uint seed = std::time(NULL);
  sys -> rng().setSeed(seed);

  auto pd = make_shared<ParticleData> (par.nParticles, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  auto initPos = initLattice(box.boxSize, par.nParticles, sc);
  

  {
    auto pos = pd -> getPos(access::location::cpu, access::mode::write);
    std::copy(initPos.begin(), initPos.end(), pos.begin());

    if (par.rot){
    auto initDir = extensions::initOrientations(par.nParticles,seed,"aligned");
    auto dir = pd -> getDir(access::location::cpu, access::mode::write);
    std::copy(initDir.begin(), initDir.end(), dir.begin());
    }
  }

  using integrator = BDHI::FCM;
  integrator::Parameters bpar;
  bpar.dt = par.dt;
  bpar.viscosity = par.viscosity;
  bpar.seed = seed;
  bpar.temperature = par.temperature;
  bpar.hydrodynamicRadius = par.radius;
  bpar.box = box;
  bpar.tolerance = par.tolerance;
  auto fcm = std::make_shared<integrator>(pd, pg, sys, bpar);
  std::ofstream file("Pos.out");
  Timer tim;
  tim.tic();
  fori(0,par.nsteps){
    if (i%par.nsave == 0){
      saveData(pd,par.nParticles, par.radius, file);
      std::cout<<i<<"\n";
    }
    fcm->forwardTime();
  }
  std::cout<<tim.toc()<<"\n";
  cudaDeviceSynchronize();

  return 0;
}
