//Raul P.Pelaez 2021, DPStokes test
#include "uammd.cuh"
#include"utils/InputFile.h"
#include"Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"

using namespace uammd;

using Scheme = DPStokesSlab_ns::DPStokes;

struct Parameters {
  real Lxy;
  real H;
  real viscosity;
  real dt;
  int numberParticles;
  int nxy;
  int nz = -1;
  real w;
  real w_d = 0;
  real beta;
  real beta_d = 0;
  real alpha;
  real alpha_d = -1;
  //Can be either none, bottom or slit
  std::string wallMode;
};


auto string2WallMode(std::string mode){
  if(mode == "none") return Scheme::WallMode::none;
  else if(mode == "bottom") return Scheme::WallMode::bottom;
  else if(mode == "slit") return Scheme::WallMode::slit;
  else
    throw std::runtime_error("Invalid wall mode");
}

auto createDPStokes(std::shared_ptr<ParticleData> pd, Parameters ipar){
  Scheme::Parameters par;
  par.dt = ipar.dt;
  par.nx = ipar.nxy;
  par.ny = ipar.nxy;
  par.nz = ipar.nz;
  par.w = ipar.w;
  par.w_d = ipar.w_d;
  par.beta = ipar.beta;
  par.beta_d = ipar.beta_d;
  par.alpha = ipar.alpha;
  par.alpha_d = ipar.alpha_d;
  par.mode = string2WallMode(ipar.wallMode);
  par.viscosity = ipar.viscosity;
  par.Lx = par.Ly = ipar.Lxy;
  par.H = ipar.H;
  auto dpstokes = std::make_shared<Scheme>(par);
  return dpstokes;
}


auto initializeParticles(std::shared_ptr<System> sys, Parameters par){
  auto pd = std::make_shared<ParticleData>(par.numberParticles, sys);
  auto pos = pd->getPos(access::cpu, access::write);
  pos[0] = make_real4(0,0,1,0);
  pos[1] = make_real4(0,0,-1,0);
  auto force = pd->getForce(access::cpu, access::write);
  std::fill(force.begin(), force.end(), real4());
  return pd;
}

void setParticlesInMiddlePlaneAtDistance(std::shared_ptr<ParticleData> pd, real distance, real lxy){
  auto randomPosition = make_real4(make_real2(pd->getSystem()->rng().uniform2(-0.5, 0.5)*lxy), 0,0);
  auto pos = pd->getPos(access::cpu, access::write);
  pos[0] = make_real4(-0.5,0,0,0)*distance + randomPosition;
  pos[1] = make_real4(0.5,0,0,0)*distance + randomPosition;
}


Parameters readParameters(std::string datamain);


auto computeMdot(std::shared_ptr<Scheme> stokes, std::shared_ptr<ParticleData> pd){
  auto pos = pd->getPos(access::gpu, access::read);
  auto force = pd->getForce(access::gpu, access::read);
  auto res_gpu = stokes->Mdot(pos.begin(), force.begin(), pos.size());
  std::vector<real3> res(pos.size());
  thrust::copy(res_gpu.begin(), res_gpu.end(), res.begin());
  return res;
}

void setForceOnParticle(std::shared_ptr<ParticleData> pd, int i, real3 f){
  auto force = pd->getForce(access::cpu, access::write);
  force[i] = make_real4(f,0);
}

int main(int argc, char* argv[]){
  auto sys = std::make_shared<System>(argc, argv);
  auto par = readParameters("data.main");
  auto pd = initializeParticles(sys, par);
  auto stokes = createDPStokes(pd, par);
  setForceOnParticle(pd, 0, {1,0,0});
  real dr = 0.1;
  for(real r = 0; r<=par.Lxy*0.5; r+=dr){
    setParticlesInMiddlePlaneAtDistance(pd, r, par.Lxy);
    auto mf = computeMdot(stokes, pd);
    auto mf1 = mf[1]*6*M_PI*par.viscosity;
    std::cout<<std::setprecision(2*sizeof(real))<<r<<" "<<mf1<<std::endl;
  }
  sys->finish();
  return 0;
}




Parameters readParameters(std::string datamain){
  InputFile in(datamain);
  Parameters par;
  in.getOption("Lxy", InputFile::Required)>>par.Lxy;
  in.getOption("H", InputFile::Required)>>par.H;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  in.getOption("nxy", InputFile::Required)>>par.nxy;
  in.getOption("nz", InputFile::Required)>>par.nz;
  in.getOption("w", InputFile::Required)>>par.w;
  //in.getOption("w_d", InputFile::Required)>>par.w_d;
  in.getOption("beta", InputFile::Required)>>par.beta;
  //in.getOption("beta_d", InputFile::Required)>>par.beta_d;
  in.getOption("alpha", InputFile::Required)>>par.alpha;
  //in.getOption("alpha_d", InputFile::Required)>>par.alpha_d;
  in.getOption("wallMode", InputFile::Required)>>par.wallMode;
  in.getOption("numberParticles", InputFile::Required)>>par.numberParticles;

  return par;
}
