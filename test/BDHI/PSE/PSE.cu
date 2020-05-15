/*Raul P. Pelaez 2018. BDHI::PSE tests

All psi output is adimensional (multipied by rh)



 */
#include"uammd.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/Interactor.cuh"
#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_PSE.cuh"

#include<iostream>

#include<vector>
#include<fstream>
using namespace uammd;

real temperature, viscosity, rh, tolerance;

//Pulls two particles agains each other, or just the first one if there is only one particle
class miniInteractor: public Interactor{
public:
  using Interactor::Interactor;
  real F= 0;
  void sumForce(cudaStream_t st) override{
    auto force = pd->getForce(access::location::cpu, access::mode::write);
    force.raw()[0] = make_real4(F,0,0,0);
    if(pg->getNumberParticles()>1)
      force.raw()[1] = make_real4(-F,0,0,0);
  }
  real sumEnergy() override{return 0;}
};


using std::make_shared;
using std::endl;
//Self mobility deterministic test. Pull a particle with a force, measure its velocity. It should be independent of L and psi.
long double pullForce_measureMobility(real L, real psi){
  int N = 1;
  real F = 1.0;
  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  Box box(L);
  BDHI::PSE::Parameters par;
  par.temperature = 0.0;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 1.0;
  par.box = box;
  par.tolerance = tolerance;
  par.psi = psi;
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  auto M0 = bdhi->getSelfMobility();
  {
    auto inter= make_shared<miniInteractor>(pd, pg, sys, "puller");
    inter->F = F;
    bdhi->addInteractor(inter);
  }
  int Ntest = 200;
  long double vel = 0.0;
  fori(0, Ntest){
    double posprev;
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::write);
      pos[0] = make_real4(make_real3(sys->rng().uniform3(-0.5, 0.5))*L,0);
      posprev = pos[0].x;
    }
    bdhi->forwardTime();
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      vel += (pos[0].x-posprev)/par.dt;
    }
  }
  sys->finish();
  return vel/(F*M0*Ntest);
}

bool selfMobility_pullForce_test(){
  int NL = 20;
  int Npsi = 5;
  real psi_min = 0.5/rh;
  real psi_max = 1.2/rh;
  real L_min = 16*rh;
  real L_max = 128*rh;
  real m;
  forj(0, Npsi){
    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    std::ofstream velout("selfMobility_pullForce.psi"+std::to_string(psi*rh)+".test");
    velout.precision(2*sizeof(real));
    velout<<"#L  v/(F·M0)"<<endl;
    fori(0, NL){
      real L = L_min + i*((L_max-L_min)/(real)(NL-1));
      try{
        m = pullForce_measureMobility(L, psi);
      }
      catch(...){
	continue;
      }
      CudaCheckError();
      velout<<L/rh<<" "<<abs(1.0L-m)<<endl;
    }
  }
  return true;
}


//Two particles, pull one of them, measure the velocity of the other
double pullForce_pairMobility_measureMobility(real L, real psi, real F){
  int N = 2;
  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  Box box(L);
  BDHI::PSE::Parameters par;
  par.temperature = 0.0;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.01;
  par.box = box;
  par.tolerance = tolerance;
  par.psi = psi;
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  {
    auto inter= make_shared<miniInteractor>(pd, pg, sys, "puller");
    inter->F = F;
    bdhi->addInteractor(inter);
  }
  real prevp = 0;
  real vel;
  std::ofstream out("pairMobility_pullForce.psi"+std::to_string(psi*rh)+".test");
  out<<"#distance   M0/(F·v)"<<endl;
  real minr = 0.01;
  real maxr = L*0.5;
  int nPoints = 2000;
  auto M0 = bdhi->getSelfMobility();
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos[0] = make_real4(0,0,0,0);
    pos[1] = make_real4(minr,0,0,0);
  }
  fori(1,nPoints){
    bdhi->forwardTime();
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::readwrite);

      vel = (double(pos.raw()[1].x)-double(prevp))/par.dt;
      real p = prevp;
      real r = minr + i*((maxr-minr)/(real)(nPoints-1));
      pos.raw()[1] = make_real4(r, 0, 0 ,0);
      prevp = pos.raw()[1].x;
      pos.raw()[0] = make_real4(0,0,0,0);
      if(p==0) continue;
      out<<p/rh<<" "<<-M0/(F*double(vel))<<endl;
    }
  }
  sys->finish();
  return vel/(M0*F);
}

bool deterministicPairMobility_test(){
  int Npsi = 10;
  std::vector<real2> velocities(Npsi);
  //Keep psi·a constant
  real psi_min = 0.1/rh;
  real psi_max = 1.0/rh;
  real F = 1;
  real L = 64*rh;
  std::ofstream velout("pairMobility_pullForce.test");
  velout.precision(2*sizeof(real));
  forj(0, Npsi){
    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    real m;
    try{
      m = pullForce_pairMobility_measureMobility(L, psi, F);
    }
    catch(...){
      continue;
    }
    CudaCheckError();
    velout<<" "<<psi*rh<<" "<<m<<endl;
  }
  return true;
}


bool idealParticlesDiffusion(int N, real L, real psi){
  auto sys = make_shared<System>();
  sys->rng().setSeed(0x33dbff9^time(NULL));
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  Box box(L);
  BDHI::PSE::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 1.0;
  par.box = box;
  par.tolerance = tolerance;
  par.psi = psi;
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    fori(0, pd->getNumParticles()){
      pos[i] = make_real4(sys->rng().uniform3(-L*0.5, L*0.5), 0);
    }
  }
  bdhi->forwardTime();
  std::ofstream out("pos.noise.psi"+std::to_string(psi*rh)+".dt"+std::to_string(par.dt)+".test");
  out.precision(2*sizeof(real));

  fori(0,1500){
    bdhi->forwardTime();
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    real4 *p = pos.raw();
    out<<"#"<<endl;
    forj(0,pd->getNumParticles()){
      out<<p[j]<<"\n";
    }
  }
  sys->finish();
  return true;
}

void selfDiffusion_test(){
  int Npsi = 10;
  real psi_min = 0.1/rh;
  real psi_max = 1.0/rh;
  int N=4096;
  real L = 64*rh;
  forj(0, Npsi){
    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    try{
      idealParticlesDiffusion(N, L, psi);
    }
    catch(...){
    }
    CudaCheckError();
  }
}

//Returns Var(noise)
double3 singleParticleNoise(real T, real L, real psi){
  int N = 1;
  auto sys = make_shared<System>();
  sys->rng().setSeed(1234791);
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  Box box(L);
  BDHI::PSE::Parameters par;
  par.temperature = T;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 1.0;
  par.box = box;
  par.tolerance = tolerance;
  par.psi = psi;
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  double3 prevp;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos[0] = make_real4(sys->rng().uniform3(-L*0.5, L*0.5), 0);
    prevp = make_double3(pos[0]);
  }
  double3 variance = double3();
  double3 mean = double3();
  int nsteps = 10000;
  fori(0,nsteps){
    bdhi->forwardTime();
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    real4 *p = pos.raw();
    double3 noise = make_double3(p[0]) - prevp;
    double3 delta = noise - mean;
    mean += delta/real(i+1);
    double3 delta2 = noise - mean;
    variance += delta*delta2;
    prevp = make_double3(p[0]);
  }
  variance /= real(nsteps);
  auto selfMobility = bdhi->getSelfMobility();
  real tol = 2/sqrt(nsteps);
  if(abs(variance.x-2*T*selfMobility)>tol or
     abs(variance.y-2*T*selfMobility)>tol or
     abs(variance.z-2*T*selfMobility)>tol){
    sys->log<System::ERROR>("[noiseVariance] Incorrect noise correlation for psi = %f; noiseCorr=%.5e %.5e %.5e, kT·M0=%.5e", psi, variance.x, variance.y, variance.z, T*selfMobility);
  }
  sys->finish();
  return variance/(T*selfMobility);
}


void noiseVariance_test(){
  int Npsi = 10;
  real psi_min = 0.1/rh;
  real psi_max = 1.0/rh;
  real L = 64*rh;
  real T = temperature;
  std::ofstream out("noiseVariance.test");
  out.precision(2*sizeof(real));
  forj(0, Npsi){
    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    try{
    auto noiseCorr = singleParticleNoise(T, L, psi);
    out<<psi*rh<<" "<<noiseCorr.x<<" "<<noiseCorr.y<<" "<<noiseCorr.z<<" "<<1.0<<endl;
    }
    catch(...){
    }
    CudaCheckError();
  }
}


int main( int argc, char *argv[]){
  temperature = std::stod(argv[2]);
  viscosity = std::stod(argv[3]);
  rh = std::stod(argv[4]);
  tolerance = std::stod(argv[5]);
  if(strcmp(argv[1], "selfMobility")==0) selfMobility_pullForce_test();
  if(strcmp(argv[1], "pairMobility")==0) deterministicPairMobility_test();
  if(strcmp(argv[1], "selfDiffusion")==0) selfDiffusion_test();
  if(strcmp(argv[1], "noiseVariance")==0) noiseVariance_test();
  return 0;
}
