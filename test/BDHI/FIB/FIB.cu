/*Raul P. Pelaez 2018. BDHI::FIB tests

All psi output is adimensional (multipied by rh)



 */
#include"uammd.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/Interactor.cuh"
#include"Integrator/BDHI/FIB.cuh"

#include<iostream>

#include<vector>
#include<fstream>
using namespace uammd;

real temperature, viscosity, rh;
BDHI::FIB::Scheme scheme;
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
real2 pullForce_measureVelocity(real L, real F){
  int N = 1;
  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(L);
  real3 inipos;
  {   
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos.raw()[0] = make_real4(sys->rng().uniform3(-L*0.5, L*0.5),0);
    inipos = make_real3(pos.raw()[0]);
  }
  BDHI::FIB::Parameters par; par.scheme=scheme;
  par.temperature = 0.0;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.001;
  par.box = box;  
  auto bdhi = make_shared<BDHI::FIB>(pd, pg, sys, par);
  {
    auto inter= make_shared<miniInteractor>(pd, pg, sys, "puller");
    inter->F = F;
    bdhi->addInteractor(inter);
  }
  bdhi->forwardTime();
  real veldivM0;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::read);

    real M0 = bdhi->getSelfMobility();
    veldivM0 = (pos.raw()[0].x-inipos.x)/(par.dt*M0);

    if(pos.raw()[0].y > 0 || pos.raw()[0].z > 0){
      sys->log<System::ERROR>("[pullForce] I pulled in the X direction but the particle moved in Y: %e and/or Z:%e", pos.raw()[0].y, pos.raw()[0].z);
    }
  }
  sys->finish();
  real ldivrh = L*bdhi->getHydrodynamicRadius()/rh;
  return make_real2(ldivrh,veldivM0);
}

bool selfMobility_pullForce_test(){

  int NL = 40;

  real L_min = 8*rh;
  real L_max = 128*rh;

  real F = 0.1;

  std::ofstream velout("selfMobility_pullForce.test");
  velout<<"#L  v/(F·M0)"<<endl;
  fori(0, NL){
    real L = L_min + i*((L_max-L_min)/(real)(NL-1));
    real2 tmp = pullForce_measureVelocity(L, F);
    real ldivrh = tmp.x;
    real veldivM0 = tmp.y;
    CudaCheckError();
    velout<<ldivrh<<" "<<(veldivM0/F)<<endl;
  }

  return true;
}


//Two particles, pull one of them, measure the velocity of the other
real pullForce_pairMobility_measureVelocity(real L, real F){
  int N = 2;
  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(L);
  BDHI::FIB::Parameters par; par.scheme=scheme;
  par.temperature = 0.0;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.0001;
  par.box = box;

  
  auto bdhi = make_shared<BDHI::FIB>(pd, pg, sys, par);
  {
    auto inter= make_shared<miniInteractor>(pd, pg, sys, "puller");
    inter->F = F;
    bdhi->addInteractor(inter);
  }
  double prevp = 0;
  double vel;
  std::ofstream out("pairMobility_pullForce.test");
  out<<"#distance   M0/(F·v)"<<endl;
  real minr = 0.01;
  real maxr = L*0.5;
  int nPoints = 2000;

  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos.raw()[0] = make_real4(0,0,0,0);
    pos.raw()[1] = make_real4(minr,0,0,0);
  }
  double  M0 = bdhi->getSelfMobility();
  fori(1,nPoints){
    bdhi->forwardTime();
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::readwrite);
  
      vel = (double(pos.raw()[1].x)-double(prevp))/par.dt;
      double p = prevp;
      double r = minr + i*((maxr-minr)/(double)(nPoints-1));
      pos.raw()[1] = make_real4(r, 0, 0 ,0);
      prevp = pos.raw()[1].x;
      pos.raw()[0] = make_real4(0,0,0,0);
      if(p==0) continue;
      if(isnormal((M0*F)/vel))
	out<<p/rh<<" "<<-(M0*F)/vel<<endl;
    }
  }
  sys->finish();
  return vel/M0;
}

bool deterministicPairMobility_test(){

  real vdivM;

  real F = 1;
  real L = 64*rh/0.91;
  vdivM = pullForce_pairMobility_measureVelocity(L, F);
  CudaCheckError();


  std::ofstream velout("pairMobility_pullForce.M.test");

  velout<<" "<<vdivM/F<<endl;

  
  return true;
}


bool idealParticlesDiffusion(int N, real L){
  auto sys = make_shared<System>();
  sys->rng().setSeed(0x33dbff9^time(NULL));
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(L);
  BDHI::FIB::Parameters par; par.scheme=scheme;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.001;
  par.box = box;
  
  auto bdhi = make_shared<BDHI::FIB>(pd, pg, sys, par);
  real M0 = bdhi->getSelfMobility();
  std::ofstream out("pos.noise.dt"+std::to_string(par.dt)+".L"+std::to_string(L/bdhi->getHydrodynamicRadius())+".M"+std::to_string(M0)+".test");
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    fori(0, pd->getNumParticles()){
      pos.raw()[i] = make_real4(sys->rng().uniform3(-L*0.5, L*0.5), 0);
    }
  }

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
  int N=4096;
  int NL = 10;
  real L_min = 8*rh;
  real L_max = 128*rh;
  fori(0, NL){
    real L = L_min + i*((L_max-L_min)/(real)(NL-1));
    
    idealParticlesDiffusion(N, L);
    CudaCheckError();
  }
}

//Returns L/rh, Var(noise)
real4 singleParticleNoise(real T, real L){
  int N = 1;
  auto sys = make_shared<System>();
  sys->rng().setSeed(1234791);
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(L);
  BDHI::FIB::Parameters par; par.scheme=scheme;
  par.temperature = T;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 1.0;
  par.box = box;

  auto bdhi = make_shared<BDHI::FIB>(pd, pg, sys, par);
  real3 prevp;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos.raw()[0] = make_real4(sys->rng().uniform3(-L*0.5, L*0.5), 0);
    prevp = make_real3(pos.raw()[0]);    
  }
  real3 variance = make_real3(0);
  real3 mean = make_real3(0);
  int nsteps = 10000;
  real selfMobility = bdhi->getSelfMobility();
  fori(0,nsteps){
    bdhi->forwardTime();
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    real4 *p = pos.raw();
    real3 noise = make_real3(p[0]) - prevp;   
    real3 delta = noise - mean;
    mean += delta/real(i+1);
    real3 delta2 = noise - mean;
    variance += delta*delta2;    
    prevp = make_real3(p[0]);
  }
  variance /= real(nsteps);

  real tol = 2/sqrt(nsteps);
  if(abs(variance.x-2*T*selfMobility)>tol or
     abs(variance.y-2*T*selfMobility)>tol or
     abs(variance.z-2*T*selfMobility)>tol){
    sys->log<System::ERROR>("[noiseVariance] Incorrect noise correlation for noiseCorr=%.5e %.5e %.5e, kT·M0=%.5e", variance.x, variance.y, variance.z, T*selfMobility);
  }
  sys->finish();
  real3 tmp = variance/(T*selfMobility);
  return make_real4(L/bdhi->getHydrodynamicRadius(), tmp.x, tmp.y, tmp.z);
}


void noiseVariance_test(){

  real T = temperature;  
  std::ofstream out("noiseVariance.test");
  int NL = 10;
  real L_min = 8*rh;
  real L_max = 128*rh;
  fori(0, NL){
    real L = L_min + i*((L_max-L_min)/(real)(NL-1));  
    real4 noiseCorr = singleParticleNoise(T, L);
    out<<noiseCorr.x<<" "<<noiseCorr.y<<" "<<noiseCorr.z<<" "<<noiseCorr.w<<endl;
  }
  CudaCheckError();



}


using namespace std;
int main( int argc, char *argv[]){

  temperature = std::stod(argv[2]);
  viscosity = std::stod(argv[3]);
  rh = std::stod(argv[4]);
  
  if(strcmp(argv[5], "simple_midpoint")==0){
    scheme=BDHI::FIB::MIDPOINT;
  }
  else if(strcmp(argv[5], "improved_midpoint")==0){
    scheme=BDHI::FIB::IMPROVED_MIDPOINT;
  }  
  else{
    cerr<<"ERROR: select scheme, simple_midpoint or improved_midpoint"<<endl;
    exit(1);
  }
  
  if(strcmp(argv[1], "selfMobility")==0) selfMobility_pullForce_test();
  if(strcmp(argv[1], "pairMobility")==0) deterministicPairMobility_test();
  if(strcmp(argv[1], "selfDiffusion")==0) selfDiffusion_test();
  if(strcmp(argv[1], "noiseVariance")==0) noiseVariance_test();
  
  return 0;
}
