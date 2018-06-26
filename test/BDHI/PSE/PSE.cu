
#include"uammd.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/Interactor.cuh"
#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_PSE.cuh"

#include<iostream>

#include<vector>
#include<fstream>
using namespace uammd;

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
real pullForce_measureVelocity(real L, real psi, real F){
  int N = 1;
  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(L);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos.raw()[0] = make_real4(0,0,0,0);
  }
  BDHI::PSE::Parameters par;
  par.temperature = 0.0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = 0.001;
  par.box = box;
  par.tolerance = 1e-3;
  par.psi = psi;
  
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  {
    auto inter= make_shared<miniInteractor>(pd, pg, sys, "puller");
    inter->F = F;
    bdhi->addInteractor(inter);
  }
  bdhi->forwardTime();
  real vel;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
  
    vel = pos.raw()[0].x/par.dt;

    if(pos.raw()[0].y > 0 || pos.raw()[0].z > 0){
      sys->log<System::ERROR>("[pullForce] I pulled in the X direction but the particle moved in Y: %e and/or Z:%e", pos.raw()[0].y, pos.raw()[0].z);
    }
  }
  sys->finish();
  return vel;
}

bool selfMobility_pullForce_test(){

  int NL = 40;
  int Npsi = 10;
  std::vector<real3> velocities(NL*Npsi);

  real psi_min = 0.1;
  real psi_max = 1.0;

  real L_min = 8;
  real L_max = 128;

  real F = 0.1;

  forj(0, Npsi){
    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    std::ofstream velout("selfMobility_pullForce.psi"+std::to_string(psi)+".test");
    velout<<"#L  v/(F·M0)"<<endl;
    fori(0, NL){
      real L = L_min + i*((L_max-L_min)/(real)(NL-1));
      velocities[i+NL*j] = make_real3(L, psi, pullForce_measureVelocity(L, psi, F));      
      CudaCheckError();
      real M0 = 1/(6*M_PI)*(1-2.837297/L);
      velout<<L<<" "<<(velocities[i+NL*j].z/(F*M0))<<endl;
    }

  } 
  return true;
}


//Two particles, pull one of them, measure the velocity of the other
real pullForce_pairMobility_measureVelocity(real L, real psi, real F){
  int N = 2;
  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(L);
  BDHI::PSE::Parameters par;
  par.temperature = 0.0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = 0.01;
  par.box = box;
  par.tolerance = 1e-3;
  par.psi = psi;
  
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  {
    auto inter= make_shared<miniInteractor>(pd, pg, sys, "puller");
    inter->F = F;
    bdhi->addInteractor(inter);
  }
  real prevp = 0;
  real vel;
  std::ofstream out("pairMobility_pullForce.psi"+std::to_string(psi)+".test");
  out<<"#distance   M0/(F·v)"<<endl;
  real minr = 0.01;
  real maxr = L*0.5;
  int nPoints = 2000;

  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos.raw()[0] = make_real4(0,0,0,0);
    pos.raw()[1] = make_real4(minr,0,0,0);
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
      double M0 = 1/(6*M_PI)*(1-2.837297/L);
      if(p==0) continue;
      out<<p<<" "<<-M0/(F*double(vel))<<endl;
    }
  }
  sys->finish();
  return vel;
}

bool deterministicPairMobility_test(){
  int Npsi = 10;
  std::vector<real2> velocities(Npsi);

  real psi_min = 0.1;
  real psi_max = 1.0;

  real F = 1;
  real L = 64;  
  forj(0, Npsi){

    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    velocities[j] = make_real2(psi, pullForce_pairMobility_measureVelocity(L, psi, F));
    CudaCheckError();
  }

  std::ofstream velout("pairMobility_pullForce.test");

  real M0 = 1/(6*M_PI);
  fori(0, velocities.size()){
    velout<<" "<<velocities[i].x<<" "<<(velocities[i].y/(F*M0*(1-2.837297/L)))<<endl;
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
  par.temperature = 1.0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = 0.0001;
  par.box = box;
  par.tolerance = 1e-5;
  par.psi = psi;
  
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  std::ofstream out("pos.noise.psi"+std::to_string(psi)+".dt"+std::to_string(par.dt)+".test");
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    fori(0, pd->getNumParticles()){
      pos.raw()[i] = make_real4(sys->rng().uniform3(-L*0.5, L*0.5), 0);
    }
  }

  fori(0,2000){
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
  real psi_min = 0.1;
  real psi_max = 1.0;
  int N=4096;
  real L = 64;  
  forj(0, Npsi){

    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    idealParticlesDiffusion(N, L, psi);
    CudaCheckError();
  }


}

//Returns Var(noise)
real3 singleParticleNoise(real T, real L, real psi){
  int N = 1;
  auto sys = make_shared<System>();
  sys->rng().setSeed(1234791);
  auto pd = make_shared<ParticleData>(N, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(L);
  BDHI::PSE::Parameters par;
  par.temperature = T;
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = 1.0;
  par.box = box;
  par.tolerance = 1e-5;
  par.psi = psi;
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
  real3 prevp;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos.raw()[0] = make_real4(sys->rng().uniform3(-L*0.5, L*0.5), 0);
    prevp = make_real3(pos.raw()[0]);    
  }
  real3 variance = make_real3(0);
  real3 mean = make_real3(0);
  int nsteps = 10000;
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
  real selfMobility = 1/(6*M_PI)*(1.0-2.3872979/L);
  real tol = 2/sqrt(nsteps);
  if(abs(variance.x-2*T*selfMobility)>tol or
     abs(variance.y-2*T*selfMobility)>tol or
     abs(variance.z-2*T*selfMobility)>tol){
    sys->log<System::ERROR>("[noiseVariance] Incorrect noise correlation for psi = %f; noiseCorr=%.5e %.5e %.5e, kT·M0=%.5e", psi, variance.x, variance.y, variance.z, T*selfMobility);
  }
  sys->finish();

  return variance;
}


void noiseVariance_test(){
  int Npsi = 10;
  real psi_min = 0.1;
  real psi_max = 1.0;
  real L = 64;
  real T = 1.0;  
  std::ofstream out("noiseVariance.test");
  
  real selfMobility = 1/(6*M_PI)*(1.0-2.3872979/L);
  forj(0, Npsi){
    real psi = psi_min + j*((psi_max-psi_min)/(real)(Npsi-1));
    real3 noiseCorr = singleParticleNoise(T, L, psi);
    out<<psi<<" "<<noiseCorr.x<<" "<<noiseCorr.y<<" "<<noiseCorr.z<<" "<<selfMobility<<endl;
    CudaCheckError();
  }


}


using namespace std;
int main( int argc, char *argv[]){

  selfMobility_pullForce_test();

  deterministicPairMobility_test();
  
  selfDiffusion_test();

  noiseVariance_test();
  
  return 0;
}
