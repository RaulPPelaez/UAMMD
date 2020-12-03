/*Raul P. Pelaez 2019. Poisson test
*/
#include"uammd.cuh"
#include"RepulsivePotential.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"utils/InputFile.h"
#include"Interactor/DoublyPeriodic/DPPoisson.cuh"
#include"Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include <fstream>
#include<random>
using namespace uammd;
using std::make_shared;
using std::endl;


class RepulsiveWall{
  RepulsivePotentialFunctor::PairParameters params;
  real H;
public:
  RepulsiveWall(real H, RepulsivePotentialFunctor::PairParameters ip):H(H),params(ip){}

  __device__ __forceinline__ real3 force(real4 pos){
    real distanceToImage = abs(abs(pos.z) - H * real(0.5))*real(2.0);
    real fz = RepulsivePotentialFunctor::force(distanceToImage * distanceToImage, params) * distanceToImage;
    return make_real3(0, 0, fz*(pos.z<0?real(-1.0):real(1.0)));
  }

  std::tuple<const real4 *> getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }
};
  
struct Parameters{
  int numberParticles;
  real Lxy, H;
  int Nxy = -1;
  int support = -1;
  real numberStandardDeviations = -1;
  real upsampling = -1;
  real temperature;
  real permitivity, permitivityBottom, permitivityTop;
  
  int numberSteps, printSteps, relaxSteps;
  real dt, viscosity, hydrodynamicRadius;

  real gw, tolerance;
  real split = -1;
  real U0, sigma, r_m, p, cutOff;
  
  std::string outfile, readFile, forcefile;

  bool slabMode;
  bool noWall = false;
};


struct UAMMD{
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<thrust::device_vector<real4>> savedPositions;
  Parameters par;
};

Parameters readParameters(std::string fileName, shared_ptr<System> sys);

void initializeParticles(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::write);
  auto charge = sim.pd->getCharge(access::location::cpu, access::mode::write);
  if(sim.par.readFile.empty()){
    std::generate(pos.begin(), pos.end(),
		  [&](){
		    real Lxy = sim.par.Lxy;
		    real H = sim.par.H;
		    real3 p;
		    real pdf;
		    do{
		      p = make_real3(sim.sys->rng().uniform3(-0.5, 0.5))*make_real3(Lxy, Lxy, H);
		      double K= 2.0/H;
		      pdf = 1.0 ; //1.0/pow(cos(K*p.z),2)*pow(cos(K),2);
		  }while(sim.sys->rng().uniform(0, 1) > pdf);
		    return make_real4(p, 0);
		  });
    std::fill(charge.begin(), charge.end(), 1);
  }
  else{
    std::ifstream in(sim.par.readFile);
    fori(0, sim.par.numberParticles){
      in>>pos[i].x>>pos[i].y>>pos[i].z>>charge[i];
      pos[i].w = 0; 
    }
  }
  thrust::copy(pos.begin(), pos.end(), sim.savedPositions->begin());
}

UAMMD initialize(int argc, char *argv[]){
  UAMMD sim;
  sim.sys = std::make_shared<System>(argc, argv);
  std::random_device r;
  auto now = static_cast<long long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  sim.sys->rng().setSeed(now);
  sim.par = readParameters("data.main", sim.sys); 
  sim.pd = std::make_shared<ParticleData>(sim.par.numberParticles, sim.sys);
  sim.savedPositions = std::make_shared<thrust::device_vector<real4>>();
  sim.savedPositions->resize(sim.par.numberParticles);
  initializeParticles(sim); 
  return sim;
}
using BDMethod = BD::Leimkuhler;
std::shared_ptr<BDMethod> createIntegrator(UAMMD sim){
  typename BDMethod::Parameters par;
  par.temperature = sim.par.temperature;
  par.viscosity = sim.par.viscosity;
  par.hydrodynamicRadius = sim.par.hydrodynamicRadius;
  par.dt = sim.par.dt;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<BDMethod>(sim.pd, pg, sim.sys, par);
}

std::shared_ptr<DPPoissonSlab> createElectrostaticInteractor(UAMMD sim){  
  DPPoissonSlab::Parameters par;    
  par.Lxy = make_real2(sim.par.Lxy);
  par.H = sim.par.H;
  DPPoissonSlab::Permitivity perm;
  perm.inside = sim.par.permitivity; 
  perm.top = sim.par.permitivityTop;
  perm.bottom = sim.par.permitivityBottom;
  par.permitivity = perm;
  par.gw = sim.par.gw;

  
  par.tolerance = sim.par.tolerance;
  if(sim.par.split>0){
    par.split = sim.par.split;
  }
  if(sim.par.upsampling > 0){
    par.upsampling=sim.par.upsampling;
  }
  if(sim.par.Nxy > 0){
    if(sim.par.split>0){
      System::log<System::CRITICAL>("ERROR: Cannot set both Nxy and split at the same time");
    }
    real upsampling = sim.par.upsampling > 0?sim.par.upsampling:1.2;
    real hxy = par.Lxy.x/sim.par.Nxy;
    real gt = upsampling*hxy;
    real gw = sim.par.gw;
    real split = sqrt(1/(4*(gt*gt - gw*gw)));
    par.split = split;
    real He = 1.25*sim.par.numberStandardDeviations*gt;
    real H = par.H;
    int Nz = ceil(M_PI*0.5*(H+4*He)/hxy);
    int3 cd = make_int3(Nz, Nz, Nz);
    cd.z = 2*cd.z-2;
    cd = nextFFTWiseSize3D(cd);
    while(cd.z%2 != 0){
      cd.z++;
      cd = nextFFTWiseSize3D(cd);
      //cd.z must be even so that cd.z = ((2*cd.z-2)+2)/2 and 2*cd.z-2 is still a friendly number
    }
    cd.z = (cd.z+2)/2;
    Nz = cd.z;
    par.cells = make_int3(sim.par.Nxy, sim.par.Nxy, Nz);
  }
  par.support = sim.par.support;
  par.numberStandardDeviations = sim.par.numberStandardDeviations;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<DPPoissonSlab>(sim.pd, pg, sim.sys, par);
}

std::shared_ptr<DPPoisson> createElectrostaticInteractorNoSlab(UAMMD sim){  
  DPPoisson::Parameters par;
  par.box = Box(make_real3(sim.par.Lxy, sim.par.Lxy, sim.par.H));
  par.epsilon = sim.par.permitivity;
  par.gw = sim.par.gw;
  par.tolerance = sim.par.tolerance;
  par.split = 0;
  //par.upsampling=2;
  auto pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return std::make_shared<DPPoisson>(sim.pd, pg, sim.sys, par);
}

std::shared_ptr<Interactor> createWallRepulsionInteractor(UAMMD sim){
  RepulsivePotentialFunctor::PairParameters potpar;
  potpar.cutOff2 = sim.par.cutOff*sim.par.cutOff;
  potpar.sigma = sim.par.sigma;
  potpar.U0 = sim.par.U0;
  potpar.r_m = sim.par.r_m;
  potpar.p = sim.par.p;
  return make_shared<ExternalForces<RepulsiveWall>>(sim.pd, sim.sys, make_shared<RepulsiveWall>(sim.par.H, potpar));
}

std::shared_ptr<RepulsivePotential> createPotential(UAMMD sim){
  auto pot = std::make_shared<RepulsivePotential>(sim.sys);
  RepulsivePotential::InputPairParameters ppar;
  ppar.cutOff = sim.par.cutOff;
  ppar.U0 = sim.par.U0;
  ppar.sigma = sim.par.sigma;
  ppar.r_m = sim.par.r_m;
  ppar.p = sim.par.p;
  sim.sys->log<System::MESSAGE>("Repulsive rcut: %g", ppar.cutOff);
  pot->setPotParameters(0, 0, ppar);
 return pot;
}

template<class UsePotential> std::shared_ptr<Interactor> createShortRangeInteractor(UAMMD sim){
  auto pot = createPotential(sim);
  using SR = PairForces<UsePotential>;
  typename SR::Parameters params;
  real Lxy = sim.par.Lxy;
  real H = sim.par.H;
  params.box = Box(make_real3(Lxy, Lxy, H));
  params.box.setPeriodicity(1,1,0);
  auto pairForces = std::make_shared<SR>(sim.pd, sim.sys, params, pot);
  return pairForces;
}

void writeSimulation(UAMMD sim){ 
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::read);
  auto charge = sim.pd->getCharge(access::location::cpu, access::mode::read);
  auto force = sim.pd->getForce(access::location::cpu, access::mode::read);
  static std::ofstream out(sim.par.outfile);
  static std::ofstream outf(sim.par.forcefile);
  Box box(make_real3(sim.par.Lxy, sim.par.Lxy, sim.par.H));
  box.setPeriodicity(1,1,false);
  real3 L = box.boxSize;
  out<<"#Lx="<<L.x*0.5<<";Ly="<<L.y*0.5<<";Lz="<<L.z*0.5<<";"<<std::endl;
  outf<<"#"<<std::endl;
  fori(0, sim.par.numberParticles){
    real3 p = box.apply_pbc(make_real3(pos[i]));
    p.z = pos[i].z;
    real q = charge[i];
    out<<std::setprecision(2*sizeof(real))<<p<<" "<<q<<"\n";
    outf<<std::setprecision(2*sizeof(real))<<force[i]<<"\n";
  }
  out<<std::flush;
}

struct CheckOverlap {
  real H;
  CheckOverlap(real H):H(H){
    
  }

  __device__ bool operator()(real4 p){
    return abs(p.z) >= (real(0.5)*H);
  }  
  
};

bool checkWallOverlap(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::read);
  //int overlappingCharges = thrust::count_if(thrust::cuda::par, pos.begin(), pos.end(), CheckOverlap(sim.par.H));
  //return overlappingCharges > 0;
  auto overlappingPos = thrust::find_if(thrust::cuda::par, pos.begin(), pos.end(), CheckOverlap(sim.par.H));
  return overlappingPos != pos.end();
}

void restoreLastSavedConfiguration(UAMMD sim) {
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::write);  
  thrust::copy(thrust::cuda::par, sim.savedPositions->begin(), sim.savedPositions->end(), pos.begin());  
}

void saveConfiguration(UAMMD sim) {
  auto pos = sim.pd->getPos(access::location::gpu, access::mode::read);
  thrust::copy(thrust::cuda::par, pos.begin(), pos.end(), sim.savedPositions->begin()); 
}


int main(int argc, char *argv[]){  
  auto sim = initialize(argc, argv);
  auto bd = createIntegrator(sim);  
  if(sim.par.slabMode){
    bd->addInteractor(createElectrostaticInteractor(sim));
  }
  else{
    bd->addInteractor(createElectrostaticInteractorNoSlab(sim));
  }
  if(sim.par.U0 > 0){
    bd->addInteractor(createShortRangeInteractor<RepulsivePotential>(sim));
  }
  if(not sim.par.noWall){
    bd->addInteractor(createWallRepulsionInteractor(sim));
  }
  int numberRetries=0;
  int numberRetriesThisStep=0;
  int lastStepSaved=0;
  constexpr int saveRate = 100;
  constexpr int maximumRetries = 1e6;
  constexpr int maximumRetriesPerStep=1e4;
  forj(0, sim.par.relaxSteps){
    bd->forwardTime();
    if(checkWallOverlap(sim)){
      numberRetries++;
      if(numberRetries>maximumRetries){
	throw std::runtime_error("Too many steps with wall overlapping charges detected, aborting run");
      }
      numberRetriesThisStep++;
      if(numberRetriesThisStep>maximumRetriesPerStep){
	throw std::runtime_error("Cannot recover from configuration with wall overlapping charges, aborting run");
      }
      j=lastStepSaved;
      restoreLastSavedConfiguration(sim);
      continue;
    }
    if(j%saveRate==0){
      numberRetriesThisStep = 0;
      lastStepSaved=j;
      saveConfiguration(sim);
    }
  }
  Timer tim;
  tim.tic();
  lastStepSaved=0;
  forj(0, sim.par.numberSteps){
    bd->forwardTime();
    if(checkWallOverlap(sim)){
      numberRetries++;
      if(numberRetries>maximumRetries){
	throw std::runtime_error("Too many steps with wall overlapping charges detected, aborting run");
      }
      numberRetriesThisStep++;
      if(numberRetriesThisStep>maximumRetriesPerStep){
	throw std::runtime_error("Cannot recover from configuration with wall overlapping charges, aborting run");
      }
      j=lastStepSaved;
      restoreLastSavedConfiguration(sim);
      continue;
    }
    if(j%saveRate==0){
      numberRetriesThisStep=0;
      lastStepSaved=j;
      saveConfiguration(sim);
    }
    if(sim.par.printSteps > 0 and j%sim.par.printSteps==0){
      writeSimulation(sim);
      numberRetriesThisStep=0;
      lastStepSaved=j;
      saveConfiguration(sim);
    }
  }
  System::log<System::MESSAGE>("Number of rejected configurations: %d (%g%% of total)", numberRetries, (double)numberRetries/(sim.par.numberSteps + sim.par.relaxSteps)*100.0);
  auto totalTime = tim.toc();
  System::log<System::MESSAGE>("mean FPS: %.2f", sim.par.numberSteps/totalTime);
  return 0;
}

Parameters readParameters(std::string datamain, shared_ptr<System> sys){
  InputFile in(datamain, sys);
  Parameters par;
  in.getOption("Lxy", InputFile::Required)>>par.Lxy;
  in.getOption("H", InputFile::Required)>>par.H;
  in.getOption("numberSteps", InputFile::Required)>>par.numberSteps;
  in.getOption("printSteps", InputFile::Required)>>par.printSteps;
  in.getOption("relaxSteps", InputFile::Required)>>par.relaxSteps;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("numberParticles", InputFile::Required)>>par.numberParticles;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>par.hydrodynamicRadius;
  in.getOption("outfile", InputFile::Required)>>par.outfile;
  in.getOption("forcefile", InputFile::Required)>>par.forcefile;
  in.getOption("U0", InputFile::Required)>>par.U0;
  in.getOption("r_m", InputFile::Required)>>par.r_m;
  in.getOption("p", InputFile::Required)>>par.p;
  in.getOption("sigma", InputFile::Required)>>par.sigma;
  in.getOption("cutOff", InputFile::Required)>>par.cutOff;
  in.getOption("readFile", InputFile::Optional)>>par.readFile;
  in.getOption("gw", InputFile::Required)>>par.gw;
  in.getOption("tolerance", InputFile::Required)>>par.tolerance;
  in.getOption("permitivity", InputFile::Required)>>par.permitivity;
  in.getOption("permitivityTop", InputFile::Required)>>par.permitivityTop;
  in.getOption("permitivityBottom", InputFile::Required)>>par.permitivityBottom;
  in.getOption("split", InputFile::Optional)>>par.split;
  in.getOption("Nxy", InputFile::Optional)>>par.Nxy;
  in.getOption("support", InputFile::Optional)>>par.support;
  in.getOption("numberStandardDeviations", InputFile::Optional)>>par.numberStandardDeviations;
  in.getOption("upsampling", InputFile::Optional)>>par.upsampling;
  in.getOption("slabMode", InputFile::Required)>>par.slabMode;
  if(in.getOption("noWall", InputFile::Optional)){
    par.noWall= true;
  }
  if(par.split < 0 and par.Nxy < 0){
    System::log<System::CRITICAL>("ERROR: I need either Nxy or split");    
  }

  return par;
}



