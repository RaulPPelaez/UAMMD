/*Raul P. Pelaez 2020-2022. BD  test code.
    Requires a data.main with the parameters (see the readParameters function).
    Intended to be used by test.bash
 */
#include "uammd.cuh"
#include "Integrator/BrownianDynamics.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/NeighbourList/VerletList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/SpectralEwaldPoisson.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"SoftPotential.cuh"
#include"RepulsivePotential.cuh"

using namespace uammd;

struct Parameters{
  real3 boxSize;
  int numberSteps;
  int printSteps;
  int relaxSteps;
  real dt;
  int numberParticles;
  real cutOff;
  real temperature;
  real viscosity;
  real hydrodynamicRadius;
  real tolerance;
  std::string outfile;
  std::string scheme;
  std::string potential;
  std::string readFile;
  bool noInteractors = false;
  //Poisson
  bool useElectro = false;
  real gw,permitivity, split;
  //LJ
  real sigma;
  real epsilon;
  bool shift;
  //Soft
  real U0, d, b;
  //Repulsive
  real r_m;
  int p;

};

struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  Parameters parameters;
};

UAMMD initialize(int argc, char* argv[]);

std::shared_ptr<Integrator> createIntegrator(UAMMD sim);

std::shared_ptr<Poisson> createElectrostaticInteractor(UAMMD sim){
  Poisson::Parameters par;
  par.box = Box(sim.parameters.boxSize);
  par.epsilon = sim.parameters.permitivity;
  par.tolerance = sim.parameters.tolerance;
  par.gw = sim.parameters.gw;
  par.split = sim.parameters.split;
  return std::make_shared<Poisson>(sim.pd, par);
}

std::shared_ptr<Interactor> createShortRangeInteractor(UAMMD sim);

template<class NVT> void runSimulation(UAMMD sim, std::shared_ptr<NVT> mc);

int main(int argc, char *argv[]){
  auto sim = initialize(argc, argv);
  std::shared_ptr<Integrator> mc;
  mc = createIntegrator(sim);
  if(not sim.parameters.noInteractors){
    mc->addInteractor(createShortRangeInteractor(sim));
    if(sim.parameters.useElectro){
      mc->addInteractor(createElectrostaticInteractor(sim));
    }
  }
  Timer tim; tim.tic();
  runSimulation(sim, mc);
  auto totalTime = tim.toc();
  System::log<System::MESSAGE>("mean FPS: %.2f", sim.parameters.numberSteps/totalTime);
  CudaCheckError();
  return 0;
}

Parameters readParameters(std::string datamain);

void initializePositions(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::write);
  auto initial =  initLattice(sim.parameters.boxSize, sim.parameters.numberParticles, sc);
  std::copy(initial.begin(), initial.end(), pos.begin());
  std::generate(pos.begin(), pos.end(),
		[&](){return make_real4(make_real3(sim.pd->getSystem()->rng().uniform3(-0.5, 0.5))*sim.parameters.boxSize, 0);});
}
void initializeCharges(UAMMD sim){
  auto charges = sim.pd->getCharge(access::location::cpu, access::mode::write);
  int i = 0;
  int N = sim.parameters.numberParticles;
  std::generate(charges.begin(), charges.end(), [i,N]() mutable { return pow(-1, ++i);});
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(charges.begin(), charges.end(), g);
}

void initializeFromFile(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::write);
  auto charge = sim.pd->getCharge(access::location::cpu, access::mode::write);
  std::ifstream in(sim.parameters.readFile);
  fori(0, sim.parameters.numberParticles){
    in>>pos[i].x>>pos[i].y>>pos[i].z>>charge[i];
    pos[i].w = 0;
  }
}

UAMMD initialize(int argc, char* argv[]){
  UAMMD sim;
  auto sys =  std::make_shared<System>(argc, argv);
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);
  std::string fileName = "data.main";
  if(argc>1){
    fileName = argv[1];
  }
  sim.parameters = readParameters(fileName);
  sim.pd = std::make_shared<ParticleData>(sim.parameters.numberParticles);
  if(!sim.parameters.readFile.empty()){
    initializeFromFile(sim);
  }
  else{
    initializePositions(sim);
    initializeCharges(sim);
  }
  return sim;
}

template<class BDMethod>
std::shared_ptr<BDMethod> createIntegratorBD(UAMMD sim){
  typename BDMethod::Parameters par;
  par.temperature = sim.parameters.temperature;
  par.viscosity = sim.parameters.viscosity;
  par.hydrodynamicRadius = sim.parameters.hydrodynamicRadius;
  par.dt = sim.parameters.dt;
  return std::make_shared<BDMethod>(sim.pd,  par);
}

std::shared_ptr<Integrator> createIntegrator(UAMMD sim){
  if(sim.parameters.scheme.compare("EulerMaruyama") == 0){
    return createIntegratorBD<BD::EulerMaruyama>(sim);
  }
  else if(sim.parameters.scheme.compare("MidPoint") == 0){
    return createIntegratorBD<BD::MidPoint>(sim);
  }
  else if(sim.parameters.scheme.compare("AdamsBashforth") == 0){
    return createIntegratorBD<BD::AdamsBashforth>(sim);
  }
  else if(sim.parameters.scheme.compare("Leimkuhler") == 0){
    return createIntegratorBD<BD::Leimkuhler>(sim);
  }
  else{
    System::log<System::CRITICAL>("Unrecognized scheme, use EulerMaruyama, MidPoint, AdamsBashforth or Leimkuhler");
    exit(1);
    return nullptr;
  }

}


template <class UsePotential> std::shared_ptr<UsePotential> createPotential(UAMMD sim);

template<> std::shared_ptr<Potential::LJ> createPotential(UAMMD sim){
  auto pot = std::make_shared<Potential::LJ>();
  Potential::LJ::InputPairParameters ppar;
  ppar.epsilon = sim.parameters.epsilon;
  ppar.shift = sim.parameters.shift;
  ppar.sigma = sim.parameters.sigma;
  ppar.cutOff = sim.parameters.cutOff*sim.parameters.sigma;
  pot->setPotParameters(0, 0, ppar);
 return pot;
}


template<> std::shared_ptr<SoftPotential> createPotential(UAMMD sim){
  auto pot = std::make_shared<SoftPotential>();
  SoftPotential::InputPairParameters ppar;
  ppar.cutOff = sim.parameters.cutOff;
  ppar.U0 = sim.parameters.U0;
  ppar.b = sim.parameters.b;
  ppar.d = sim.parameters.d;
  pot->setPotParameters(0, 0, ppar);
 return pot;
}

template<> std::shared_ptr<RepulsivePotential> createPotential(UAMMD sim){
  auto pot = std::make_shared<RepulsivePotential>();
  RepulsivePotential::InputPairParameters ppar;
  ppar.cutOff = sim.parameters.cutOff;
  ppar.U0 = sim.parameters.U0;
  ppar.sigma = sim.parameters.sigma;
  ppar.r_m = sim.parameters.r_m;
  ppar.p = sim.parameters.p;
  System::log<System::MESSAGE>("Repulsive rcut: %g", ppar.cutOff);
  pot->setPotParameters(0, 0, ppar);
 return pot;
}


template<class UsePotential> std::shared_ptr<Interactor> createShortRangeInteractor_impl(UAMMD sim){
  auto pot = createPotential<UsePotential>(sim);
  using SR = PairForces<UsePotential, VerletList>;
  typename SR::Parameters params;
  params.box = Box(sim.parameters.boxSize);
  auto pairForces = std::make_shared<SR>(sim.pd,  params, pot);
  return pairForces;
}

std::shared_ptr<Interactor> createShortRangeInteractor(UAMMD sim){
  if(sim.parameters.potential.compare("LJ") == 0){
    return createShortRangeInteractor_impl<Potential::LJ>(sim);
  }
  else if(sim.parameters.potential.compare("soft") == 0){
    return createShortRangeInteractor_impl<SoftPotential>(sim);
  }
  else if(sim.parameters.potential.compare("repulsive") == 0){
    return createShortRangeInteractor_impl<RepulsivePotential>(sim);
  }

  else{
    System::log<System::CRITICAL>("Unrecognized potential. Use LJ, soft or repulsive");
    return nullptr;
  }
}



template<class NVT>
class SimulationWriter{
  std::ofstream out;
  UAMMD sim;
  std::shared_ptr<NVT> mc;
public:
  SimulationWriter(UAMMD sim, std::shared_ptr<NVT> mc):sim(sim), mc(mc){
    out.open(sim.parameters.outfile);
    out.precision(2*sizeof(uammd::real));
  }

  void writeCurrent(){
    System::log<System::DEBUG1>("[System] Writing to disk...");
    auto par = sim.parameters;
    auto pos = sim.pd->getPos(access::location::cpu, access::mode::read);
    auto charge = sim.pd->getCharge(access::location::cpu, access::mode::read);
    const int * sortedIndex = sim.pd->getIdOrderedIndices(access::location::cpu);
    out<<"#Lx="<<0.5*par.boxSize.x<<";Ly="<<0.5*par.boxSize.y<<";Lz="<<0.5*par.boxSize.z<<";"<<std::endl;
    real3 p;
    fori(0, par.numberParticles){
      real4 pc = pos[sortedIndex[i]];
      p = make_real3(pc);
      int type = 0;
      if(sim.parameters.useElectro){
	type = (charge[sortedIndex[i]]+1)*0.5;
      }
      out<<p<<" 0.5 "<<type<<"\n";
    }
    out<<std::flush;
  }
};

template<class NVT>
void runSimulation(UAMMD sim, std::shared_ptr<NVT> mc){
  SimulationWriter<NVT> write(sim, mc);
  auto par = sim.parameters;
  forj(0, par.relaxSteps){
    mc->forwardTime();
  }
  forj(0, par.numberSteps){
    mc->forwardTime();
    if(par.printSteps > 0 and j%par.printSteps==0){
      write.writeCurrent();
    }
  }
}

Parameters readParameters(std::string datamain){
  InputFile in(datamain);
  Parameters par;
  in.getOption("boxSize", InputFile::Required)>>par.boxSize.x>>par.boxSize.y>>par.boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>par.numberSteps;
  in.getOption("printSteps", InputFile::Required)>>par.printSteps;
  in.getOption("relaxSteps", InputFile::Required)>>par.relaxSteps;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("numberParticles", InputFile::Required)>>par.numberParticles;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>par.hydrodynamicRadius;
  in.getOption("outfile", InputFile::Required)>>par.outfile;
  in.getOption("potential", InputFile::Required)>>par.potential;
  if(par.potential.compare("LJ") == 0){
    in.getOption("sigma", InputFile::Required)>>par.sigma;
    in.getOption("epsilon", InputFile::Required)>>par.epsilon;
    in.getOption("shiftLJ", InputFile::Required)>>par.shift;
    in.getOption("cutOff", InputFile::Required)>>par.cutOff;
  }
  else if(par.potential.compare("soft") == 0){
    in.getOption("U0", InputFile::Required)>>par.U0;
    in.getOption("b", InputFile::Required)>>par.b;
    in.getOption("d", InputFile::Required)>>par.d;
    in.getOption("cutOff", InputFile::Required)>>par.cutOff;
  }
  else if(par.potential.compare("repulsive") == 0){
    in.getOption("U0", InputFile::Required)>>par.U0;
    in.getOption("r_m", InputFile::Required)>>par.r_m;
    in.getOption("p", InputFile::Required)>>par.p;
    in.getOption("sigma", InputFile::Required)>>par.sigma;
    in.getOption("cutOff", InputFile::Required)>>par.cutOff;
  }
  else if(par.potential.compare("none") == 0){
    par.noInteractors = true;
  }

  in.getOption("scheme", InputFile::Required)>>par.scheme;
  in.getOption("readFile", InputFile::Optional)>>par.readFile;
  if(in.getOption("useElectrostatics")){
    par.useElectro = true;
    in.getOption("gw", InputFile::Required)>>par.gw;
    in.getOption("tolerance", InputFile::Required)>>par.tolerance;
    in.getOption("permitivity", InputFile::Required)>>par.permitivity;
    in.getOption("split", InputFile::Required)>>par.split;
  }

  return par;
}

