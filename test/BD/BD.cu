/*Raul P. Pelaez 2020. BD  test code.

  Runs a simulation with particles interacting with LJ.

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
#include"Interactor/Potential/Potential.cuh"
#include"SoftPotential.cuh"

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
  std::string outfile;
  std::string scheme;
  std::string potential;
  //LJ
  real sigma;
  real epsilon;
  bool shift;
  //Soft
  real U0, d, b;

};

struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<ParticleGroup> pg;
  std::shared_ptr<System> sys;
  Parameters parameters;
};

UAMMD initialize(int argc, char* argv[]);

std::shared_ptr<Integrator> createIntegrator(UAMMD sim);

std::shared_ptr<Interactor> createShortRangeInteractor(UAMMD sim);

template<class NVT> void runSimulation(UAMMD sim, std::shared_ptr<NVT> mc);

int main(int argc, char *argv[]){
  auto sim = initialize(argc, argv);
  std::shared_ptr<Integrator> mc;
  mc = createIntegrator(sim);
  mc->addInteractor(createShortRangeInteractor(sim));
  Timer tim; tim.tic();
  runSimulation(sim, mc);
  auto totalTime = tim.toc();
  sim.sys->log<System::MESSAGE>("mean FPS: %.2f", sim.parameters.numberSteps/totalTime);
  sim.sys->finish();
  return 0;
}

Parameters readParameters(std::string datamain, shared_ptr<System> sys);

void initializePositions(UAMMD sim){
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::write);
  auto initial =  initLattice(sim.parameters.boxSize, sim.parameters.numberParticles, sc);
  std::copy(initial.begin(), initial.end(), pos.begin());
}

UAMMD initialize(int argc, char* argv[]){
  UAMMD sim;
  sim.sys = std::make_shared<System>(argc, argv);
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sim.sys->rng().setSeed(seed);
  sim.parameters = readParameters("data.main", sim.sys);
  sim.pd = std::make_shared<ParticleData>(sim.parameters.numberParticles, sim.sys);
  initializePositions(sim);
  sim.pg = std::make_shared<ParticleGroup>(sim.pd, sim.sys, "All");
  return sim;
}

template<class BDMethod>
std::shared_ptr<BDMethod> createIntegratorBD(UAMMD sim){
  typename BDMethod::Parameters par;
  par.temperature = sim.parameters.temperature;
  par.viscosity = sim.parameters.viscosity;
  par.hydrodynamicRadius = sim.parameters.hydrodynamicRadius;
  par.dt = sim.parameters.dt;
  return std::make_shared<BDMethod>(sim.pd, sim.pg, sim.sys, par);
}

std::shared_ptr<Integrator> createIntegrator(UAMMD sim){
  if(sim.parameters.scheme.compare("EulerMaruyama") == 0){
    return createIntegratorBD<BD::EulerMaruyama>(sim);
  }
  else if(sim.parameters.scheme.compare("MidPoint") == 0){
    return createIntegratorBD<BD::MidPoint>(sim);
  }
  else{
    sim.sys->log<System::CRITICAL>("Unrecognized scheme, use EulerMaruyama pr MidPoint");
    exit(1);
    return nullptr;
  }
    
}


template <class UsePotential> std::shared_ptr<UsePotential> createPotential(UAMMD sim);

template<> std::shared_ptr<Potential::LJ> createPotential(UAMMD sim){
  auto pot = std::make_shared<Potential::LJ>(sim.sys);
  Potential::LJ::InputPairParameters ppar;
  ppar.epsilon = sim.parameters.epsilon;
  ppar.shift = sim.parameters.shift;
  ppar.sigma = sim.parameters.sigma;
  ppar.cutOff = sim.parameters.cutOff*sim.parameters.sigma;
  pot->setPotParameters(0, 0, ppar);
 return pot;
}


template<> std::shared_ptr<SoftPotential> createPotential(UAMMD sim){
  auto pot = std::make_shared<SoftPotential>(sim.sys);
  SoftPotential::InputPairParameters ppar;
  ppar.cutOff = sim.parameters.cutOff;
  ppar.U0 = sim.parameters.U0;
  ppar.b = sim.parameters.b;
  ppar.d = sim.parameters.d;
  pot->setPotParameters(0, 0, ppar);
 return pot;
}


template<class UsePotential> std::shared_ptr<Interactor> createShortRangeInteractor_impl(UAMMD sim){
  auto pot = createPotential<UsePotential>(sim);
  using SR = PairForces<UsePotential, VerletList>;
  typename SR::Parameters params;
  params.box = Box(sim.parameters.boxSize);
  auto pairForces = std::make_shared<SR>(sim.pd, sim.pg, sim.sys, params, pot);
  return pairForces;
}

std::shared_ptr<Interactor> createShortRangeInteractor(UAMMD sim){
  if(sim.parameters.potential.compare("LJ") == 0){
    return createShortRangeInteractor_impl<Potential::LJ>(sim);
  }
  else if(sim.parameters.potential.compare("soft") == 0){
    return createShortRangeInteractor_impl<SoftPotential>(sim);
  }
  else{
    System::log<System::CRITICAL>("Unrecognized potential. Use LJ or soft");
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
    sim.sys->log<System::DEBUG1>("[System] Writing to disk...");
    auto par = sim.parameters;
    auto pos = sim.pd->getPos(access::location::cpu, access::mode::read);
    const int * sortedIndex = sim.pd->getIdOrderedIndices(access::location::cpu);
    out<<"#Lx="<<0.5*par.boxSize.x<<";Ly="<<0.5*par.boxSize.y<<";Lz="<<0.5*par.boxSize.z<<";"<<std::endl;
    real3 p;
    fori(0, par.numberParticles){
      real4 pc = pos[sortedIndex[i]];
      p = make_real3(pc);
      out<<p<<" 0.5 0\n";
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

Parameters readParameters(std::string datamain, shared_ptr<System> sys){
  InputFile in(datamain, sys);
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
  in.getOption("cutOff", InputFile::Required)>>par.cutOff;
  in.getOption("potential", InputFile::Required)>>par.potential;
  if(par.potential.compare("LJ") == 0){
    in.getOption("sigma", InputFile::Required)>>par.sigma;
    in.getOption("epsilon", InputFile::Required)>>par.epsilon;
    in.getOption("shiftLJ", InputFile::Required)>>par.shift;
  }
  else if(par.potential.compare("soft") == 0){
    in.getOption("U0", InputFile::Required)>>par.U0;
    in.getOption("b", InputFile::Required)>>par.b;
    in.getOption("d", InputFile::Required)>>par.d;
  }

  in.getOption("scheme", InputFile::Required)>>par.scheme;
  
  return par;
}


