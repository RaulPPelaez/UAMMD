/*Raul P. Pelaez 2019. Dissipative Particle Dynamics example.

Particles start in a random uniform configuration. They evolve following DPD integration.
The parameters are set to reproduce the results in [1]. You can check the RDF* and see that it perfectly reproduces Figure 3 in [1].

Reads some parameters from a file called "data.main.dpd", if not present it will be auto generated with some default parameters.

You can visualize the reuslts with superpunto
*You can use https://github.com/RaulPPelaez/RadialDistributionFunction to compute the RDF
  With the default parameters you can do:
  $ ./dpd | rdf -N 16384 -Nsnapshots 20 -L 16 -nbins 50 -rcut 2.5 > rdf.dat
  To generate Fig 3. in [1]

References:
[1] On the numerical treatment of dissipative particle dynamics and related systems. Leimkuhler and Shang 2015. https://doi.org/10.1016/j.jcp.2014.09.008
 */

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/VerletNVE.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/Potential/DPD.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>


using namespace uammd;

using std::make_shared;
using std::endl;

real3 boxSize;
real dt;
std::string outputFile;
int numberParticles;
int numberSteps, printSteps;
real temperature;
real cutOff;
real gamma_dpd;
real A;


void readParameters(std::shared_ptr<System> sys, std::string file);

int main(int argc, char *argv[]){
  auto sys = make_shared<System>(argc, argv);
  readParameters(sys, "data.main.dpd");

  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(numberParticles, sys);

  Box box(boxSize);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto vel = pd->getVel(access::location::cpu, access::mode::write);
    std::generate(pos.begin(), pos.end(),
		  [&](){
		    return make_real4(sys->rng().uniform3(-box.boxSize.x*0.5,box.boxSize.x*0.5), 0);
		  });
    std::generate(vel.begin(), vel.end(),
		  [&](){
		    return make_real3(sys->rng().gaussian(0, 1),
				      sys->rng().gaussian(0, 1),
				      sys->rng().gaussian(0, 1));
		  });
  }

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  std::ofstream out(outputFile);

  using NVE = VerletNVE;
  NVE::Parameters par;

  par.dt = dt;
  par.initVelocities = false;

  auto verlet = make_shared<NVE>(pd, pg, sys, par);

  {
    using PairForces = PairForces<Potential::DPD>;

    Potential::DPD::Parameters dpd_params;
    dpd_params.cutOff = cutOff;
    dpd_params.temperature = temperature;
    dpd_params.gamma = gamma_dpd;
    dpd_params.A = A;
    dpd_params.dt = par.dt;

    auto pot = make_shared<Potential::DPD>(sys, dpd_params);

    PairForces::Parameters params;
    params.box = box;
    auto pairforces = make_shared<PairForces>(pd, pg, sys, params, pot);

    verlet->addInteractor(pairforces);
  }

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  Timer tim;
  tim.tic();
  //Thermalization
  forj(0, 1000){
    verlet->forwardTime();
  }
  //Run the simulation
  forj(0, numberSteps){
    verlet->forwardTime();

    //Write results
    if(printSteps>0 and j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      fori(0, numberParticles){
	real4 pc = pos[sortedIndex[i]];
	p = box.apply_pbc(make_real3(pc));
	out<<p<<" 0.5 0\n";
      }

    }

    if(j%500 == 0){
      pd->sortParticles();
    }
  }

  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps/totalTime);
  sys->finish();

  return 0;
}

void readParameters(std::shared_ptr<System> sys, std::string file){

  {
    if(!std::ifstream(file).good()){
      std::ofstream default_options(file);
      default_options<<"boxSize 16 16 16"<<std::endl;
      //Number of particles and box size set to match dens=4 as in [1]
      default_options<<"numberParticles 16384"<<std::endl;
      default_options<<"dt 0.01"<<std::endl;
      default_options<<"numberSteps 100000"<<std::endl;
      default_options<<"printSteps 5000"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"temperature 1.0"<<std::endl;
      //Set too match parameters in [1]
      default_options<<"cutOff  1.0"<<std::endl;
      default_options<<"temperature  1.0"<<std::endl;
      default_options<<"gamma  4.0"<<std::endl;
      default_options<<"A  25.0"<<std::endl;

    }
  }

  InputFile in(file, sys);

  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("outputFile", InputFile::Required)>>outputFile;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("cutOff", InputFile::Required)>>cutOff;
  in.getOption("A", InputFile::Required)>>A;
  in.getOption("gamma", InputFile::Required)>>gamma_dpd;



}