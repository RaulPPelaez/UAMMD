/*Raul P. Pelaez 2019. SPH Dambreak problem example

  A cube of fluid particles falling from the upper left corner of a closed box, a wave is formed when the fluid front reaches the right wall. the fluid is modelated with Smoothed Particle Hydrodynamics.

  This executable looks for a "dam.options" file with parameters.
  If not found the file will be auto generated with some default options.

*/
#include"uammd.cuh"
#include"Integrator/VerletNVE.cuh"
#include"Interactor/SPH.cuh"
#include"Interactor/ExternalForces.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"

#include<fstream>
using namespace uammd;



real in_Kwall;
real g;
//Gravity and wall interactions
struct Gravity{
  real m_g;
  real m_Kwall;
  real3 L;
  Gravity(real3 L ): m_g(g),m_Kwall(in_Kwall), L(L*0.8){}
  __device__ __forceinline__ real3 force(const real4 &pos){
    real3 f = real3();
    if(pos.x<-L.x*0.5) f.x = m_Kwall;  if(pos.x>L.x*0.5) f.x = -m_Kwall;
    if(pos.y<-L.y*0.5) f.y = m_Kwall;  if(pos.y>L.y*0.5) f.y = -m_Kwall;
    if(pos.z<-L.z*0.5) f.z = m_Kwall;  if(pos.z>L.z*0.5) f.z = -m_Kwall;

    f.z += m_g;
    return f;
  }

  auto getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return pos.begin();
  }
};



using std::make_shared;
using std::cout;
using std::endl;

int numberParticles, numberSteps, printSteps;
real3 boxSize;
real dt;
string outputFile;
real viscosity, gasStiffness, support, restDensity;

void readParameters(std::shared_ptr<System> sys, std::string file);
int main(int argc, char *argv[]){
  auto sys = make_shared<System>(argc, argv);
  readParameters(sys, "dam.options");

  ullint seed = 0xf31337Bada55D00dULL;
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(numberParticles, sys);
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  Box box(boxSize);
  {//Initial positions
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto L = boxSize*0.8;
    real3 l = L;
    l.x *= 0.25;
    auto initial =  initLattice(l, numberParticles, fcc);
    std::transform(initial.begin(), initial.end(),
		   pos.begin(), [&](real4 p){p+=make_real4(L.x/8.0-L.x*0.49,0,0,0); p.w=0; return p;});
  }

  std::ofstream out(outputFile);
  { //Particles start at rest
    auto vel = pd->getVel(access::location::cpu, access::mode::write);
    std::fill(vel.begin(), vel.end(), real3());
  }

  //SPH needs some simplectic integrator without a thermostat
  VerletNVE::Parameters par;
  par.dt = dt;
  par.initVelocities = false; //Do not modify initial velocities

  auto verlet = make_shared<VerletNVE>(pd, sys, par);

  { //Add SPH interactor
    SPH::Parameters params;
    params.support = support;
    params.viscosity = viscosity;
    params.gasStiffness = gasStiffness;
    params.restDensity = restDensity;
    params.box = box;
    auto sph = make_shared<SPH>(pd, pg, sys, params);
    verlet->addInteractor(sph);
  }

  {//Add gravity+wall interactor
    auto gravity = make_shared<ExternalForces<Gravity>>(pd, pg, sys, make_shared<Gravity>(boxSize));
    verlet->addInteractor(gravity);
  }

  sys->log<System::MESSAGE>("RUNNING!!!");
  pd->sortParticles();
  Timer tim;
  tim.tic();
  forj(0,numberSteps){
    verlet->forwardTime();
    if(printSteps and j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      fori(0, numberParticles){
	real4 pc = pos.raw()[sortedIndex[i]];
	//p = box.apply_pbc(make_real3(pc));
	p = make_real3(pc);
	int type = 5;
	out<<p<<" "<<2<<" "<<type<<"\n";
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
      default_options<<"boxSize 150 32 50"<<std::endl;
      default_options<<"numberParticles 16000"<<std::endl;
      default_options<<"dt 0.007"<<std::endl;
      default_options<<"numberSteps 100000"<<std::endl;
      default_options<<"printSteps 100"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"viscosity 10"<<std::endl;
      default_options<<"gasStiffness 60"<<std::endl;
      default_options<<"support 2.4"<<std::endl;
      default_options<<"restDensity 0.3"<<std::endl;
      default_options<<"Kwall  20"<<std::endl;
      default_options<<"g -1"<<std::endl;
    }
  }

  InputFile in(file, sys);

  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("outputFile", InputFile::Required)>>outputFile;
  in.getOption("Kwall", InputFile::Required)>>in_Kwall;
  in.getOption("g", InputFile::Required)>>g;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
  in.getOption("gasStiffness", InputFile::Required)>>gasStiffness;
  in.getOption("support", InputFile::Required)>>support;
  in.getOption("restDensity", InputFile::Required)>>restDensity;

}
