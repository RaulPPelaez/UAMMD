/*Raul P. Pelaez 2019. Lattice Boltzmann example, work in progress, not in a useful state atm
*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
#include"Integrator/Hydro/LBM.cuh"
#include"utils/InputFile.h"

#include<fstream>


using namespace uammd;
using namespace std;

real3 boxSize;
int numberSteps;
int printSteps;
int N;
real particleRadius;

real soundSpeed;
real viscosity;
real relaxTime;
real dt;
int3 cells;
void readParameters(shared_ptr<System> sys);

int main(int argc, char *argv[]){

  auto sys = make_shared<System>(argc, argv);
  readParameters(sys, "data.main.lbm");
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(N, sys);


  Box box(boxSize);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto radius = pd->getRadius(access::location::cpu, access::mode::write);

    fori(0,N){
      pos.raw()[i] = make_real4(sys->rng().uniform3(-boxSize.x*0.5, boxSize.x*0.5), 0);
      pos.raw()[i].z = 0;
      radius.raw()[i] = particleRadius;
    }
    //pos.raw()[0] = make_real4(-boxSize.x*0.5+20,0,0,0);

  }

  ofstream out("kk");
  using LBM = Hydro::LBM::D3Q19;

  LBM::Parameters par;
  par.dt = dt;
  par.box = box;
  par.ncells = cells;
  par.soundSpeed = soundSpeed;
  par.relaxTime = relaxTime;
  par.viscosity = viscosity;


  auto lbm = make_shared<LBM>(pd, sys, par);
  sys->log<System::MESSAGE>("RUNNING!!!");
  lbm->writePNG();
  Timer tim;
  tim.tic();
  //Run the simulation
  forj(0,numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    lbm->forwardTime();

    //Write results
    if(printSteps >0 && j%printSteps==0)
    {
      sys->log<System::DEBUG>("[System] Writing to disk...");
      lbm->writePNG();
    }
  }

  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps/totalTime);
  //sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();

  return 0;
}


void readParameters(shared_ptr<System> sys, std::string file){

  {
    if(!std::ifstream(file).good()){
      std::ofstream default_options(file);
      default_options<<"boxSize   200 75 100"<<std::endl;
      default_options<<"numberSteps  10000"<<std::endl;
      default_options<<"printSteps   3"<<std::endl;
      default_options<<"viscosity 9000"<<std::endl;
      default_options<<"numberParticles 3"<<std::endl;
      default_options<<"particleRadius 10"<<std::endl;
      default_options<<"dt    1"<<std::endl;
      default_options<<"cells      200 75 100"<<std::endl;
      default_options<<"soundSpeed 10"<<std::endl;
      default_options<<"relaxTime 1"<<std::endl;

    }
  }
  InputFile in(file, sys);

  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("relaxTime", InputFile::Required)>>relaxTime;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
  in.getOption("numberParticles", InputFile::Required)>>N;
  in.getOption("particleRadius", InputFile::Required)>>particleRadius;
  in.getOption("soundSpeed", InputFile::Required)>>soundSpeed;
  in.getOption("cells", InputFile::Required)>>cells.x>>cells.y>>cells.z;




}
