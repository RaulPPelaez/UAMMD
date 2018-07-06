/*Raul P. Pelaez 2017.
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

  auto sys = make_shared<System>();
  readParameters(sys);
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


void readParameters(shared_ptr<System> sys){

  InputFile in("data.main.lbm", sys);

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
