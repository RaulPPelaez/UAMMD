/*Raul P. Pelaez 2019. The Deserno potential.
  runs a BD simulation using the Deserno Potential
*/
#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
#include"misc/Deserno.cuh"
#include<fstream>
#include"utils/InputFile.h"
#include<vector>

using namespace std;
using namespace uammd;

ullint seed = 0xf31337Bada55D00dULL^time(NULL);
real3 boxSize;

std::string outfile;
std::string init_posfile, harmonicbonds, FENEbonds;
real temperature;
real viscosity, hydrodynamicRadius;
real dt;
real tolerance = 1e-6;

int nsteps, relaxSteps;
int printSteps;

real wc;
void readParameters(shared_ptr<System> sys);

int main(int argc, char * argv[]){  
  auto sys = make_shared<System>(argc, argv);
  sys->rng().setSeed(seed);
  readParameters(sys);
  ifstream in(init_posfile);
  int N;
  if(!in) sys->log<System::CRITICAL>("Could not read %s!", init_posfile.c_str());
  in>>N;
  auto pd = make_shared<ParticleData>(N, sys);
  vector<int> types(N);
  {
    auto ps = pd->getPos(access::location::cpu, access::mode::write);
    real4 * pos = ps.raw();

    fori(0,N){
      in>>pos[i].x>>pos[i].y>>pos[i].z>>pos[i].w;
      types[i] = pos[i].w;
    }
  }
  

  Box box(boxSize);
  
  BD::EulerMaruyama::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = hydrodynamicRadius;
  par.dt = dt;

  auto bd = make_shared<BD::EulerMaruyama>(pd, sys, par);
  {
    Deserno::Parameters params;
    params.box = box;
    params.fileHarmonic = harmonicbonds;
    params.fileFENE = FENEbonds;
    params.wc = wc;
    auto deserno = make_shared<Deserno>(pd, sys, params);

    bd->addInteractor(deserno);
  }
  ofstream out(outfile);

  forj(0,relaxSteps)  bd->forwardTime();
  
  forj(0,nsteps){

    bd->forwardTime();
    if(j%printSteps==0){
    auto ps = pd->getPos(access::location::cpu, access::mode::read);
    real4 * pos = ps.raw();

    out<<"#"<<endl;
    fori(0,N){
      //real3 pi = box.apply_pbc(make_real3(pos[i]));
      real3 pi = make_real3(pos[i]);
      out<<pi<<" 0.5 "<<types[i]<<"\n";
    }
    }
  }
  
  
  
  return 0;
}






void readParameters(shared_ptr<System> sys){
  InputFile in("data.main", sys); 
  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("outfile", InputFile::Required)>>outfile;
  in.getOption("init_posfile", InputFile::Required)>>init_posfile;
  in.getOption("harmonicbonds", InputFile::Required)>>harmonicbonds;
  in.getOption("FENEbonds", InputFile::Required)>>FENEbonds;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("hydrodynamicRadius", InputFile::Required)>>hydrodynamicRadius;
  in.getOption("tolerance", InputFile::Optional)>>tolerance;
  in.getOption("nsteps", InputFile::Required)>>nsteps;
  in.getOption("relaxSteps", InputFile::Required)>>relaxSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("seed", InputFile::Optional)>>seed;
  in.getOption("wc", InputFile::Required)>>wc;
}