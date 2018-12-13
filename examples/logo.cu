/*Raul P. Pelaez 2017. A short range forces example.

This file contains a good example of how UAMMD works and how to configure and launch a simulation.

Runs a Brownian Hydrodynamics simulation with particles starting in a periodic box at low temperature.
Optionally a LJ interaction between the particles can be activated.
  
Needs cli input arguments with a system size, etc, look for "argv"

the input arguments now are:
 ./pse n L dt nsteps printSteps T psi WCA_switch tolerance hydrodynamicRadius

 n -> numberParticles = 2^n
 The particles will start in a cube of size L*0.8

Or just run: ./pse 15 60 0.01 50000 200 0.2 1.0 1.5 5e-2 0.5
for a quick test

You can visualize the reuslts with superpunto

*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_FCM.cuh"
#include"Interactor/ExternalForces.cuh"
#include<fstream>
using std::cerr;
using std::endl;
using std::ofstream;
using std::make_shared;
using namespace uammd;


#include"../.res/logo.c"

struct Faller: public ParameterUpdatable{
  real zwall;
  real k = 1.0;
  real g = 0.1;
  Faller(real zwall):zwall(zwall){
  }
  
  __device__ __forceinline__ real3 force(const real4 &pos){
    real fz = 0;
    if(pos.z<=zwall)fz += k*pow(fabs(pos.z-zwall),2.0f);
    return make_real3(0.0f, 0.0f, fz-g);
  }
  
  std::tuple<const real4 *> getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }

  void updateSimulationTime(real time){
    //You can be aware of changes in some parameters by making the functor ParameterUpdatable
    //and overriding the update function you want, see misc/ParameterUpdatable.h for a list    
  }
};

int main(int argc, char *argv[]){

  int N = 4e5;
  
  auto sys = make_shared<System>(argc, argv);

  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(N, sys);

  real Lx= 256;
  real Ly= 128;
  real Lz =160;
  
  Box box(make_real3(Lx,Ly,Lz));
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    int i = 0;
    while(i<N){
      int w = uammd_image.width;
      int h =  uammd_image.height;
      real2 img_dim = make_real2(w,h);
      real2 tmp;
      do{
	tmp = make_real2(sys->rng().gaussian2(0.5, 1))*img_dim;
      }while(tmp.x<=0 or tmp.x >= w or tmp.y <=0 or tmp.y>=h-1);
      
      int2 pixel= make_int2( tmp.x+0.5, tmp.y+0.5);
      pixel.y = h-pixel.y-1;
      if(uammd_image.pixel_data[4*(pixel.y*w+pixel.x)] != '\0'){
	continue;
      }
      
      real2 pos_t = (tmp/img_dim-0.5);
      real fac = 1.01;
      pos_t = make_real2(fac*pos_t.x*Lx, fac*pos_t.y*Lx*(h/(real)w)+Lz*0.5-fac*0.5*Lx*(h/(real)w));
      pos.raw()[i] = make_real4(pos_t.x, sys->rng().uniform(-0.5, 0.5)*Ly*0.25, pos_t.y, 0);
      
      i++;
    }    
  }
  

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  ofstream out("kk");
  double hydrodynamicRadius =  4;

  real radius=0.7;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::read);

    const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      
    out<<"#Lx="<<Lx*0.5<<";Ly="<<Ly*0.5<<";Lz="<<Lz*0.5<<";"<<endl;
    real3 p;
    fori(0,N){
      real4 pc = pos.raw()[sortedIndex[i]];
      p = make_real3(pc);
      int type = pc.w;
      out<<p<<" "<<radius<<" "<<type<<"\n";
    }
    out<<std::flush;
  }

  

  BDHI::FCM::Parameters par;
  par.temperature = 1.0;
  par.viscosity = 1.0;
  par.hydrodynamicRadius =  hydrodynamicRadius;
  par.dt = 0.01;
  par.box = box;
  par.tolerance = 1e-2;
  //par.psi=0.3;
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::FCM>>(pd, pg, sys, par);
   
  auto extForces = make_shared<ExternalForces<Faller>>(pd, sys,
						       make_shared<Faller>(-box.boxSize.z*0.5));

  bdhi->addInteractor(extForces);
  

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  Timer tim;
  tim.tic();
  int nsteps = std::atoi(argv[1]);
  int printSteps = std::atoi(argv[2]);

  forj(0,nsteps){

    bdhi->forwardTime();

    if(j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");

      auto pos = pd->getPos(access::location::cpu, access::mode::read);

      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      
    out<<"#Lx="<<Lx*0.5<<";Ly="<<Ly*0.5<<";Lz="<<Lz*0.5<<";"<<endl;
      real3 p;
      fori(0,N){
	real4 pc = pos.raw()[sortedIndex[i]];
	p = make_real3(pc);
	int type = pc.w;
	out<<p<<" "<<radius<<" "<<type<<"\n";
      }
    }

    if(j%500 == 0){
      pd->sortParticles();
    }
  }
  
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);

  sys->finish();

  return 0;
}
