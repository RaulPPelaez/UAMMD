/*Raul P. Pelaez 2017. An SPH example.

Particles start in a cube configuration and are inside a prism box (longer in z). There is gravity and the bottom wall of the box is repulsive.
The SPH particles will fall until hitting the bottom and then flow around until the energy from the fall is dissipated.
  
Needs cli input arguments with a system size, etc, look for "argv"

Or just run: ./a.out 14 45 0.01 0.9 20000 500 -20

for a quick test

You can visualize the reuslts with superpunto

  
 */

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/VerletNVE.cuh"
#include"Interactor/SPH.cuh"
#include"Interactor/ExternalForces.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>


using namespace uammd;
using namespace std;

//An HarmonicWall functor to be used in a ExternalForces module (See ExternalForces.cuh)
struct HarmonicWall{
  real zwall;
  real k = 0.05;
  HarmonicWall(real zwall):zwall(zwall){
  }
  
  __device__ __forceinline__ real3 force(const real4 &pos){

    real3 rij = make_real3(0,0, pos.z- zwall);
    real r2 = dot(rij, rij);
    
    real invr2 = real(1.0)/(r2+0.001f);
    real3 f = make_real3(0, 0, k);
    if(r2 < 8.0f && r2> 0.1f && pos.z > 0.0f){
      f.z -= 1.0f*invr2*invr2;
    }
    if(pos.z>zwall){
      f.z -= k;
    }

    return f;
  }

  //If this function is not present, energy is assumed to be zero
  // __device__ __forceinline__ real energy(const real4 &pos){
    
  //   return real(0.5)*k*pow(pos.z-zwall, 2);
  // }

  std::tuple<const real4 *> getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }


};



int main(int argc, char *argv[]){

  if(argc<8){
    std::cerr<<"ERROR, I need some parameters!!\nTry to run me with:\n./"<<argv[0]<<" 14 45 0.01 0.9 20000 500 -20"<<std::endl;
    exit(1);
  }
  int N = pow(2,atoi(argv[1]));//atoi(argv[1]));

  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.
  
  auto sys = make_shared<System>();

  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL;
  sys->rng().setSeed(seed);

  //ParticleData stores all the needed properties the simulation will need.
  //Needs to start with a certain number of particles, which can be changed mid-simulation
  //If UAMMD is to be used as a plugin for other enviroment or custom code, ParticleData should accept references to
  // properties allocated and handled by the user, this is a non-implemented work in progress as of now though.
  auto pd = make_shared<ParticleData>(N, sys);

  //Some modules need a simulation box (i.e PairForces for the PBC)
  real3 L =make_real3(std::stod(argv[2]));
  L.z = 2*L.y;
  Box box(L);//std::stod(argv[2]));
  
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);    

    auto initial =  initLattice(make_real3(box.boxSize.x*std::stod(argv[4])), N, fcc);
    
    //Start in a cubic lattice, pos.w contains the particle type
    //auto initial = cubicLattice(box.boxSize, N);
    
    fori(0,N){
      pos.raw()[i] = initial[i]+make_real4(0,0, -std::stod(argv[7]), 0);
      //Type of particle is stored in .w
      //pos.raw()[i].w = sys->rng().uniform(0,1) > std::stod(argv[7])?0:1;
      pos.raw()[i].w = 0;
    }    
  }
  

  //Modules can work on a certain subset of particles if needed, the particles can be grouped following any criteria
  //The builtin ones will generally work faster than a custom one. See ParticleGroup.cuh for a list
  
  //A group created with no criteria will contain all the particles  
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  
  ofstream out("kk");
  
  //Some modules need additional parameters, in this case VerletNVT needs dt, temperature...
  //When additional parameters are needed, they need to be supplied in a form similar to this:
  {
    auto vel = pd->getVel(access::location::cpu, access::mode::write);
    fori(0, N) vel.raw()[i] = make_real3(0);
  }
      
  VerletNVE::Parameters par;
  par.dt = std::stod(argv[3]);
  //If set to true (default), VerletNVE will compute Energy at step 0 and modify the velocities accordingly
  par.initVelocities = false;
  auto verlet = make_shared<VerletNVE>(pd, pg, sys, par);

  //Harmonic walls acting on different particle groups
  //This two interactors will cause particles in group pg2 to stick to a wall in -Lz/4
  //And the ones in pg3 to +Lz/4
  auto extForces = make_shared<ExternalForces<HarmonicWall>>(pd, pg, sys,
							     make_shared<HarmonicWall>(box.boxSize.z*0.5));
  
  //Add interactors to integrator.
  verlet->addInteractor(extForces);

  
  SPH::Parameters params;
  params.box = box;  //Box to work on
  //These are the default parameters,
  //if any parameter is not present, it will revert to the default in the .cuh
  params.support = 1.0;
  params.viscosity = 50.0;
  params.gasStiffness = 100.0;
  params.restDensity = 0.4;

  auto sph = make_shared<SPH>(pd, pg, sys, params);


  
  //You can add as many modules as necessary
  verlet->addInteractor(sph);

  //You can issue a logging event like this, a wide variety of log levels exists (see System.cuh).
  //A maximum log level is set in System.cuh, every logging event with a level superior to the max will result in
  // absolutely no overhead, so dont be afraid to write System::DEBUGX log calls.
  sys->log<System::MESSAGE>("RUNNING!!!");

  //Ask ParticleData to sort the particles in memory!
  //It is a good idea to sort the particles once in a while during the simulation
  //This can increase performance considerably as it improves coalescence.
  //Sorting the particles will cause the particle arrays to change in order and (possibly) address.
  //This changes will be informed with signals and any module that needs to be aware of such changes
  //will acknowedge it through a callback (see ParticleData.cuh).
  pd->sortParticles();
        
  Timer tim;
  tim.tic();
  int nsteps = std::atoi(argv[5]);
  int printSteps = std::atoi(argv[6]);
  //Run the simulation
  forj(0,nsteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    verlet->forwardTime();

    //Write results
    if(j%printSteps==1)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      //continue;
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      fori(0,N){
	real4 pc = pos.raw()[sortedIndex[i]];
	p = box.apply_pbc(make_real3(pc));
	int type = pc.w;
	out<<p<<" "<<0.5*(type==1?2:1)<<" "<<type<<endl;
      }

    }    
    //Sort the particles every few steps
    //It is not an expensive thing to do really.
    if(j%500 == 0){
      pd->sortParticles();
    }
  }
  
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
  //sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();

  return 0;
}