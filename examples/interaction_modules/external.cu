/*Raul P. Pelaez 2021
  Copy pastable example of an External interaction.

  ExternalForces in UAMMD will compute forces and energies acting on particles independently of each other.

  UAMMD interactor modules always need some kind of specialization. 
  For example, UAMMD offers a BondedForces module and provides some specialization (such as FENE or Harmonic bonds).
  You can, however, specialize it with any structure that follows the necessary rules.
  In this code you have some examples with specializations for external interactions.   

  
*/

#include"uammd.cuh"
#include"Interactor/ExternalForces.cuh"
using namespace uammd;

//This struct contains the basic uammd modules for convenience.
struct UAMMD{
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
};


//External potential acting on each particle independently. In this example particles experience gravity
//and there is a wall at the bottom
struct GravityAndWall{
  real g = 1;
  real zwall = -16;
  real kwall = 1;
  GravityAndWall(/*Parameters par*/){
    
  }

  //This function will be called for each particle
  //The arguments will be modified according to what was returned by getArrays below
  __device__ real3 force(real4 pos /*, real mass */){
    real3 f = {0,0,-g};
    //A soft wall that prevents particles from crossing the wall (well they will cross it if determined enough)
    real dist = pos.z - zwall;
    //If particle is near the wall
    if(fabs(dist)<real(10.0)){
      //If particle is above the wall:
      if(dist<0){
	real d2 = dist*dist;
	f += {0,0, kwall/(d2 + real(0.1))};
      }//If particle has crossed the wall lets flip the gravity so it crosses again
      else{
	f.z = g;
      }
    }
    return f;
  }
  
  //Energy can be ommited in the integrators this example use. It defaults to 0.
  //__device__ real energy(real4 pos){ return 0;}

  //Optionally a compute function may be defined and will be called for each particle when ExternalForces::compute is called
  // __device__ void compute(real4 pos /*, real mass */){
  //   int id = blockIdx.x*blockDim.x + threadIdx.x;
  //   if(id==0)printf("hello\n");
  // }
  
  auto getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::gpu, access::read);    
    return pos.begin();
    //If more than one property is needed this would be the way to do it:
    //auto mass = pd->getMass(access::gpu, access::read);
    //return std::make_tuple(pos.begin(), mass.begin());
  }
  
};


//An External interaction can be ParameterUpdatable (see advanced/ParameterUpdatable.cu)
//This allows you to define a family of functions that can make this object aware of changes in the simulation state, such as the simulation time
//In this example we will use it to move the height of the wall according to time
struct MovingWall: public ParameterUpdatable{
  real zwall = -16;
  real kwall = 1;
  MovingWall(/*Parameters par*/){
    
  }

  //This function will be called for each particle
  //The arguments will be modified according to what was returned by getArrays below
  __device__ real3 force(real4 pos /*, real mass */){
    real3 f = real3();
    //A soft wall that prevents particles from crossing the wall (well they will cross it if determined enough)
    real dist = pos.z - zwall;
    //If particle is near the wall
    if(fabs(dist)<real(10.0)){
      //If particle is above the wall:
      if(dist<0){
	real d2 = dist*dist;
	f += {0,0, kwall/(d2 + real(0.1))};
      }
    }
    return f;
  }
  
  auto getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::gpu, access::read);    
    return pos.begin();
  }

  //This function will be called whenever the simulation time is modified (i.e each step)
  virtual void updateSimulationTime(real newTime) override{
    //Now the wall oscillates 
    zwall = -16 + 2*sin(newTime);
  }
};



//You can use this function to create an interactor that can be directly added to an integrator
std::shared_ptr<Interactor> createExternalPotentialInteractor(UAMMD sim){
  //You can pass an instance of the specialization as a shared_ptr, which allows you to modify it from outside the interactor module at any time.
  auto gr = std::make_shared<GravityAndWall>();
  auto ext = std::make_shared<ExternalForces<GravityAndWall>>(sim.pd, sim.sys, gr);
  return ext;  
}

//Initialize UAMMD with some arbitrary particles
UAMMD initializeUAMMD(){
  UAMMD sim;
  sim.sys = std::make_shared<System>();
  constexpr int numberParticles = 100;
  sim.pd = std::make_shared<ParticleData>(sim.sys, numberParticles);
  auto pos = sim.pd->getPos(access::gpu, access::write);
  thrust::fill(thrust::cuda::par, pos.begin(), pos.end(), real4());
  return sim;
}

int main(){
  // auto sim = initializeUAMMD();
  // auto ext = createExternalPotentialInteractor(sim);
  // ext->sumForce(0);
  // ext->sumForce( );
  // ext->compute();
  return 0;
}
