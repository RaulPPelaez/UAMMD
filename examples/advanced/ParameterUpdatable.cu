/*Raul P. Pelaez 2021. ParameterUpdatable[1] example.
  
  When a module (and Integrator[2] or Interactor[3]) modifies some simulation parameter, such as the simulation box or the simulation time, it will communicate this change to the modules it is in charge of using the ParameterUpdatable interface.
  For example an Integrator will inform all its Interactors every time thesimulation time changes (AKA each step).

  On the other hand, Interactors will usually issue their own communications in addition to forwarding the ones received by the Integrators.
  For example the PairForces[4] Interactor needs a Potential[5], which might be ParameterUpdatable, any communication transmitted by the Integrator to PairForces will be forwarded to the Potential. So a Potential can be aware of the changes in, for example, the simulation time.

  See ParameterUpdatable.h for a list of parameters that are processed via this interface, at the time of writing this example these are:
  real TimeStep
  real SimulationTime
  Box Box
  real Temperature
  real Viscosity


We will see three examples of situations where we can take advantage of this interface:

 -An ExternalForces potential that changes with time
 -A small Interactor that needs to know the current simulation time
 -A PairForces Potential that needs to now if and when the temperature or the simulation time changes
 */


#include"uammd.cuh"
#include"Integrator/VerletNVT.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/PairForces.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>

using namespace uammd;
using std::make_shared;
using std::endl;


//An ExternalForces potential enconding and oscillating wall.
//The height of the wall will oscilate with time
//The class needs to inherit from ParameterUpdatable
struct MovingWall: public ParameterUpdatable{
  real zwall = -20;
  real kwall = 2*M_PI*0.5;
  real amplitude = 3;
  real k = 0.1;
  real time = 0;
  __device__ real3 force(const real4 &pos){
    return make_real3(0, 0, -k*(pos.z-(zwall+amplitude*sin(kwall*time))));
  }
  
  std::tuple<const real4 *> getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }

  //This function is part of the ParameterUpdatable interface
  //The Integrator will call ExternalForces::updateSimulationTime with the current time each step
  //Then, ExternalForces will call this function below.
  virtual void updateSimulationTime(real newTime) override{
    time = newTime;
  }
};


//Interactor is special because it already includes ParameterUpdatable.
//so any class inheriting from Interactor is automatically also ParameterUpdatable.
class SmallInteractor: public Interactor{
  real currentTime = 0;
public:
  //A simple constructor that just initializes the Interactor
  SmallInteractor(shared_ptr<ParticleData> pd, shared_ptr<System> sys):Interactor(pd, sys, "IrrelevantName"){}

  //Part of the Interactor interface, needs to be defined to compile
  virtual void sumForce(cudaStream_t st) override{
    //As an example, we will make particles experience some gravity starting at time=10.
    constexpr real startingTime = 10;
    if(currentTime<startingTime){
      //Simply sum {0,0,-1,0} to every particle's force
      constexpr real4 gravity = {0,0,-1,0};
      auto forces = pd->getForce(access::location::gpu, access::mode::readwrite);      
      thrust::transform(thrust::cuda::par.on(st),
       			forces.begin(), forces.end(),
			thrust::make_constant_iterator(gravity),
			forces.begin(),
			thrust::plus<real4>());
    }
  }

  //Part of the ParameterUpdatable interface, this is a non-pure virtual function, meaning that its existance is optional
  //If it does not exists calls to SmallInteractor::updateSimulationtime will be silently ignored.
  virtual void updateSimulationTime(real newTime) override{
    //Just store the new time
    currentTime = newTime;
  }  
};

__device__ real lj_force(real epsilon, real r2){
  const real invr2 = real(1.0)/r2;
  const real invr6 = invr2*invr2*invr2;
  const real fmoddivr = epsilon*(real(-48.0)*invr6 + real(24.0))*invr6*invr2;
  return fmoddivr;
}

//A simple LJ Potential for PairForces, the interaction starts being weak and but gets stronger with time
//see the example customPotentials.cu for more information about Potentials
struct SimpleLJ: public ParameterUpdatable{
  real rc = 2.5;
  //Epsilon and temperature will be provided via the ParameterUpdatable interface
  real epsilon = 0; //the LJ interaction strength
  real temperature = 0;
  //A host function returning the maximum required cut off for the interaction
  real getCutOff(){
    return rc;
  }
  
  struct Force{
    real4 *force;
    Box box;
    real rc;
    real epsilon;
    Force(Box i_box, real i_ep, real i_rc, real4* i_force):
      box(i_box),
      rc(i_rc),
      epsilon(i_ep),
      force(i_force){}
    
    //For each pair computes and returns the LJ force
    __device__ real3 compute(real4 pi, real4 pj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      if(r2>0 and r2< rc*rc){
	return lj_force(epsilon, r2)*rij;
      }
      return real3();
    }

    __device__ void set(int id, real3 total){      
      force[id] += make_real4(total);
    }
  };

  //Return an instance of the Transverser that will compute only the force (because the energy pointer is null)
  Force getForceTransverser(Box box, std::shared_ptr<ParticleData> pd){
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();    
    return Force(box, epsilon, rc, force);
  }

  virtual void updateTemperature(real newTemperature) override{
    this->temperature = newTemperature;
  }
  
  virtual void updateSimulationTime(real newTime) override{
    //The LJ strength will start at 0.5 and increase slowly up to 1     
    this->epsilon = (1-0.5*exp(-0.1*newTime));
  }  
};



template<class UsePotential>
shared_ptr<Interactor> createPairForcesWithPotential(Box box, shared_ptr<ParticleData> pd, shared_ptr<System> sys){
  using PF = PairForces<UsePotential>;
  typename PF::Parameters par;
  par.box = box;
  auto pot = std::make_shared<UsePotential>();
  return std::make_shared<PF>(pd, sys, par, pot);  
}


//Construct an UAMMD simulation with all the interactions defined above
int main(int argc, char *argv[]){
  int numberParticles = 1<<14;
  auto sys = make_shared<System>(argc, argv);
  auto pd = make_shared<ParticleData>(numberParticles, sys);
  Box box({32,32,40});
  //Initial positions
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    //Start in an fcc lattice
    auto initial = initLattice(box.boxSize, numberParticles, fcc);
    std::copy(initial.begin(), initial.end(), pos.begin());
  }
  using NVT = VerletNVT::GronbechJensen;
  NVT::Parameters par;
  par.temperature = 0.6;
  par.dt = 0.005;
  par.friction = 1.0;
  auto verlet = make_shared<NVT>(pd, sys, par);
  auto ext = make_shared<ExternalForces<MovingWall>>(pd, sys);  
  verlet->addInteractor(ext);
  auto small = make_shared<SmallInteractor>(pd, sys);
  verlet->addInteractor(small);
  auto lj = createPairForcesWithPotential<SimpleLJ>(box, pd, sys);
  verlet->addInteractor(lj);

  sys->log<System::MESSAGE>("RUNNING");
  std::ofstream out("/dev/stdout");
  int numberSteps = 1000000;
  int printSteps  = 1000;
  forj(0, numberSteps){
    verlet->forwardTime();
    if(printSteps>0 and j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      fori(0, numberParticles){
	real4 pc = pos[sortedIndex[i]];
	p = box.apply_pbc(make_real3(pc));
	int type = pc.w;
	out<<p<<" "<<0.5*(type==1?2:1)*pow(2,1/6.)<<" "<<type<<"\n";
      }

    }
  }
  sys->finish();
  return 0;
}


