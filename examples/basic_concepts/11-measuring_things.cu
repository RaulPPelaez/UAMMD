/* Raul P. Pelaez 2021
   At this point you have the necessary tools to construct and run a simulation using UAMMD.
   Just running the simulation is not very interesting, though. Typically we need to measure something from the simulation state or perform some kind of post-processing.
   In this tutorial we will see how to print particle positions to a file and how to measure the energy of each particle.

   We will start from the LJ liquid simulation we coded in a previous tutorial and will add a couple of functions to measure particle properties.
 */

#include<uammd.cuh>
#include"Integrator/VerletNVT.cuh" //Each Integrator has a particular include
#include"Interactor/PairForces.cuh" //The same is true for Interactors
#include "Interactor/Potential/Potential.cuh" //We will also need this header, which has the LJ potential
#include"utils/InitialConditions.cuh" //For the initLattice function
#include<random>

using namespace uammd;

//Lets take some utilities from the previous tutorials. This block is just a near copy paste from before
//-------------------------------------------------------------------------------------------------------------
//Lets group here a few parameters that our example is going to use. For the time being, lets simply hardcode some values
struct Parameters{
  int numberParticles = 16384;
  real boxSize = 32;
  real friction = 1.0;
  real temperature = 2.0;
  real dt = 0.005; //Time integration step
};

//I like to place these basic UAMMD objects in a struct so it is easy to pass them around
struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  Parameters par;
};

//Creates and returns a UAMMD struct with the basics that we have covered in previous tutorials
UAMMD initializeUAMMD(int argc, char *argv[]){
  UAMMD sim;
  sim.par = Parameters(); //Default parameters
  //Initialize ParticleData
  sim.pd = std::make_shared<ParticleData>(sim.par.numberParticles);
  return sim;
}

//This function stores a vector with the particle positions ordered by id (name) and returns it
std::vector<real4> vector_from_pd_positions(UAMMD sim){
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  auto pos_by_id = thrust::make_permutation_iterator(pos.begin(), id2index);
  return std::vector<real4>(pos_by_id, pos_by_id + pos.size());
}

//This function places particle positions forming an FCC lattice inside a cubic box of size L
void placeParticlesInFCCLattice(UAMMD sim){
  //This function returns a CPU vector of real4 numbers with positions insice a lattice, fcc in this case.
  //But there are a few others you can try, like sc (simple cubic), bcc (body centered cubic), etc
  auto initial = initLattice(make_real3(sim.par.boxSize), sim.par.numberParticles, fcc);
  auto positions = sim.pd->getPos(access::location::cpu, access::mode::write);
  //We will simply copy this vector into positions:
  std::copy(initial.begin(), initial.end(), positions.begin());
}

//This function constructs and returns a Verlet NVT integrator, you will notice it is quite similar to the BD one in the previous example.
std::shared_ptr<Integrator> createVerletNVTIntegrator(UAMMD sim){
  //There are a couple of NVT verlet integration algorithms, see the wiki page for more info.
  using Verlet = VerletNVT::GronbechJensen;
  //Integrators always have a Parameters type inside of them that their constructor needs:
  Verlet::Parameters par;
  //Lets simply copy the parameters we hardcoded at the beginning 
  par.dt = sim.par.dt;
  par.temperature = sim.par.temperature;
  par.friction = sim.par.friction;
  //Verlet will use particle masses in ParticleData if available, otherwise all particles will be assumed to have the mass in par.mass
  //Which defaults to 1 if not set.
  //par.mass = 2.0;
  //You should check the relevant wiki page for information on how to construct each integrator, but 99% of the time you will see
  //The arguments are the same as for this one:
  auto verlet = std::make_shared<Verlet>(sim.pd, par);
  //I like to store Integrators in shared pointers to easily pass them around.
  //You might have noticed that the we are returning a pointer to "Integrator" instead of BD
  //This is ok because all integrators in UAMMD inherit from the base class Integrator. Meaning that they can maskarade as one
  //This means that you can store any integrator in a variable with type std::shared_ptr<Integrator>.
  return verlet;
}

//This function creates and returns a LJ potential, lets define it below the next function
std::shared_ptr<Potential::LJ> createLJPotential(UAMMD sim);

//In a similar way to Integrator theres the Interactor base class, all Interactors can pass as a pointer to Interactor.
//This function will create and return a PairForces module specialized for a LJ potential
std::shared_ptr<Interactor> createLJInteraction(UAMMD sim){
  //LJ potential is encoded under this name
  using LJ = Potential::LJ;
  //We first create the potential using some hardcoded values
  auto lj_pot = createLJPotential(sim);
  //Then we create the PairForces Interactor
  //We specialize the module for the LJ potential
  using PairForces = PairForces<LJ>;
  //Additionally we could pass a second template parameter to indicate the usage of a certain neighbour list type
  //The default is CellList
  //using PairForces = PairForces<LJ, VerletList>;
  //Similar as with Integrators, Interactors always have a Parameter type inside
  //PairForces needs a simulation box, here we create a cubic one using the size hardcoded above.
  PairForces::Parameters par;
  auto box = Box(sim.par.boxSize);
  //A Box is periodic by default, but we could set it to be aperiodic in some direction using this function:
  //box.setPeriodicity(true, true, false); //This box would now be aperiodic in z
  //Alternative we could set the box size to be infinity in some direction:
  //auto box = Box(make_real3(Lx, Ly, Lz)); //Where any dimension can be std::numeric_limits<real>::infinity();
  par.box = box;
  //Similar to an Integrator we can create this module now:
  //Now we have to also pass the lj potential instance as an argument
  auto pairForces = std::make_shared<PairForces>(sim.pd, par, lj_pot);
  return pairForces;
}

//Lets see how to create and configure the LJ potential
std::shared_ptr<Potential::LJ> createLJPotential(UAMMD sim){
  auto pot = std::make_shared<Potential::LJ>();
  //We must instruct the potential with parameters for each interaction pair type
  //LJ will interpret the fouth component of the position as type, in this example all particles have zero type.  
  Potential::LJ::InputPairParameters par;
  par.epsilon = 1.0;
  par.shift = true; //If set to true the LJ potential is shifted, otherwise it is simply trucated
  par.sigma = 1;
  par.cutOff = 2.5*par.sigma;
  //Once the InputPairParameters has been filled accordingly for a given pair of types,
  //a potential can be informed like this:
  pot->setPotParameters(0, 0, par);
  //If other types are needed you can call this as many times as needed:
  //pot->setPotParameters(0, 1, par); //It is considered simmetric, so 0,1 will also set 1,0
  //pot->setPotParameters(1, 1, par);

  //The Potential interface is quite generic and can encode a lot of pair interactions
  //You might need a more sophisticated potential or require more control about the interaction somehow than what this somewhat limited
  // LJ potential offers.
  //In that case check the example advanced/customPotentials.cu, where you will see how to write your own Potential.
  return pot;
}

//-----------------------------------------------------------------------------------------------------------------

//This function will sum the current energies in pd->getEnergy
real sumParticleEnergies(UAMMD sim){
  //Request particle energy container
  auto energy = sim.pd->getEnergy(access::gpu, access::read);
  //sum every element
  real Etot = thrust::reduce(thrust::cuda::par, energy.begin(), energy.end(), real(0.0));
  return Etot;
}

//This function takes an integrator and computes the per particle energy from it.
real measureEnergyPerParticle(UAMMD sim, std::shared_ptr<Integrator> integrator){
  //We start by setting to zero the particle energies:
  {
    auto energy = sim.pd->getEnergy(access::gpu, access::write);
    thrust::fill(thrust::cuda::par, energy.begin(), energy.end(), real(0.0));
  }
  //Then we sum the energy due to the integrator (kinetic energy in the case of verlet):
  //Calling this function will compute, for each, particle the corresponding energy and sum it to pd->getEnergy
  integrator->sumEnergy();
  //Now pd->getEnergy holds the kinetic energy, which we can sum to get the total kinetic energy
  real K = sumParticleEnergies(sim);
  //From this we can compute, for example, the current instantaneous temperature
  real currentTemperature = K/(1.5*sim.par.numberParticles);
  System::log<System::MESSAGE>("Current temperature: %g", currentTemperature);
  //Now we sum the potential energy by requesting each interactor that the integrator holds:
  for(auto i: integrator->getInteractors()){
    i->sum({.force=false, .energy=true, .virial=false});
  }
  //Finally the total energy is the sum of the kinetic and potential energy
  real totalEnergy = sumParticleEnergies(sim);
  real energyPerParticle = totalEnergy/sim.par.numberParticles;
  return energyPerParticle;
}

//This function will print particle positions and velocities to a file called particles.dat
void writeSimulation(UAMMD sim){
  //Lets store a file stream statically
  static std::ofstream out ("particles.dat");
  //Now we print positions and velocities as we have done in previous examples.
  //Notice that any property inside pd can also be printed this way.
  //Keep in mind that it is possible some properties are meaningless in a given simulation.
  //For example, in our current MD simulation Verlet is not using particles energies and thus they are not computed.
  //Unless you are explicitly summing energy as we are doing in this case.
  //On the other hand, if you try to print now particle charges you will get a warning message stating that they have not been used by UAMMD. The contents of pd->getCharge will be undefined.
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  auto vel = sim.pd->getVel(access::cpu, access::read);
  //A permutation iterator takes an iterator and an index iterator and the indirection when accessed
  auto pos_by_id = thrust::make_permutation_iterator(pos.begin(), id2index); //pos_by_id[i] is now equivalent to pos[id2index[i]]
  auto vel_by_id = thrust::make_permutation_iterator(vel.begin(), id2index);
  //Lets separate each frame by a line containing #
  out<<"#\n";
  for(int i = 0; i<sim.par.numberParticles; i++)
    out<<pos_by_id[i]<<" "<<vel_by_id[i]<<std::endl;
}

//We will create a LJ liquid molecular dynamics simulation by mixing a verlet NVT integrator with a LJ potential
int main(int argc, char* argv[]){
  //Lets construct the simulation as before:
  auto sim = initializeUAMMD(argc, argv);
  placeParticlesInFCCLattice(sim);
  auto verlet = createVerletNVTIntegrator(sim);
  auto lj_interaction = createLJInteraction(sim);
  verlet->addInteractor(lj_interaction);

  //In order to thermalize the system we let it run for a while
  int Nrelax = 40000;
  for(int i = 0; i< Nrelax; i++){    
    verlet->forwardTime();
  }

  //Now we compute the system energy and print positions to a file every Nmeasure steps.
  int Nmeasure = 1000;
  int Nsteps = 10000;
  real averageEnergy = 0;
  int averageCounter = 0;
  for(int i = 0; i< Nsteps; i++){    
    verlet->forwardTime();
    if(i%Nmeasure==0){
      real energyPerParticle = measureEnergyPerParticle(sim, verlet);
      //If you have not changed the parameters in this tutorial the simulation has temperature=2 and density = 16384/32^3 = 0.5
      //You can check the literature to see that for this case the total energy in equilibrium should be 0.38331
      averageCounter++;
      averageEnergy += (energyPerParticle - averageEnergy)/averageCounter;
      System::log<System::MESSAGE>("Total system energy: %g (instantaneous) %g (average)", energyPerParticle, averageEnergy);
      writeSimulation(sim);
    }
  }
  //Destroy the UAMMD environment and exit
  sim.pd->getSystem()->finish();
  return 0;
}
