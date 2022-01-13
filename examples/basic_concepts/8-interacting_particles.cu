/* Raul P. Pelaez 2021
   Adding interactions to an Integrator.

   At this point you already know how to initialize an UAMMD environment, modify particle properties and move particles using Integrators.
   In this tutorial we will cover the last of the basic UAMMD modules: Interactor
   Interactors can be added to Integrators and encode, well, interactions.
   For example the Interactor called BondedForces allows to join particles pairs with springs.
   An Interactor is able to compute forces and energies resulting from a certain interaction.

   Look in the wiki pages or in other examples for a list of all available Interactors, we will cover one here, PairForces, which computes forces and energies between neighbouring particles ( such as a LJ potential interaction).
   
   We also need to initialize particles differently from previous tutorials, since we are going to be using a LJ interaction 
     it is a bad idea to start with a random distribution of particles. UAMMD offers a function called initLattice that we can leverage here.
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
  int numberParticles = 1e5;
  real boxSize = 64;
  real friction = 1.0;
  real temperature = 1.0;
  real dt = 0.001; //Time integration step
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
  //Initialize System and ParticleData
  sim.pd = std::make_shared<ParticleData>(sim.par.numberParticles);
  return sim;
}

//Prints the positions of particle with names from 0 to 9 using what we saw in the previous tutorial
void printFirst10Particles(UAMMD sim){
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  //A permutation iterator takes an iterator and an index iterator and the indirection when accessed
  auto pos_by_id = thrust::make_permutation_iterator(pos.begin(), id2index); //pos_by_id[i] is now equivalent to pos[id2index[i]]
  std::cout<<"Particles with names from 0 to 9:"<<std::endl;
  std::cout<<"Name\tposition"<<std::endl;
  for(int i = 0; i<10; i++)
    std::cout<<i<<"\t"<<pos_by_id[i]<<std::endl;
}

//This function stores a vector with the particle positions ordered by id (name) and returns it
auto vector_from_pd_positions(UAMMD sim){
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  auto pos_by_id = thrust::make_permutation_iterator(pos.begin(), id2index);
  return std::vector<real4>(pos_by_id, pos_by_id + pos.size());
}
//-----------------------------------------------------------------------------------------------------------------


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
auto createVerletNVTIntegrator(UAMMD sim){
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
auto createLJInteraction(UAMMD sim){
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

//We will create a LJ liquid molecular dynamics simulation by mixing a verlet NVT integrator with a LJ potential
int main(int argc, char* argv[]){
  auto sim = initializeUAMMD(argc, argv);
  placeParticlesInFCCLattice(sim);
  //Now lets create an Integrator just like before, to change things around, lets use a constant temperature verlet integrator: 
  auto verlet = createVerletNVTIntegrator(sim);
  //Lets also create a PairForces Interactor set to a short range LJ potential
  auto lj_interaction = createLJInteraction(sim);
  //Now we add the interaction to the verlet Integrator:
  verlet->addInteractor(lj_interaction);

  //Interactors are relatively small interface that only expose the following functions:
  //sumForce(); //Sums to pd->getForce() the forces acting on each particle due to the interaction
  //sumEnergy(); //Sums to pd->getEnergy() the energies acting on each particle due to the interaction
  //sumForceEnergy(); //Can be implemented as {sumForce(); sumEnergy();}, computes both at the same time.
  
  //When we call forwardTime now verlet will ask lj_interaction for the forces acting on each particle and use them when integrating the movement.
  //Note that this integrator does not need the particle energies, so they are not computed. If you want them, you may call lj_interaction->sumEnergy(); But remember to set the contents of pd->getEnergy() to zero first!

  int Nsteps = 1000;
  for(int i = 0; i< Nsteps; i++){
    verlet->forwardTime();
  }
  //And there you go, a LJ liquid MD simulation.
  //If we want to print the positions we can do it at any point like in the previous example,
  //Lets for example write the positions of the first 10 particles:
  printFirst10Particles(sim);
  
  //See the wiki for a list of all the Interactors available and how to use them.
  //In the examples/interaction_modules folder you can also see copy-pastable examples for each one.

  
  
  //Destroy the UAMMD environment and exit
  sim.pd->getSystem()->finish();
  return 0;
}
