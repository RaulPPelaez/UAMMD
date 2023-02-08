/* Raul P. Pelaez 2021
   Moving particles with Integrators.
   In this tutorial we will learn about another basic UAMMD module, Integrator.
   Integrator is a small interface that allows to encode the concept of evolving particles due to some dynamics.
   We will see how to create and use a Brownian Dynamics (BD) Integrator which will allow us to simulate ideal (non interacting) particles.
   After this tutorial, you will have almost every tool you need to construct simulations, the only thing left is how to add interactions. Which we will cover in the next tutorial.

 */

#include <iterator>
#include<uammd.cuh>
#include"Integrator/BrownianDynamics.cuh" //Each Integrator has a particular include
#include<random>

using namespace uammd;

//Now that we are closing in a full blown particle simulation, lets take the chance to encapsulate in functions what we have learned in previous tutorials.
//Consider this a suggestion based on how I usually organize these UAMMD simulation codes.

// Lets group here a few parameters that our example is going to use. For the
// time being, lets simply hardcode some values
// Later, we will see how to read these parameters from a file.
struct Parameters{
  int numberParticles = 1e5;
  real boxSize = 64;
  //We need the following parameters for a Brownian dynamics Integrator:
  //Since the diffusion coefficient of a spherical particle in BD is D = T/(6*pi*viscosity*hydrodynamicRadius)
  // lets choose some parameters such that D=1
  real viscosity = 1.0/(6*M_PI);
  real hydrodynamicRadius = 1.0;
  real temperature = 1.0;
  real dt = 0.1; //Time integration step
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

//This function places particle positions randomly inside a cubic box of size L
void randomlyPlaceParticles(UAMMD sim){
  auto positions = sim.pd->getPos(access::location::cpu, access::mode::write);
  std::mt19937 gen(sim.pd->getSystem()->rng().next());
  std::uniform_real_distribution<real> dist(-0.5, 0.5);
  auto rng = [&](){return dist(gen);};
  std::generate(positions.begin(), positions.end(), [&](){ return make_real4(rng(), rng(), rng(), 0)*sim.par.boxSize;});
}

//This function constructs and returns a Brownian Dynamics integrator.
std::shared_ptr<Integrator> createBrownianDynamicsIntegrator(UAMMD sim){
  //Most Integrators in UAMMD are created in a very similar manner
  //First we choose the Integrator, in this case we are going to use the most basic Brownian Dynamics Integrator, Euler Maruyama
  using BD = BD::EulerMaruyama;
  //You can see that there are several other integration algorithms in the BD namespace, if you want to know more about them
  // see the wiki page for Brownian Dynamics.
  //Integrators always have a Parameters type inside of them that their constructor needs:
  BD::Parameters par;
  //Lets simply copy the parameters we hardcoded at the beginning
  par.dt = sim.par.dt;
  par.temperature = sim.par.temperature;
  par.hydrodynamicRadius = sim.par.hydrodynamicRadius;
  par.viscosity = sim.par.viscosity;
  //You should check the relevant doc page for information on how to construct each integrator, but 99% of the time you will see
  //The arguments are the same as for this one:
  auto bd = std::make_shared<BD>(sim.pd, par);
  //I like to store Integrators in shared pointers to easily pass them around.
  //You might have noticed that the we are returning a pointer to "Integrator" instead of BD
  //This is ok because all integrators in UAMMD inherit from the base class Integrator. Meaning that they can pass as one
  //This means that you can store any integrator in a variable with type std::shared_ptr<Integrator>.
  return bd;
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

//This function fills a vector with the particle positions ordered by id (name) and returns it
auto vector_from_pd_positions(UAMMD sim){
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  auto pos_by_id = thrust::make_permutation_iterator(pos.begin(), id2index);
  return std::vector<real4>(pos_by_id, pos_by_id + pos.size());
}

int main(int argc, char* argv[]){
  auto sim = initializeUAMMD(argc, argv);
  randomlyPlaceParticles(sim);
  //Now lets create an Integrator, it has the ability to take the current state of the particles and take it to the next time step.
  auto bd = createBrownianDynamicsIntegrator(sim);
  //The main function in the Integrator interface (and the only one we are going to use in this tutorial) is Integrator::forwardTime.
  //This function takes the simulation state at the current time, t0, (usually encoded in the relevant particle properties) and takes it to the next, t0+dt
  //In the case of a Brownian dynamics simulation of ideal particles, this means moving the particles randomly according to a certain diffusion coeficient (which we set to 1). Every other particle property (such as velocities) is irrelevant and thus left untouched.
  //First, lets print the first 10 particles
  printFirst10Particles(sim);
  //Now lets forward time:
  bd->forwardTime();
  //And print the positions again:
  printFirst10Particles(sim);
  //You can see particles moved a bit. Thats it, a Brownian Dynamics simulation of ideal particles.
  //Now you can call forwardTime as many times as needed.
  //Now that we are here, lets also check that the integration is correct.
  //Given that no forces are affecting particles their movement should come only from thermal noise, AKA diffusion.
  //We set the diffusion coefficient to 1 and dt is 0.1, so the displacement of each particle after a step (independently on each direction) should have zero mean and standard deviation D*dt=0.1, lets check it
  //Lets store particle positions:
  auto positions_previous = vector_from_pd_positions(sim);
  //take the positions to the next step
  bd->forwardTime();
  //And store the positions again:
  auto positions_after = vector_from_pd_positions(sim);
  //Create a vector with the displacements
  std::vector<real4> displacements;
  std::transform(positions_after.begin(), positions_after.end(),
		 positions_previous.begin(),
		 std::back_inserter(displacements),
		 std::minus<real4>());
  //The values in this vector should have zero mean:
  //In fact, since we are basically summing gaussian random numbers of mean 0 and std sqrt(2*dt) the mean should be on the order of
  // 1/sqrt(N=1e5) = 0.003
  real3 mean = make_real3(std::accumulate(displacements.begin(), displacements.end(), real4()))/displacements.size();
  std::cout<<"Mean displacement in each direction: "<<mean<<" expected something of the order of: "<<1/sqrt(sim.par.numberParticles)<<std::endl;
  //Lets now check the standard deviation:
  real3 stddev = sqrt(thrust::transform_reduce(displacements.begin(), displacements.end(),
					       [=](auto x){auto d=make_real3(x)-mean; return d*d;},
					       real3(),
					       std::plus<real3>())/displacements.size());
  //The standard deviation should be sqrt(2*D*dt) = sqrt(2*0.1)
  //Lets print the diffusion coefficient, which should be 1 -> D = stddev/sqrt(2*0.1)
  std::cout<<"Measured diffusion coefficient in each direction (should be 1): "<<stddev/sqrt(2*0.1)<<std::endl;
  //You might think the algorithmic way in which we wrote this is unnecesary here, but with minimal changes
  // we could have performed all post processing operations in the GPU, so this is good training.
  //Had we used C-style loops for this, porting it to the GPU would result in quite different (much more complex) code.

  //Integrators provide a couple more functions, you can check them in the relevant wiki page if you are impatient, but we will soon cover them.

  //Destroy the UAMMD environment and exit
  sim.pd->getSystem()->finish();
  return 0;
}
