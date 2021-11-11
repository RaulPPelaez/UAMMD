/* Raul P. Pelaez 2021
   Lets jump to the next basic UAMMD component: ParticleData
   UAMMD uses this class to handle particle information, such as positions, forces, charges,... 
   Besides serving as a communication element to share particles between modules, ParticleData allows to access particles from 
   CPU or GPU transparently.

   In this tutorial we will see hoy to create a set of particles and modify them.

   ParticleData provides additional functionality, we will cover some of it in next examples, but you can check ParticleData's wiki page for a more in depth list.

 */

//uammd.cuh is the basic uammd include containing, among other things, the System and ParticleData structs.
#include<uammd.cuh>
#include<random>
#include<thrust/random.h>
using namespace uammd;

int main(int argc, char* argv[]){
  //Initialize System
  auto sys = std::make_shared<System>(argc, argv);
  //Lets create 1e5 particles:
  const int numberParticles = 1e5;
  auto pd = std::make_shared<ParticleData>(sys, numberParticles);
  //From now on we can check the number of particles with   pd->getNumParticles();
  //ParticleData exposes a collection of functions to request the different properties of the particles.
  //UAMMD is a GPU code, so there is the issue of working with two separated memory spaces. ParticleData handles it for you.
  //You have to help it though by informing it of your usage intentions when a property is requested.
  //In particular ParticleData needs to know from where you are going to access the memory (the device, CPU or GPU) and if you are going to modify it or just read it.
  //When using UAMMD we will typically start by initializing the positions, so lets do that:
  {
    //We will request ParticleData for the positions, with the intention of writing to them from the CPU:
    //It is important to be aware that as long as the positions handle above exists it holds the sole ownership of the property
    //If another part of the code tries to request positions while this handle is alive an error will be thrown
    // because there is no guarantee that the positions are not changing at that moment.
    //In order to avoid this, ensure that the handle goes out of scope as soon as you are done with it.
    //Never store these handles. Simply request them again.
    auto positions = pd->getPos(access::location::cpu, access::mode::write);
    //Another way to get the number of particles is: positions.size();
    //The object returned by ParticleData works as a container, similar to std::vector or thrust::device_vector.
    //UAMMD stores the positions and forces as real4, with the fourth element of position being available to use for whatever, for example "particle type".
    //The fourth element in the force is unused and you should simply ignore it.
    //Lets fill the positions with random numbers:
    //We will use the C++ standard random generator, seeding it with the System rng
    std::mt19937 gen(sys->rng().next());
    //An uniform distribution between -0.5 and 0.5
    std::uniform_real_distribution<real> dist(-0.5, 0.5);
    auto rng = [&](){return dist(gen);};
    real L = 64; //Lets randomly set positions inside a cubic box of size L
    std::generate(positions.begin(), positions.end(), [&](){ return make_real4(rng(), rng(), rng(), 0)*L;});
    //If we had requested positions using access::location::gpu instead of cpu, the above std::generate line will have resulted in
    // a segfault, because positions would have had pointed to a memory chunk in GPU memory space.
    //Alternatively we could have written this using a loop:
    //for(auto &p: positions)  p = make_real4(rng(), rng(), rng(), 0);
    //It is however a good idea to work with std algorithms whenever possible, since it will make our code run in the GPU with simple changes.
  }
  {
    //Lets see how to do the same operation in the GPU:
    auto positions = pd->getPos(access::location::gpu, access::mode::write);
    auto seed = sys->rng().next();
    real L = 64; //Lets randomly set positions inside a cubic box of size L
    auto it = thrust::make_counting_iterator<int>(0);
    thrust::transform(thrust::cuda::par,
		      it, it + numberParticles,
		      positions.begin(),
		      [=]__device__(int i){
			thrust::default_random_engine rng(seed);
			//Take into account that this is a device lambda function, we are now in a massively parallel environment
			//Luckily, thrust has parallel random generator capabilities
			rng.discard(3*i);
			//UAMMD also exposes a GPU friendly rng called Saru, but we will leave that for an advance tutorial
			//An uniform distribution between -0.5 and 0.5
			thrust::uniform_real_distribution<real> dist(-0.5, 0.5);
			return make_real4(dist(rng), dist(rng), dist(rng), 0)*L;
		      });
  }
  //Try to remove the brackets that ensure the position handles to go out of scope (and change the variable names so it compiles).
  //An error will arise if you try to run it because the second request is illegal (given that another handle has ownershp at the time).
  //There is an exception to this rule: As many handles as needed can be used at the same time if the intention is access::mode::read.
  //This is why it is important to call pd->get* with the right access flags.
  //ParticleData maitains CPU and GPU copies of the properties and uses the access mode to detect if one has to be updated (requiring a CPU-GPU memory copy), so do not lie to ParticleData or expect undefined behavior.  
  
  //Similar to how we got a handle to the positions, there are several other properties in ParticleData:
  //Get particle velocities to read them from the cpu:
  auto velocities = pd->getVel(access::cpu, access::read);
  //You will notice the above call issues a warning log event if you execute it.
  //Requesting a property for the first time with the intention of reading makes little sense, since it will be filled with garbage.
  //Since we have not created any kind of UAMMD module that sets the v5elocities the above call is the first.

  //Another example: get the particle masses with the intention of writing to them from the gpu:
  auto mass = pd->getMass(access::gpu, access::write);
  //You can check the full list of available properties in ParticleData.cuh
  //The same thing that happend with vel happens now with force and energy, both containers will be filled with garbage because
  //There is no module filling them for us. At this point, they are merely containers with funny names.
  auto forces = pd->getForce(access::gpu, access::read);
  auto energies = pd->getEnergy(access::cpu, access::read);
  //We will see how to instruct UAMMD to fill these two with meaningful information, and also to evolve positions according to some
  // dynamics in following tutorials.
  //Destroy the UAMMD environment and exit
  sys->finish();
  return 0;
}
