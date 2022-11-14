/* Raul P. Pelaez 2022. Your first Interactor.

   We learned how to write a new Integrator in the previous example. Now, we
   will see how to write a new interaction using Interactor.

   If we follow the rules of Interactor, we will be able to add it to already
   existing Integrators, using it as any of the already existing ones.

   Like Integrator, Interactor is a small interface, requiring us to overload
   just a couple of functions. Lets see how to do that by creating a simple one.
 */

#include <uammd.cuh>
#include <Integrator/BDHI/BDHI_FCM.cuh> //We will add out Interactor to a hydrodynamic Integrator.
#include<random>
using namespace uammd;

// We start by defining a class that inherits from Interactor.
//This forces us to overload the required methods that any Integrator must provide.
class MyFirstInteractor: public Interactor{

  public:
  MyFirstInteractor(std::shared_ptr<ParticleData> pd):
    //The base class (Interactor) constructor must then be invoked with that pd and a name for our module
    Interactor(pd, "MyFirstInteractor"){
    //Any initialization for our module also happens here
    this->someParameter = 1.0;
  }

  //The most important function in Interactor is "sum".
  //It can be called at any time and it is expected to fill the forces, energies and/or virials of the particles based on their current state.
  //"sum" must always sum to the existing properties, never overwritting them.
  //Our simple Interactor will just add some force and energy to the first particle in the X direction. Also, we will make this interaction turn off after a certain arbitrary time.
  // As an exercise you can let your imagination fly here and implement any arbitrarily complex interaction.
  //See for instance the file Interactor/SpectralEwaldPoisson.cuh, which contains a complex electrostatics interactor.
  void sum(Computables comp, cudaStream_t st = 0) override{
    //Like in Integrator, being an Interactor makes some members available to us:
    //pd: An instance of ParticleData
    //pg: An instance of ParticleGroup (something we will discuss in future tutorials).
    //sys: An instance of System
    sys->log<System::DEBUG2>("[MyFirstInteractor] Computing interaction");
    //The argument "comp" contains a series of booleans, if one of them is true this call is supposed to sum to
    // the relevant ParticleData array. Ideally, if this function is called with some requirement that the Interactor
    // is not able to satisfy, the call should throw an error. For instance, suppose our Interactor is not capable of computing
    // the virial:
    const real turnOffTime = 10;
    if(this->currentSimulationTime < turnOffTime){
      if(comp.virial){
	System::log<System::EXCEPTION>("[MyFirstInteractor] I cannot compute the virial");
	throw std::runtime_error("Invalid Interactor requirement");
      }
      if(comp.force){
	//Sum forces to each particle
	auto forces = pd->getForce(access::cpu, access::write);
	forces[0].x += 1;
      }
      if(comp.energy){
	//Sum energies to each particle
	auto energies = pd->getEnergy(access::cpu, access::write);
	energies[0] += 1;
      }
    }
  }

  //We already discussed the ParameterUpdatable interface in the previous tutorial. We saw there that every Interactor is ParameterUpdatable by default.
  //In principle, the Interactor is unaware of time, but let us imagine that our interaction changes with time. We can use the ParameterUpdatable mechanism to make our Interactor aware of time by overriding one function from it. In this case updateSimulationTime:
  void updateSimulationTime(real newTime) override{
    //Everytime the Integrator::forwardTime is called the Integrator will update the simulation time of all its interactors.
    //This will include this interactor.
    this->currentSimulationTime = newTime;
  }

private:
    //Some parameters, note that typically you would take some of these in the constructor, or read them from a file as we saw in previous examples.
  real someParameter;
  real currentSimulationTime = 0;
};

// Creates a Force Couplin Method integrator.
// This is a brownian hydrodynamics Integration scheme, which includes
// hydrodynamic interactions.
//When pulling one particle, as our Interactor is doing, we will see how the surrounding ones are dragged with it.
auto createFCMIntegrator(std::shared_ptr<ParticleData> pd){
  using Scheme = BDHI::FCMIntegrator;
  //Some arbitrary parameters
  Scheme::Parameters par;
  par.box = Box({32,32,32});
  par.temperature = 1.0;
  par.viscosity = 1.0;
  par.dt = 0.1;
  par.hydrodynamicRadius = 1.0;
  auto bdhi = std::make_shared<Scheme>(pd, par);
  return bdhi;
}


//This function places particle positions randomly inside a cubic box of size L
void randomlyPlaceParticles(std::shared_ptr<ParticleData> pd){
  auto positions = pd->getPos(access::location::cpu, access::mode::write);
  std::mt19937 gen(pd->getSystem()->rng().next());
  std::uniform_real_distribution<real> dist(-0.5, 0.5);
  auto rng = [&](){return dist(gen);};
  real3 lbox = {32,32,32}; //An arbitrary box size
  std::generate(positions.begin(), positions.end(), [&](){return make_real4(make_real3(rng(), rng(), rng())*lbox, 0);});
}


int main(){
  int numberParticles = 1e5;
  auto pd = std::make_shared<ParticleData>(numberParticles);
  randomlyPlaceParticles(pd);
  auto fcm = createFCMIntegrator(pd);
  //Our new Interactor is just like every other, so we can add it to an Integrator
  auto myInteractor = std::make_shared<MyFirstInteractor>(pd);
  fcm->addInteractor(myInteractor);

  //Now fcm will take into account our interaction.
  //As an exercise, you can evolve the simulation for some time and visualize the results.
  fcm->forwardTime();
  return 0;
}
