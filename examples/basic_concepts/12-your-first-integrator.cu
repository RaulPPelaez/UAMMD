/* Raul P. Pelaez 2022. Your first Integrator.
   In previous examples we used Integrators provided by UAMMD.
   You might want to define how particles are updated yourself.
   In this tutorial, we will define our own Integrator to do just that.
   If we follow the rules our Integrator will fit in nicely with the rest of
   UAMMD, allowing us to take advantage of, for instance, Interactors.

   For more information about Integrator see its documentation page. In this
   tutorial we will omit some advanced functionality.
 */

#include <Interactor/SpectralEwaldPoisson.cuh> //We will add electrostatics later in this example
#include <random>
#include <uammd.cuh>
using namespace uammd;

// We start by defining a class that inherits from Integrator.
// This forces us to overload the required methods that any Integrator must
// provide.
class MyFirstIntegrator : public Integrator {

public:
  // Let us start with the constructor which, at the very least, must take in a
  // ParticleData instance. Usually, an Integrator will also require some
  // parameters in its constructor.
  MyFirstIntegrator(std::shared_ptr<ParticleData> pd)
      : // The base class (Integrator) constructor must then be invoked with
        // that pd and a name for our module
        Integrator(pd, "MyFirstIntegrator") {
    // Any initialization for our module also happens here
    this->someParameter = 1.0;
  }

  // The most important method we must override is forwardTime.
  // This function is in charge of updating "pd" to the next step in time
  void forwardTime() override {
    // Being an Integrator grants this class access to some useful members, in
    // particular:
    //  pd: The provided ParticleData instance
    //  pg: A ParticleGroup (something we will discuss in future tutorials).
    //  sys: The System instance assigned to this Integrator
    //  interactors: A vector with all the Interactors that have been added to
    //  this Integrator updatables: A vector with all the ParameterUpdatables
    //  (something we will discuss in future tutorials) that have been added to
    //  this Integrator
    // Lets use them all to forward the simulation.
    System::log<System::DEBUG2>("[MyFirstIntegrator] It is good practice to "
                                "include debug messages when relevant");
    // We will implement some simple Euler Brownian dynamics update rule in the
    // CPU with some hardcoded parameters
    // We will start by setting particle forces to 0:
    resetForces(); // Defined below
    // Then, we will invoke any Interactors that have been added to this
    // Integrator to compute the current forces:
    computeCurrentForces(); // Defined below
    // Finally, we will use these forces to forward the simulation.
    updatePositions(); // Defined below
    // There is another part of the update flow that we should discuss.
    // It is related with the "updatables" member and the ParameterUpdatale
    // UAMMD interface. You can read all about it in the documentation. Mainly,
    // it solves the problem of communicating parameters changes to anything
    // that wants to be aware of them. Note that Interactors are
    // ParameterUpdatable by default. So when an Interactor is added via
    // addInteractor it will be available twice: As an Interactor in
    // "interactors" and as a ParameterUpdatable in "updatables". One common use
    // for this is to communicate the new simulation time:
    this->steps++;
    for (auto u : updatables)
      u->updateSimulationTime(steps * dt);
    // There are a lot of update* functions (see ParameterUpdatable.cuh) to
    // signal changes on different parameters. For instance:
    //  for(auto u: updatables){
    //    u->updateTemperature(newTemp);
    //    u->updateTimeStep(dt);
    //    u->updateViscosity(newVis);
    //    ...
    //  }
    // Typically, you would use the first step to inform of any relevant common
    // parameters of your Integrator, such as the time step, temperature, etc,
    // that other modules might be interested in.
  }

private:
  // Some parameters, note that typically you would take some of these in the
  // constructor, or read them from a file as we saw in previous examples.
  real someParameter;
  real dt = 0.01; // The time step.
  int steps = 0;  // The number of steps thus far

  // This function sets the particles forces to zero.
  void resetForces() {
    auto force = pd->getForce(access::gpu, access::write);
    thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
  }

  // This function goes through every Interactor summing the current parrticle
  // forces
  void computeCurrentForces() {
    // Integrator makes the member "interactors" available to us, which contains
    // all Interactors that have been added to this
    //  Integrator. For instance, a LJ module, or some custom interaction.
    // Note that at this point, all particle forces are set to 0.
    // The following loop queries each Interactor to update pd->getForce (adding
    // to it) with its computed forces. For our simple Euler Integrator, we just
    // need the forces, which we can request like so
    for (auto i : interactors)
      i->sum({.force = true, .energy = false, .virial = false});
  }

  // This function updates the particle positions using the current forces with
  // a BD Euler update rule. Let us do the computations in the CPU, but as an
  // exercise, you could make this into a GPU code.
  //  Hint:
  //     Add __device__ to the lambda (between [=] and (real4... to make it into
  //     a GPU lambda and change std:: by thrust::. Also change access::cpu to
  //     access::gpu.
  void updatePositions() {
    const real mobility = 1.0; // In BD we have dX = M*F*dt
    // We will not include fluctuations for simplicity, but you can use Saru,
    // exposed by UAMMD, to generate random numbers in parallel. See the
    // documentation page for it. Maybe you can take this opportunity to learn
    // how to use Saru here by adding fluctuations to our Integrator.
    //  In that case, the equation changes to: dX = M*F*dt +
    //  sqrt(2*temperature*M*dt)*dW Being dW a Gaussian random number
    //  (uncorrelated for each particle/coordinate) with 0 mean and 1 standard
    //  deviation.
    auto pos = pd->getPos(access::cpu, access::readwrite);
    auto force = pd->getForce(access::cpu, access::read);
    std::transform(pos.begin(), pos.end(), force.begin(), pos.begin(),
                   [=](real4 pos_i, real4 force_i) {
                     real3 p = make_real3(pos_i);
                     real3 f = make_real3(force_i);
                     // UAMMD usually ignores pos.w, but we should leave it
                     // untouched
                     return make_real4(p + mobility * f * dt, pos_i.w);
                   });
  }
};

// Creates an electrostatics Interactor, a module called Poisson in UAMMD.
auto createElectrostaticsInteractor(std::shared_ptr<ParticleData> pd) {
  // Let us initialize the particle charges to 1
  auto charges = pd->getCharge(access::cpu, access::write);
  std::fill(charges.begin(), charges.end(), 1);
  // Some arbitrary parameters
  Poisson::Parameters par;
  par.box = Box({32, 32, 32});
  par.epsilon = 1; // Permittivity
  par.gw = 1;      // Width of the charges
  return std::make_shared<Poisson>(pd, par);
}

// This function places particle positions randomly inside a cubic box of size L
void randomlyPlaceParticles(std::shared_ptr<ParticleData> pd) {
  auto positions = pd->getPos(access::location::cpu, access::mode::write);
  std::mt19937 gen(pd->getSystem()->rng().next());
  std::uniform_real_distribution<real> dist(-0.5, 0.5);
  auto rng = [&]() { return dist(gen); };
  real3 lbox = {32, 32, 32}; // An arbitrary box size
  std::generate(positions.begin(), positions.end(), [&]() {
    return make_real4(make_real3(rng(), rng(), rng()) * lbox, 0);
  });
}

int main() {
  // Let us create an instance of our new integrator
  int numberParticles = 1e5;
  auto pd = std::make_shared<ParticleData>(numberParticles);
  randomlyPlaceParticles(pd);
  auto myIntegrator = std::make_shared<MyFirstIntegrator>(pd);
  // MyFirstIntegrator is just an Integrator, so we can add an Interactor to it:
  // For instance, lets add electrostatics
  auto elec = createElectrostaticsInteractor(pd);
  myIntegrator->addInteractor(elec);
  // Now we would call forwardTime as many times as required.
  myIntegrator->forwardTime();
  return 0;
}
