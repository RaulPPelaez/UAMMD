/* Raul P. Pelaez 2021

   This code contains a collection of functions that create and return an instance of the different UAMMD integrators.
   The code in itself does not do much, rather it is intended  to serve as a copy pastable collection of snippets.

 */

#include<uammd.cuh>

using namespace uammd;

//I like to place these basic UAMMD objects in a struct so it is easy to pass them around
struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  //Parameters par; //For this example parameters will be hardcoded
};

//Creates and returns a UAMMD struct with the basics that we have covered in previous tutorials
UAMMD initializeUAMMD(int argc, char *argv[], int numberParticles){
  UAMMD sim;
  //Initialize System and ParticleData
  sim.pd = std::make_shared<ParticleData>(numberParticles);
  return sim;
}


#include "Integrator/Hydro/ICM_Compressible.cuh"
//Inertial Coupling Method
auto createIntegratorICM(UAMMD sim){
  using ICM = Hydro::ICM_Compressible;
  ICM::Parameters par;
  par.dt = 0.1;
  real3 L = make_real3(32,32,32);
  par.box = Box(L);
  par.hydrodynamicRadius = 1;
  par.bulkViscosity = 1.0;
  par.speedOfSound = 1.0;
  par.shearViscosity = 1.0;
  par.temperature = 1;
  auto icm = std::make_shared<ICM>(sim.pd, par);
  return icm;
}

int main(int argc, char* argv[]){
  int N = 16384;
  {
    auto sim = initializeUAMMD(argc, argv, N);
    auto integrator = createIntegratorICM(sim);
    for(int i= 0; i<100; i++){
      integrator->forwardTime();
    }
  }
  return 0;
}
