/*Raul  P. Pelaez  2022. Computing  some  force (in  this case  triply
   periodic electrostatics)  with UAMMD to  use them in  some external
   code.

   In particular, we  write a function that has  input (positions) and
   output   (forces)  in   cpu  memory   and  computes   electrostatic
   forces.  This function  could  be called  from  another purely  CPU
   codebase.

   In this example  the parameters for the  interaction are hardcoded,
   but naturally  in a real  implementation you would need  to provide
   them somehow.
 */

#include"uammd.cuh"
#include"Interactor/SpectralEwaldPoisson.cuh"
using namespace uammd;

//This struct contains the basic uammd modules for convenience.
struct UAMMD{
  std::shared_ptr<ParticleData> pd;
  //std::vector<std::shared_ptr<Interactor>> interactors;
  std::shared_ptr<Interactor> interactor;
};

namespace helper{
  //Creates an instance of a Poisson  Interactor that can be added to an
  //integrator.   When  using   Electrostatics  remember  to  initialize
  //particle  charges   via  ParticleData::getCharges  in   addition  to
  //positions.
  auto createElectrostaticInteractor(std::shared_ptr<ParticleData> pd){
    Poisson::Parameters par;
    par.box = Box(make_real3(32,32,32));
    par.epsilon = 1;
    par.gw = 1.0;
    par.tolerance = 1e-8;
    auto poisson = std::make_shared<Poisson>(pd, par);
    return poisson;
  }

  //Sets particle forces to zero
  void resetForces(std::shared_ptr<ParticleData> pd){
    auto forces = pd->getForce(access::gpu, access::write);
    thrust::fill(forces.begin(), forces.end(), real4());
  }

}

//Initialized the necessary UAMMD structures
auto initializeUAMMD(int numberParticles){
  UAMMD sim;
  sim.pd = std::make_shared<ParticleData>(numberParticles);
  return sim;
}


// Returns a CPU  vector with the electrostatic forces  acting on each
// particle.  The   format  of  the   input  and  output   arrays  are
// [fx1,fy1,fz1,...,fxN,fyN,fzN]
std::vector<real> computeElectrostaticForces(UAMMD sim, std::vector<real> positions){
  //We initialize the interactor if not already done
  if(not sim.interactor)
    sim.interactor = helper::createElectrostaticInteractor(sim.pd);
  //We copy the input positions to UAMMD
  {
    auto pos = sim.pd->getPos(access::cpu, access::write);
    for(int i = 0; i< pos.size(); i++){
      pos[i].x = positions[3*i];
      pos[i].y = positions[3*i + 1];
      pos[i].z = positions[3*i + 2];
    }
    //We also need to set charges, which could be provided as a parameter also.
    auto charge = sim.pd->getCharge(access::gpu, access::write);
    //Lets hardcode them to 1 for this example
    thrust::fill(charge.begin(), charge.end(), 1);
  }
  //Lets ensure forces in UAMMD are set to zero.
  helper::resetForces(sim.pd);
  //Compute current electrostatic forces
  sim.interactor->sum({.force = true});
  //Download forces from UAMMD into the desired output format in the CPU
  auto forces = sim.pd->getForce(access::cpu, access::read);
  std::vector<real> cpu_forces(3*forces.size());
  for(int i = 0; i< forces.size(); i++){
    cpu_forces[3*i] = forces[i].x;
    cpu_forces[3*i+1] = forces[i].y;
    cpu_forces[3*i+2] = forces[i].z;
  }
  return cpu_forces;
}


int main(int argc, char *argv[]){
  //Lets say you have some code that holds some particle positions
  int numberParticles = 2;
  std::vector<real> positions(numberParticles, 0);
  //Two particles, one at (1,0,0) and the other at (0,0,0)
  positions[0] = 1;
  //Lets say you want to  compute the electrostatic force between them
  //using UAMMD You can first initialize uammd and store the necessary
  //modules in  a structure.  This  allows to keep a  persistent UAMMD
  //state,  which means  that  you  do not  have  to initialize  UAMMD
  //everytime you  want the forces.   Alternative, we could  have hide
  //the  UAMMD struct  instance inside  the computeElectrostaticForces
  //function as  a static  variable. While this  has its  downsides it
  //would allow us to expose just the one function.
  auto sim =  initializeUAMMD(numberParticles);
  //The function we wrote takes a  CPU position vector with the format
  //xyz for  each particle and  returns the electrostatic  forces with
  //the same format  also in the CPU. Note that  the computation takes
  //place  in the  GPU,  so  doing it  like  so  requires two  CPU-GPU
  //copies. However,  we are  assuming that  the resulting  forces are
  //required for a CPU code.
  auto electrostatic_forces = computeElectrostaticForces(sim, positions);
  std::cout<<"The electrostatic force between the particles is (located at (";
  std::cout<<positions[0]<<","<<positions[1]<<","<<positions[2]<<") and (";
  std::cout<<positions[3]<<","<<positions[4]<<","<<positions[5]<<") is:\n";
  std::cout<<"F01 = ("<<electrostatic_forces[3]<<","<<electrostatic_forces[4]<<","<<electrostatic_forces[5]<<")"<<std::endl;
  return 0;
}
