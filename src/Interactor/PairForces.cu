/*Raul P. Pelaez 2017. PairForces definition.

  PairForces Module is an interactor that computes short range forces.
   Computes the interaction between neighbour particles (pairs of particles closer tan rcut).
    
  For that, it uses a NeighbourList and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces   for more info.

See misc/Potential.cuh and https://github.com/RaulPPelaez/UAMMD/wiki/Potential for more info on potentials and how to implement them.
*/

#include<cub/cub.cuh>

template<class NL, class Potential>
PairForces<NL, Potential>::PairForces():
  PairForces<NL, Potential>(gcnf.rcut, gcnf.L, gcnf.N){
}
template<class NL, class Potential>
PairForces<NL, Potential>::PairForces(real rcut):
  PairForces<NL, Potential>(rcut, gcnf.L, gcnf.N){
}
template<class NL, class Potential>
PairForces<NL, Potential>::PairForces(real rcut, real3 L, int N):
  Interactor(128, L, N, 0),
  nl(rcut, L, N),
  pot(rcut),
  energy(N), cubTempStorage(nullptr){
  name = "PairForces";
}


namespace PairForces_ns{
  /*Force transverser. Sums, for each particle, the force due to all its neighbours.

    A Transverser is given to a NeighbourList. It contains functions to compute the interaction between a pair of particles (compute), and to accumulate the results for each neighbour (operator real4 += real4).

    For a more general transverser see PairForcesDPD.cu

    See more info in https://github.com/RaulPPelaez/UAMMD/wiki/Neighbour-List

    forceTransverser needs a force functor that returns f/r given the distance between i and j and its colors.
  */
  template<class forceFunctor>
  class forceTransverser{
  public:
    /*I need the device pointer to force, the invrc2 and the force texture*/
    forceTransverser(real4 *newForce,
		     forceFunctor force,
		     BoxUtils box):
      newForce(newForce), force(force), box(box){};
    /*Compute the force between two positions. 
      For a particle i, this function will be called for all its neighbours j*/
    inline  __device__ real3 compute(const real4 &R1,const real4 &R2){
      real3 r12 = make_real3(R2)-make_real3(R1);
      box.apply_pbc(r12);
      /*Squared distance*/
      const real r2 = dot(r12,r12);
      /*Get the force from the functor*/
      const real3 f =  force(r2, R1.w, R2.w)*r12;
      return  f;
    }
    /*For each particle i, this function will be called for all its neighbours j with the result of compute_ij and total, with the value of the last time accumulate was called, starting with zero()*/
    inline __device__ void accumulate(real3& total, const real3& current){total += current;}
    /*This function will be called for each particle i, once when all neighbours have been transversed, with the particle index and the value total had the last time accumulate was called*/
    /*Update the force acting on particle pi, pi is in the normal order*/
    inline __device__ void set(uint pi, const real3 &totalForce){     
      newForce[pi] += make_real4(totalForce.x, totalForce.y, totalForce.z, real(0.0));
    }
    /*Starting value, can also be used to initialize in-kernel parameters, as it is called at the start*/
    inline __device__ real3 zero(){return make_real3(real(0.0));}


  private:
    /*All this will be available in device functions*/
    real4* newForce; /*global force array*/
    forceFunctor force;
    BoxUtils box; /*Box utlities (apply_pbc)*/
  };
}
template<class NL, class Potential>
void PairForces<NL, Potential>::sumForce(){
  /*Create a box*/
  BoxUtils box(L);
  
  /*Update neighbour list*/
  nl.makeNeighbourList();
  

  if(gcnf.color2name.size()>1){
    /*Get the functor that computes the force from the pot object*/
    auto forceFunctor = pot.template getForceFunctor<true>();
    /*Create an instance of the transverser*/
    auto ft = PairForces_ns::forceTransverser<decltype(forceFunctor)>(force.d_m,
								      forceFunctor,
								      box);
    /*transverse the list with it*/
    nl.transverse(ft);
  }
  else{
    /*Get the functor that computes the force from the pot object*/
    auto forceFunctor =  pot.template getForceFunctor<false>();
    /*Create an instance of the transverser*/
    auto ft = PairForces_ns::forceTransverser<decltype(forceFunctor)>(force.d_m,
								      forceFunctor,
								      box);
    /*transverse the list with it*/
    nl.transverse(ft);    
  }
  cudaDeviceSynchronize();

}

namespace PairForces_ns{
  /*Very similar to forceTransverser, but fills an array with reals, containing the total energy acting on each particle due to each neighbours
    energyTransverser needs an energy functor that returns E given the distance between i and j and its colors.
  */
  template<class energyFunctor>
  class energyTransverser{
  public:
    /*I need the device pointer to force, the invrc2 and the force texture*/
    energyTransverser(real *d_energy,
		      energyFunctor energy,
		      BoxUtils box):
      d_energy(d_energy), energy(energy), box(box){};
    /*Compute the force between two positions. 
      For a particle i, this function will be called for all its neighbours j*/
    inline  __device__ real compute(const real4 &R1,const real4 &R2){
      real3 r12 = make_real3(R2)-make_real3(R1);
      box.apply_pbc(r12);
      /*Squared distance*/
      const real r2 = dot(r12,r12);
      /*Get the force from the functor*/
      return energy(r2, R1.w, R2.w);
    }
    /*For each particle i, this function will be called for all its neighbours j with the result of compute_ij and total, with the value of the last time accumulate was called, starting with zero()*/
    inline __device__ void accumulate(real& total, const real& current){total += current;}
    /*This function will be called for each particle i, once when all neighbours have been transversed, with the particle index and the value total had the last time accumulate was called*/
    /*Update the force acting on particle pi, pi is in the normal order*/
    inline __device__ void set(uint pi, const real &totalEnergy){d_energy[pi] = totalEnergy;}
    /*Starting value, can also be used to initialize in-kernel parameters, as it is called at the start*/
    inline __device__ real zero(){return real(0.0);}


  private:
    /*All this will be available in device functions*/
    real *d_energy; /*global energy array*/
    energyFunctor energy;
    BoxUtils box; /*Box utlities (apply_pbc)*/
  };

}
template<class NL, class Potential>
real PairForces<NL, Potential>::sumEnergy(){
  /*Create a box*/
  BoxUtils box(L);
  
  /*Update neighbour list*/
  nl.makeNeighbourList();
  

  if(gcnf.color2name.size()>1){
    /*Get the functor that computes the energy from the pot object*/
    auto energyFunctor = pot.template getEnergyFunctor<true>();
    /*Create an instance of the transverser*/
    auto ft = PairForces_ns::energyTransverser<decltype(energyFunctor)>(energy.d_m,
									energyFunctor,
									box);
    /*transverse the list with it*/
    nl.transverse(ft);
  }
  else{ /*Same for the case when there is only one particle type*/
    auto energyFunctor =  pot.template getEnergyFunctor<false>();
    auto ft = PairForces_ns::energyTransverser<decltype(energyFunctor)>(energy.d_m,
									energyFunctor,
									box);
    /*transverse the list with it*/
    nl.transverse(ft);    
  }


  /*Reduce the energy array to get the total energy*/
  real *d_E;
  gpuErrchk(cudaMalloc(&d_E, sizeof(real)));
  if(!cubTempStorage){
    cub::DeviceReduce::Sum(cubTempStorage, cubTempStorageBytes, energy.d_m, d_E, N);
    gpuErrchk(cudaMalloc(&cubTempStorage, cubTempStorageBytes));
  }
  cub::DeviceReduce::Sum(cubTempStorage, cubTempStorageBytes, energy.d_m, d_E, N);
  
  real E = 0.0;
  gpuErrchk(cudaMemcpy(&E, d_E, sizeof(real), cudaMemcpyDeviceToHost));
  cudaFree(d_E);
  /*Return energy per particle*/
  return E/(real)N;
}
