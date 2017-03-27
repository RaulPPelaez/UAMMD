/*Raul P. Pelaez 2017. PairForces definition.

  PairForces Module is an interactor that computes short range forces.
    Computes the interaction between neighbour particles (pairs of particles closer tan rcut).
    
  For that, it uses a NeighbourList and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces   for more info.
*/
#include"PairForces.cuh"
//#include"GPUutils.cuh" /*It is already included*/


template<class NL>
PairForces<NL>::PairForces(std::function<real(real,real)> Ffoo,
		       std::function<real(real,real)> Efoo):
  PairForces<NL>(gcnf.rcut, gcnf.L, gcnf.N, Ffoo, Efoo){}
template<class NL>
PairForces<NL>::PairForces(real rcut, real3 L, int N,
		       std::function<real(real,real)> Ffoo,
		       std::function<real(real,real)> Efoo):
  Interactor(), nl(rcut,L, N){
  pot = Potential(Ffoo, Efoo, 4096*rcut/real(2.5)+1, rcut, 1);
  name = "PairForces";
}



/*Force transverser. Sums, for each particle, the force due to all its neighbours.

  A Transverser is given to a NeighbourList. It contains functions to compute the interaction between a pair of particles (compute), and to accumulate the results for each neighbour (operator real4 += real4).

  For a more general transverser see PairForcesDPD.cu

  See more info in https://github.com/RaulPPelaez/UAMMD/wiki/Neighbour-List
 */
template<bool many_types> /*many_types controls if more than one interaction type exists*/
class forceTransverser{
public:
  /*I need the device pointer to force, the invrc2 and the force texture*/
  forceTransverser(real4 *newForce, float invrc2, cudaTextureObject_t texForce):
    newForce(newForce), invrc2(invrc2), texForce(texForce){};
  /*Compute the force between two positions. 
    For a particle i, this function will be called for all its neighbours j*/
  inline  __device__ real3 compute(const real4 &R1,const real4 &R2){
    real3 r12 = make_real3(R2)-make_real3(R1);
    apply_pbc(r12);
    /*Squared distance*/
    /*Squared distance between 0 and 1*/
    const real r2 = dot(r12,r12);
    real r2c = r2*invrc2;
    /*Reescale for current type pair*/
    real epsilon = real(1.0); 
    /*Get the force from the texture*/
    const real3 f = (epsilon*(real) tex1D<float>(texForce, r2c))*r12;
    return  f;
  }
  /*For each particle i, this function will be called for all its neighbours j with the result of compute_ij and total, with the value of the last time accumulate was called, starting with zero()*/
  inline __device__ void accumulate(real3& total, const real3& current){
    total += current;
  }
  /*This function will be called for each particle i, once when all neighbours have been transversed, with the particle index and the value total had the last time accumulate was called*/
  /*Update the force acting on particle pi, pi is in the normal order*/
  inline __device__ void set(uint pi, const real3 &totalForce){
    newForce[pi] += make_real4(totalForce.x, totalForce.y, totalForce.z, real(0.0));
  }
  inline __device__ real3 zero(){
    return make_real3(real(0.0));
  }


private:
  /*All this will be available in device functions*/
  real4* newForce; /*global force array*/
  float invrc2;    /*Precompute this expensive number*/
  cudaTextureObject_t texForce; /*Texture containing the potential*/
};

template<class NL>
void PairForces<NL>::sumForce(){

  /*Update neighbour list*/
  nl.makeNeighbourList();

  /*Create an instance of the transverser*/
  auto ft = forceTransverser<false>(force.d_m,
				    1.0/(nl.getRcut()*nl.getRcut()),
				    pot.getForceTexture().tex);
  /*transverse the list with it*/
  nl.transverse(ft);
}


template class PairForces<CellList>;