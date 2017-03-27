/*Raul P.Pelaez. 2017. External Forces Module.
  Computes a force acting on each particle due to an external potential.
  i.e harmonic confinement, gravity...
  
  Needs a functor transverser with an overload () operator as:

  struct externaltor{
    externaltor(real k): k(k){}
    inline __device__ real4 operator()(const real4 &pos, int i){
      return -k*pos;
    }
    real k;
  };

*/
#ifndef EXTERNALFORCES_CUH
#define EXTERNALFORCES_CUH
#include"Interactor.h"
#include"globals/defines.h"
#include"globals/globals.h"
#include"third_party/type_names.h"
template<class Transverser>
class ExternalForces: public Interactor{
public:

  ExternalForces(Transverser tr):Interactor(), tr(tr){
    name = "ExternalForces";
  }
  ~ExternalForces(){}
  
  void sumForce() override; /*implemented below*/   
  real sumEnergy() override{return 0;}
  real sumVirial() override{return 0;}

  void print_info(){
    std::cerr<<"\t Transversing with: "<<type_name<Transverser>()<<std::endl;
  }
  
private:
  Transverser tr;
};


namespace ExternalForces_ns{
  template<class Transverser>
  __global__ void computeGPUD(const real4 * __restrict__ pos,
			      real4 * __restrict__ force, Transverser tr, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;

    force[i] += tr(pos[i], i);
  }
}

template<class Transverser>
void ExternalForces<Transverser>::sumForce(){
  int Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int Nblocks  = (N+Nthreads-1)/Nthreads;
  //const size_t info_size =
  //      sizeof(std::result_of<decltype(&Transverser::getInfo)(Transverser, int)>::type);
    
  ExternalForces_ns::computeGPUD<Transverser><<<Nblocks, Nthreads>>>(pos, force, tr, N);
}





#endif