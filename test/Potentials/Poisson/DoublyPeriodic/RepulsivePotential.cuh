#include "uammd.cuh"
#include"Interactor/Potential/Potential.cuh"

using namespace uammd;

struct RepulsivePotentialFunctor{
  struct InputPairParameters{
    real cutOff;
    real sigma, U0;
    real r_m;
    int p;
  };

  struct __align__(16) PairParameters{
    real cutOff2;
    real sigma, U0;
    real r_m;
    int p;
  };

  static __device__ __host__ real force(real r2, PairParameters params){
    if(r2 >= params.cutOff2 or r2 == 0) return 0;
    const real r = sqrtf(r2);
    const int p = params.p;
    const real sigma = params.sigma;
    const real U0 = params.U0;
    const real r_m = params.r_m;
    real rr = r;
    if(r<r_m){
      rr = r_m;
    }
    real sdivr = sigma/rr;
    real sdivrp=pow(sdivr,p);
    real fmod = real(-4.0)*U0*p*sdivrp*(real(2.0)*sdivrp - real(1.0))/rr;
    return fmod/r;
  }

  static __device__ real energy(real r2, PairParameters params){
    return 0;
  }

  static PairParameters processPairParameters(InputPairParameters in_par){
    PairParameters params;
    params.cutOff2 = in_par.cutOff*in_par.cutOff;
    params.sigma = in_par.sigma;
    params.p = in_par.p;
    params.U0 = in_par.U0;
    params.r_m = in_par.r_m;
    return params;
  }

};

using RepulsivePotential = Potential::Radial<RepulsivePotentialFunctor>;
