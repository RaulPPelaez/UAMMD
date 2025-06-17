#include "Interactor/Potential/Potential.cuh"
#include "uammd.cuh"

using namespace uammd;

struct SoftPotentialFunctor {
  struct InputPairParameters {
    real cutOff;
    real b, d, U0;
  };

  struct __align__(16) PairParameters {
    real cutOff2;
    real b, d, U0;
  };

  __device__ real force(real r2, PairParameters params) {
    if (r2 >= params.cutOff2 or r2 == 0)
      return 0;
    const real r = sqrtf(r2);
    const real d = params.d;
    const real b = params.b;
    const real U0 = params.U0;
    if (r < d) {
      return -U0 / (b * r);
    } else {
      return -U0 * exp((d - r) / b) / (b * r);
    }
  }

  __device__ real energy(real r2, PairParameters params) {
    if (r2 >= params.cutOff2 or r2 == 0)
      return 0;
    const real r = sqrtf(r2);
    const real d = params.d;
    const real b = params.b;
    const real U0 = params.U0;
    const real rdb = (d - r) / b;
    if (r < d) {
      return (U0 + U0 * rdb);
    } else {
      return U0 * exp(rdb);
    }
  }

  static PairParameters processPairParameters(InputPairParameters in_par) {
    PairParameters params;
    params.cutOff2 = in_par.cutOff * in_par.cutOff;
    params.b = in_par.b;
    params.d = in_par.d;
    params.U0 = in_par.U0;
    return params;
  }
};

using SoftPotential = Potential::Radial<SoftPotentialFunctor>;
