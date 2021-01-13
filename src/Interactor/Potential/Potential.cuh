/*Raul P. Pelaez 2017-2021. Implemented Potentials

  In this file there are functors and objects describing different potentials.
  A single Potential might be written to handle a variety of similar potentials.
  For example, all radial potentials need only the distance between particles,
    so RadialPotential is defined, and the particular potential is a functor passed to it (See LJFunctor).

See RadialPotential on how to implement a RadialPotential.
For more information on how to code a new Potential see examples/customPotentials.cu

 */
#ifndef POTENTIAL_CUH
#define POTENTIAL_CUH
#include"utils/Box.cuh"
#include"PotentialBase.cuh"
#include"RadialPotential.cuh"
namespace uammd{
  namespace Potential{
    //LJFunctor is a radial potential that can be passed to RadialPotential to be used as a Potential in
    //a module (i.e PairForces, NBodyForces...). Encodes the Lennard Jonnes potential
    //RadialPotential expects a functor with the rules of this one:
    //   -InputPairParameters, a type with the necessary parameters to differentiate between type pairs
    //   -PairParameters, a type with type pair parameters that the GPU computation will use (can be an alias of InputPairParameters).
    //   -A force*, energy and virial (not yet) functions taking a squared distance and a PairParameters
    //   -A processPairParameters function that transforms between InputPairParameters and PairParameters
    //* Notice that the force function in a RadialPotential must, in fact, return the modulus of the force divided by the distance, |f|/r.
    struct LJFunctor{
      struct InputPairParameters{
	real cutOff, sigma, epsilon;
	bool shift = false; //Shift the potential so lj(rc) = 0?
      };

      struct PairParameters{
	real cutOff2;
	real sigma2, epsilonDivSigma2;
	real shift = 0.0; // Contains lj_force(rc)
      };

      static inline __device__ real force(real r2, PairParameters params){
	if(r2 >= params.cutOff2) return 0;
	const real invr2 = params.sigma2/r2;
	const real invr6 = invr2*invr2*invr2;
	real fmod = params.epsilonDivSigma2*(real(-48.0)*invr6 + real(24.0))*invr6*invr2;
	fmod += params.shift?(params.shift*rsqrt(r2)):real(0.0);
	return fmod;
      }

      static inline __device__ real energy(real r2, PairParameters params){
	if(r2 >= params.cutOff2) return 0;
	real invr2 = params.sigma2/r2;
	real invr6 = invr2*invr2*invr2;
	real E = params.epsilonDivSigma2*params.sigma2*real(4.0)*invr6*(invr6-real(1.0));
	if(params.shift != real(0.0)){
	  //With shift, u(r) = lj(r)-lj(rc)  -(r-rc)·(dlj(r)/dr|_rc)
	  real rc = sqrt(params.cutOff2);
	  real invrc2 = real(params.sigma2)/(params.cutOff2);
	  real invrc6 = invrc2*invrc2*invrc2;
	  E += -(sqrt(r2)-rc)*params.shift - real(4.0)*params.epsilonDivSigma2*params.sigma2*invrc6*(invrc6-real(1.0));
	}
	return E;
      }

      static inline __host__ PairParameters processPairParameters(InputPairParameters in_par){
	PairParameters params;
	params.cutOff2 = in_par.cutOff*in_par.cutOff;
	params.sigma2 = in_par.sigma*in_par.sigma;
	params.epsilonDivSigma2 = in_par.epsilon/params.sigma2;
	if(in_par.shift){
	  real invCutOff2 = params.sigma2/params.cutOff2;
	  real invrc6 = invCutOff2*invCutOff2*invCutOff2;
	  real invrc7 = invrc6*sqrtf(invCutOff2);
	  real invrc13 = invrc7*invrc6;
	  params.shift = params.epsilonDivSigma2*(real(48.0)*invrc13 - real(24.0)*invrc7);
	}
	else params.shift = real(0.0);
	return params;
      }

    };

    using LJ = Radial<LJFunctor>;
  }
}
#endif
