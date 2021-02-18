/*Raul P. Pelaez 2017-2021. Implemented Potentials

For more information on how to code a new Potential see examples/customPotentials.cu
See RadialPotential.cuh for an additional example.
 */
#ifndef POTENTIAL_CUH
#define POTENTIAL_CUH
#include"utils/Box.cuh"
#include"PotentialBase.cuh"
#include"RadialPotential.cuh"
namespace uammd{
  namespace Potential{
    //LJFunctor encodes the Lennard Jonnes potential
    //this class is meant to be used to specialize RadialPotential, which can then to be used as the Potential for PairForces
    //RadialPotential expects a functor with the rules of this one:
    //   -InputPairParameters, a type with the necessary parameters to differentiate between type pairs
    //   -PairParameters, a type with type pair parameters that the GPU computation will use (can be an alias of InputPairParameters).
    //   -A force and energy functions taking a squared distance and a PairParameters
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
	real shift = 0.0; // Contains energy(rc)
      };
      //Returns the modulus of the force divided by r
      static inline __device__ real force(real r2, PairParameters params){
	if(r2 >= params.cutOff2) return 0;
	const real invr2 = params.sigma2/r2;
	const real invr6 = invr2*invr2*invr2;
	real fmod = params.epsilonDivSigma2*(real(-48.0)*invr6 + real(24.0))*invr6*invr2;
	//fmod += params.shift?(params.shift*rsqrt(r2)):real(0.0);
	return fmod;
      }
      //returns the energy per particle
      static inline __device__ real energy(real r2, PairParameters params){
	if(r2 >= params.cutOff2) return 0;
	real invr2 = params.sigma2/r2;
	real invr6 = invr2*invr2*invr2;
	real E = params.epsilonDivSigma2*params.sigma2*real(4.0)*invr6*(invr6-real(1.0)) - params.shift;
	// if(params.shift != real(0.0)){
	//   ////With shift, u(r) = lj(r)-lj(rc)  -(r-rc)·(dlj(r)/dr|_rc)
	//   // real rc = sqrt(params.cutOff2);
	//   // real invrc2 = real(params.sigma2)/(params.cutOff2);
	//   // real invrc6 = invrc2*invrc2*invrc2;
	//   // E += -(sqrt(r2)-rc)*params.shift - real(4.0)*params.epsilonDivSigma2*params.sigma2*invrc6*(invrc6-real(1.0));
	// }
	return real(0.5)*E;
      }

      static inline __host__ PairParameters processPairParameters(InputPairParameters in_par){
	PairParameters params;
	params.cutOff2 = in_par.cutOff*in_par.cutOff;
	params.sigma2 = in_par.sigma*in_par.sigma;
	params.epsilonDivSigma2 = in_par.epsilon/params.sigma2;
	if(in_par.shift){
	  real invCutOff2 = params.sigma2/params.cutOff2;
	  real invrc6 = invCutOff2*invCutOff2*invCutOff2;
	  //params.shift = params.epsilonDivSigma2*(real(48.0)*invrc13 - real(24.0)*invrc7);
	  params.shift = params.epsilonDivSigma2*params.sigma2*real(4.0)*invrc6*(invrc6-real(1.0));
	}
	else params.shift = real(0.0);
	return params;
      }

    };

    using LJ = Radial<LJFunctor>;
  }
}
#endif
