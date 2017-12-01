/*Raul P. Pelaez 2017. Some Brownian Hydrodynamics utilities and definitions
 */
#ifndef BDHI_CUH
#define BDHI_CUH

#include"global/defines.h"
#include"utils/Box.cuh"
#include<vector>
namespace uammd{
  namespace BDHI{
    
    //Parameters that BDHI modules may need
    struct Parameters{
      //The 3x3 shear matrix is encoded as an array of 3 real3
      std::vector<real3> K;
      real temperature;
      real viscosity;
      real hydrodynamicRadius;
      real tolerance = 1e-3;
      real dt;
      bool is2D = false;
      Box box;
    };


    //The Rotne-Prager-Yamakawa tensor
    struct RotnePragerYamakawa{
      real rh;
      real invrh;
      RotnePragerYamakawa(real rh): rh(rh){this->invrh = 1.0/rh;}
      /*RPY tensor as a function of distance, r*/
      /*M(r) = 0.75*M0*( f(r)*I + g(r)*r(diadic)r )*/
      /*c12.x = f(r) * 0.75*M0    ->M0 is outside               
	c12.y = g(r) * 0.75*M0*/
      inline __host__  __device__  real2  RPY(real r) const{
	/*Distance in units of rh*/
	r *= invrh;
	const real invr  = real(1.0)/r;

	real2 c12;

	if(r >= real(2.0)){
	  const real invr2 = invr*invr;
	  c12.x = (real(0.75) + real(0.5)*invr2)*invr;
	  c12.y = (real(0.75) - real(1.5)*invr2)*invr;
	}
	else{
	  c12.x = real(1.0)-real(0.28125)*r;
	  if(r>real(0.0))
	    c12.y = real(0.09375)*r;
	}

	return c12;
      }
      
      inline __host__  __device__  real2  operator()(real r) const{
	return RPY(r);
      }
      /*Helper function for divergence in RDF,
      computes {f(r+dw)-f(r), g(r+dw)-g(r)}
      See diffusionDot for more info
      */
      inline __device__ real2  divergenceRDF(real3 rij, real3 dwij) const{

	const real r    = sqrtf(dot(rij, rij));
	const real3 rpdwij = rij+dwij;
	const real rpdw = sqrtf(dot(rpdwij, rpdwij));

	return RPY(rpdw)-RPY(r);
      }              

    
    };    
  }
}


#endif