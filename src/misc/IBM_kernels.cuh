/*Raul P. Pelaez 2018. Immerse boundary kernels

USAGE:
Create with:

Kernel kern(cellSize, tolerance);
\delta(rij) can be computed with kern.delta(rij);
Each kernel will need a certain number of support cells given by Kernel::support
support 3 means up to first neighbours (27 cells in 3D).

the expected hydrodynamic radius for the given cellsize and tolerance can be obtained with kern.getHydrodynamcRadius(IBM::SpatialDiscretization::Spectral); (or any other SpatialDiscretization).


INTERFACE:

An IBM kernel must be a class with these characteristics:

   -A constructor that takes a cell size and a tolerance (which might be unused)
   -A "delta" member device function that takes a real3 distance and returns a real weight, prototype:
       	__device__ real delta(real3 rvec);
   -A function that returns the expected hydrodynamic radius for a given SpatialDiscretization (returns -1 if unknown):
      real getHydrodynamicRadius(SpatialDiscretization sd) const;
   -A public member called support with the necessary support cells (3 means first neighbours, 27 cells in total in 3D).


REFERENCES:
[1] Charles S. Peskin. The immersed boundary method (2002). DOI: 10.1017/S0962492902000077
[2] Fluctuating force-coupling method for simulations of colloidal suspensions. Eric E. Keaveny. 2014.
 */
#ifndef IBMKERNELS_CUH
#define IBMKERNELS_CUH
#include"misc/TabulatedFunction.cuh"
namespace uammd{
  namespace IBM_kernels{
    enum class SpatialDiscretization{Staggered, Centered, Spectral};     

    struct GaussianKernel{
      int support;
      GaussianKernel(real3 h, real tolerance){
	this->support = int(-log10(tolerance)+0.5);
	this->hydrodynamicRadius = (0.5*support)*h.x/3.0;
	//this->hydrodynamicRadius = h.x*(1.0-log10(tolerance)/10.0);
	//eq. 8 in [2], \sigma_\Delta
	real sigma = hydrodynamicRadius/sqrt(M_PI); //hydrodynamic radius is sqrt(M_PI)*sigma;
	this->prefactor = pow(2*M_PI*sigma*sigma, -1.5);
	this->tau  = -0.5/(sigma*sigma);	  
	//According to [2] the Gaussian kernel can be considered 0 beyond 3*a, so P >= 3*a/h
	//this->support = 2*int(3.0*hydrodynamicRadius/h.x+0.5)+1;
	
	
	
	
      }

      inline __host__ __device__ real phi(real r) const{
	return pow(prefactor,1/3.0)*exp(tau*r*r);
      }
      
      inline __device__ real delta(real3 rvec) const{	    
	return prefactor*exp(tau*dot(rvec, rvec));
      }
      inline real getHydrodynamicRadius(SpatialDiscretization sd) const{
	return hydrodynamicRadius;
      }

    private:
      real prefactor;
      real tau;
      real hydrodynamicRadius;
    };

    
    namespace PeskinKernel{
      //[1] Charles S. Peskin. The immersed boundary method (2002). DOI: 10.1017/S0962492902000077      
      //Standard 3-point Peskin interpolator
      struct threePoint{
	real3 invh;
	static constexpr int support = 3;
	threePoint(real3 h, real tolerance = 0):invh(1.0/h){}
	inline __host__ __device__ real phi(real r) const{
	  if(r<real(0.5)){
	    constexpr real onediv3 = real(1/3.0);
	    return onediv3*(real(1.0) + sqrt(real(1.0)+real(-3.0)*r*r));
	  }
	  else if(r<real(1.5)){
	    constexpr real onediv6 = real(1/6.0);
	    const real omr = real(1.0) - r;
	    return onediv6*(real(5.0)-real(3.0)*r - sqrt(real(1.0) + real(-3.0)*omr*omr));
	  }
	  else return 0;
	}
	inline __device__ real delta(real3 rvec) const{
	    
	  return invh.x*invh.y*invh.z*phi(fabs(rvec.x*invh.x))*phi(fabs(rvec.y*invh.y))*phi(fabs(rvec.z*invh.z));
	}

        inline real getHydrodynamicRadius(SpatialDiscretization sd) const{
	  switch(sd){
	  case SpatialDiscretization::Staggered: return 0.91/invh.x;
	    //case SpatialDiscretization::Spectral:  return 0.971785649216029/invh.x; //exact at 0.5h, +-1e-2 variation across unit cell
	  case SpatialDiscretization::Spectral:  return 0.975/invh.x; //+-1e-2 variation across unit cell
	  default: return -1; 	  
	  }
	}	
      };

      //Standard 4-point Peskin interpolator
      struct fourPoint{
	real3 invh;
	static constexpr int support = 4;
	fourPoint(real3 h, real tolerance = 0):invh(1.0/h){}
	inline __host__  __device__ real phi(real r) const{
	  constexpr real onediv8 = real(0.125);
	  if(r<real(1.0)){
	    return onediv8*(real(3.0) - real(2.0)*r + sqrt(real(1.0)+real(4.0)*r*(real(1.0)-r)));
	  }
	  else if(r<real(2.0)){
	    return onediv8*(real(5.0) - real(2.0)*r - sqrt(real(-7.0) + real(12.0)*r-real(4.0)*r*r));
	  }
	  else return 0;
	}
	inline __device__ real delta(real3 rvec) const{
	  return invh.x*invh.y*invh.z*phi(fabs(rvec.x*invh.x))*phi(fabs(rvec.y*invh.y))*phi(fabs(rvec.z*invh.z));
	}
	
	inline real getHydrodynamicRadius(SpatialDiscretization sd) const{
	  switch(sd){
	  case SpatialDiscretization::Staggered: return 1.255/invh.x;
	    //case SpatialDiscretization::Spectral:  return 1.31275/invh.x;
	    //case SpatialDiscretization::Spectral:  return 1.321553589/invh.x; //exact at x=0
	    case SpatialDiscretization::Spectral:  return 1.3157892485/invh.x; //exact at x=0.5h, +-4e-3 variation across unit cell
	  default: return -1; 	  
	  }
	}
      };
    }
    
    namespace GaussianFlexible{
      //[1] Yuanxun Bao, Jason Kaye and Charles S. Peskin. A Gaussian-like immersed-boundary kernel with three continuous derivatives and improved translational invariance. http://dx.doi.org/10.1016/j.jcp.2016.04.024
      //Adapted from https://github.com/stochasticHydroTools/IBMethod/
      struct sixPoint{
      private:
	static constexpr real K = 0.714075092976608; //59.0/60.0-sqrt(29.0)/20.0;
	TabulatedFunction<real> phi_tab;
	real3 invh;
      public:	
	sixPoint(real3 h, real tolerance = 1e-7):
	  invh(1.0/h),
	  phi_tab(int(1e5*(-log10(tolerance)/20.0)), 0, 3, phi)
	{}
	~sixPoint() = default;

	static constexpr int support = 6;
	/* the new C3 6-pt kernel */
	static inline __host__  __device__ real phi(real r){
	  //if (r <= real(-3) || r>=real(3)) return 0;
	  if (r>=real(3)) return 0;
	  real R = r - ceil(r) + real(1.0);  /* R between [0,1] */
	  real R2 = R * R;
	  real R3 = R2*R;
	  const real alpha = real(28.);
	  const real beta  = real(9.0/4.0) - real(1.5) * (K + R2) + (real(22./3)-real(7.0)*K)*R - real(7./3.)*R3;
	  real gamma = real(0.25) * ( real(0.5)*(real(161.)/real(36)
						 - real(59.)/real(6)*K
						 + real(5)*K*K)*R2
				      + real(1.)/real(3)*(real(-109.)/real(24)
							  + real(5)*K)*R2*R2
				      + real(5.)/real(18)*R3*R3);
	  
	  const real discr = beta*beta - real(4.0) * alpha * gamma;
	    
	  const int sgn = ((real(1.5)-K)>0) ? 1 : -1;   /* sign(3/2 - K) */	    
	  const real prefactor = real(1.)/(real(2)*alpha) * ( -beta + sgn * sqrt(discr) );
	  // if (r<=real(-2)){
	  //   return prefactor;
	  // }
	  // else if (r <= real(-1)){
	  //   const real rp2 = r+real(2.0);
	  //   return real(-3.)*prefactor -
	  //     real(1./16) +
	  //     real(1./8)*( K+rp2*rp2 ) +
	  //     real(1./12)*(real(3)*K-real(1.0))*rp2 + real(1./12)*rp2*rp2*rp2; 
	  // }
	  if (r <= real(0) ){
	    const real rp1 = r+real(1.0);
	    return  real(2.)*prefactor +
	      real(0.25) +
	      real(1./6)*(real(4)-real(3)*K)*rp1 -
	      real(1./6)*rp1*rp1*rp1;
	  }
	  else if (r <= real(1) ){
	    return real(2.0)*prefactor +
	      real(5./8) - real(0.25)*( K+r*r );
	  }
	  else if (r <= real(2) ){
	    const real rm1 = r+real(-1.0);
	    return real(-3.0)*prefactor +
	      real(0.25) -
	      real(1./6.)*(real(4)-real(3)*K)*rm1 +
	      real(1./6)*rm1*rm1*rm1;
	  }
	  else if (r <= real(3) ){
	    const real rm2 = r+real(-2.0);
	    return prefactor -
	      real(1./16) +
	      real(1./8)*(K+rm2*rm2) - real(1./12)*(real(3)*K-real(1))*rm2 - real(1./12)*rm2*rm2*rm2;
	  }
	  return real(0.0);
	}

	inline __device__ real delta(real3 rvec) const{
	  //Uncomment to evaluate instead of reading from a table
	  //return invh.x*invh.y*invh.z*phi(fabs(rvec.x*invh.x))*phi(fabs(rvec.y*invh.y))*phi(fabs(rvec.z*invh.z));
	  return invh.x*invh.y*invh.z*phi_tab(fabs(rvec.x*invh.x))*phi_tab(fabs(rvec.y*invh.y))*phi_tab(fabs(rvec.z*invh.z));
	}


	inline real getHydrodynamicRadius(SpatialDiscretization sd) const{
	  switch(sd){
	  case SpatialDiscretization::Staggered: return -1;
	  case SpatialDiscretization::Spectral:  return 1.519854/invh.x; //exact at x=0.5h, +-1e-4 variation across unit cell
	  default: return -1; 	  
	  }
	}

	
      };
    }

    //[1] Taken from https://arxiv.org/pdf/1712.04732.pdf
    struct BarnettMagland{
    private:
      real3 invh;
      real maxDist;
      real beta;
      real pref;
      real hydrodynamicRadius;
    public:
      int support;
      BarnettMagland(real3 h, real tolerance = 1e-10):
	invh(1.0/h){	
	this->support = int(-log10(tolerance)+0.5);
	this->maxDist = support*0.5;
	this->pref = 1.0/support*invh.x*invh.y*invh.z;
	beta=(sqrt(2*M_PI)*support);
      }
      inline __host__  __device__ real phi(real r) const{
	if(r>=maxDist) return 0;
	const real z = r/maxDist;
	return exp(beta*(sqrt(real(1.0)-z*z)-real(1.0)));

      }
      inline __device__ real delta(real3 rvec) const{
	return pref*phi(rvec.x*invh.x)*phi(rvec.y*invh.y)*phi(rvec.z*invh.z);
      }
	
      inline real getHydrodynamicRadius(SpatialDiscretization sd) const{
	real a,b,c,d;
	switch(sd){
	case SpatialDiscretization::Staggered: return -1;
	case SpatialDiscretization::Spectral:
	  //This is an empirical fit
	  a = 0.292065997466564;
	  b = 6.18161635236421;
	  c = -13.5250825121145;
	  d = 22.6266864452691;
	  return (a+b/support+c/(support*support)+d/(support*support*support))/invh.x;	  
	default: return -1; 	  
	}
      }
    };



    

  }
}

#endif
