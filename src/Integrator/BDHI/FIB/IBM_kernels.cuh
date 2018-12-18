/*Raul P. Pelaez 2018. Immerse boundary kernels

USAGE:
Create with:

Kernel kern(cellSize);
\delta(rij) can be computed with kern.delta(rij);
Each kernel will need a certain number of support cells given by Kernel::support
support 3 means up to first neighbours (27 cells in 3D).

REFERENCES:
[1] Charles S. Peskin. The immersed boundary method (2002). DOI: 10.1017/S0962492902000077
 */
#ifndef IBMKERNELS_CUH
#define IBMKERNELS_CUH
namespace uammd{
  namespace IBM{      
    namespace PeskinKernel{
      //[1] Charles S. Peskin. The immersed boundary method (2002). DOI: 10.1017/S0962492902000077
      
      //Standard 3-point Peskin interpolator
      struct threePoint{
	real3 invh;
	static constexpr int support = 3;
	threePoint(real3 h):invh(1.0/h){}
	inline __device__ real phi(real r) const{
	  if(r<real(0.5)){
	    constexpr real onediv3 = real(1/3.0);
	    return onediv3*(real(1.0) + sqrt(real(1.0)+real(-3.0)*r*r));
	  }
	  else if(r<=real(1.5)){
	    constexpr real onediv6 = real(1/6.0);
	    const real omr = real(1.0) - r;
	    return onediv6*(real(5.0)-real(3.0)*r - sqrt(real(1.0) + real(-3.0)*omr*omr));
	  }
	  else return 0;
	}	  
	inline __device__ real delta(real3 rvec) const{
	    
	  return invh.x*invh.y*invh.z*phi(fabs(rvec.x*invh.x))*phi(fabs(rvec.y*invh.y))*phi(fabs(rvec.z*invh.z));
	}
	  
      };

      //Standard 4-point Peskin interpolator
      struct fourPoint{
	real3 invh;
	static constexpr int support = 4;
	fourPoint(real3 h):invh(1.0/h){}
	inline __device__ real phi(real r) const{
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
	  
      };
    }
    
    namespace GaussianFlexible{
      //[1] Yuanxun Bao, Jason Kaye and Charles S. Peskin. A Gaussian-like immersed-boundary kernel with three continuous derivatives and improved translational invariance. http://dx.doi.org/10.1016/j.jcp.2016.04.024
      //Adapted from https://github.com/stochasticHydroTools/IBMethod/
      struct sixPoint{
	sixPoint(real3 h):invh(1.0/h){}
      private:
	const real K = 59.0/60.0-sqrt(29.0)/20.0;
	inline __device__ int sgn (real x) const{
	  return ( (x > real(0)) ? 1 : ((x < real(0)) ? -1 : 0) );
	}
      public:
	real3 invh;
	static constexpr int support = 6;
	/* the new C3 6-pt kernel */
	inline __device__ real phi(real r) const{
	  real R = r - ceil(r) + real(1.0);  /* R between [0,1] */
	  real R2 = R * R;
	  real R3 = R2*R;
	  const real alpha = real(28.);
	  const real beta  = real(9.0/4.0) - real(1.5) * (K + R2) + (real(22./3)-real(7.0)*K)*R - real(7./3.)*R3;
	  real gamma;
	  {
	    const real r2pre=real(0.25)*real(0.5)*(real(161./36) - real(59./6)*K + real(5)*K*K);
	    gamma = r2pre*R2 + real(0.25/3)*((real(-109./24) + real(5.0)*K)*R2*R2 + real(5./18)*R3*R3 );
	  }
	  const real discr = beta*beta - real(4.0) * alpha * gamma;
	    
	  const int sgn = ((real(1.5)-K)>0) ? 1 : -1;   /* sign(3/2 - K) */
	    
	  if (r <= real(-3) || r>=real(3)) return 0;
	  const real prefactor = real(1.)/(real(2)*alpha) * ( -beta + sgn * sqrt(discr) );
	  if (r<=real(-2)){
	    return prefactor;
	  }
	  else if (r <= real(-1)){
	    const real rp2 = r+real(2.0);
	    return real(-3.)*prefactor -
	      real(1./16) +
	      real(1./8)*( K+rp2*rp2 ) +
	      real(1./12)*(real(3)*K-real(1.0))*rp2 + real(1./12)*rp2*rp2*rp2; 
	  }
	  else if (r <= real(0) ){
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
	  return invh.x*invh.y*invh.z*phi(rvec.x*invh.x)*phi(rvec.y*invh.y)*phi(rvec.z*invh.z);
	}

      };
    }

  }
}

#endif
