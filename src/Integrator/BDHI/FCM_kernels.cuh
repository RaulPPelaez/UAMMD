/*Raul P. Pelaez 2020, Kernels (Window  functions) for FCM.
  These are adapted from kernels in the IBM module, they are adapted to provide the resulting hydrodynamic radius and the support for a given tolerance.
 */
#ifndef FCM_KERNELS_CUH
#define FCM_KERNELS_CUH
#include "BDHI.cuh"
#include "misc/IBM_kernels.cuh"
namespace uammd{
  namespace BDHI{
    namespace FCM_ns{
      namespace Kernels{
	class Gaussian {
	  IBM_kernels::Gaussian kern;
	  static real computeUpsampling(real tolerance){
	    real amin = 0.55;
	    real amax = 1.65;
	    real x = -log10(2*tolerance)/10.0;
	    real factor = std::min(amin + x*(amax-amin), amax);
	    return factor;
	  }
	  real a;
	public:
	  int support;
	  Gaussian(real h, real tolerance):
	    kern(h*computeUpsampling(tolerance)){
	    const real dr = 1e-5*h;
	    real r = dr;
	    while(this->phi(r)>tolerance){
	      r+=dr;
	    }
	    this->support = std::max(3, int(2*r/h + 0.5));
	    a=h*computeUpsampling(tolerance)*sqrt(M_PI);
	  }

	  static real adviseGridSize(real hydrodynamicRadius, real tolerance){
	    real factor = computeUpsampling(tolerance);
	    return hydrodynamicRadius/(sqrt(M_PI)*factor);
	  }

	  real fixHydrodynamicRadius(real hydrodynamicRadius, real h){
	    return a;
	  }

	  __host__ __device__ real phi(real r) const{
	    return kern.phi(r);
	  }

	  __host__ __device__ real delta(real3 r, real3 h) const{
	    return phi(r.x)*phi(r.y)*phi(r.z);
	  }

	};

	class BarnettMagland{
	  IBM_kernels::BarnettMagland bm;

	  IBM_kernels::BarnettMagland initBM(real tolerance){
	    real w = computeW(tolerance);
	    real beta=sqrt(2*M_PI)*w*2;
	    return IBM_kernels::BarnettMagland(w, beta);
	  }
	  static real computeW(real tolerance){
	    real w = std::max(1.5, int(-log10(tolerance) + 2)/2.0);
	    w = std::min(real(9.0), w);
	    return w;
	  }
	  static real computeUpsampling(real w){
	    //Empirical fit from the taylor expansion of the BM kernel, the first term goes with x^2
	    //Which allows to make a simil with a gaussian such that \sigma = w/sqrt(beta)
	    return 1.36409985665115*pow(w,-0.53028415751646);
	    //This is the fitted data:
	    //fac = 4.863392908150500e-01; //w=7
	    //fac = 7.617116421559339e-01; //w=3
	    //fac = 1.258133110081786; //w=1
	    //fac = 1.097370161054899; //w=1.5
	    //fac = 9.482257823550468e-01; //w=2
	    //fac = 8.409078753286086e-01; //w=2.5
	    //fac = 7.009245406092016e-01; //w=3.5
	    //fac = 6.525743222080990e-01; //w=4
	    //fac = 5.269758326738209e-01; //w=6
	  }
	  real a;
	public:
	  int support;

	  BarnettMagland(real h, real tolerance):
	    bm(initBM(tolerance)){
	    support = int(2*bm.w + 0.5);
	    a = h;
	  }

	  static real adviseGridSize(real hydrodynamicRadius, real tolerance){
	    real w = computeW(tolerance);
	    real upsampling = computeUpsampling(w);
	    return hydrodynamicRadius*upsampling;
	  }

	  real fixHydrodynamicRadius(real hydrodynamicRadius, real h) const{
	    real upsampling = computeUpsampling(bm.w);
	    return h/upsampling;
	  }

	  __host__ __device__ real phi(real r) const{
	    return bm.phi(r/a)/a;
	  }

	  __host__ __device__ real delta(real3 r, real3 h) const{
	    return phi(r.x)*phi(r.y)*phi(r.z);
	  }
	};

	namespace Peskin{

	  class threePoint{
	    IBM_kernels::Peskin::threePoint kern;
	  public:
	    static constexpr int support = 3;

	    threePoint(real h, real tolerance): kern(h){
	    }

	    static real adviseGridSize(real hydrodynamicRadius, real tolerance){
	      return hydrodynamicRadius/0.975;
	    }

	    real fixHydrodynamicRadius(real hydrodynamicRadius, real h) const{
	      return h*0.975;
	    }

	    __host__ __device__ real phi(real r) const{
	      return kern.phi(r);
	    }

	    __host__ __device__ real delta(real3 r, real3 h) const{
	      return kern.phi(r.x)*kern.phi(r.y)*kern.phi(r.z);
	    }

	  };

	  class fourPoint{
	    IBM_kernels::Peskin::fourPoint kern;
	  public:
	    static constexpr int support = 4;

	    fourPoint(real h, real tolerance): kern(h){
	    }

	    static real adviseGridSize(real hydrodynamicRadius, real tolerance){
	      return hydrodynamicRadius/1.3157892485;
	    }

	    real fixHydrodynamicRadius(real hydrodynamicRadius, real h) const{
	      return h*1.3157892485;
	    }

	    __host__ __device__ real phi(real r) const{
	      return kern.phi(r);
	    }

	    __host__ __device__ real delta(real3 r, real3 h) const{
	      return kern.phi(r.x)*kern.phi(r.y)*kern.phi(r.z);
	    }

	  };
	}

	namespace GaussianFlexible{
	  class sixPoint{
	    IBM_kernels::GaussianFlexible::sixPoint kern;
	  public:
	    static constexpr int support = 6;

	    sixPoint(real h, real tolerance): kern(h, tolerance){
	    }

	    static real adviseGridSize(real hydrodynamicRadius, real tolerance){
	      return hydrodynamicRadius/1.519854;
	    }

	    real fixHydrodynamicRadius(real hydrodynamicRadius, real h) const{
	      return h*1.519854;
	    }

	    __host__ __device__ real phi(real r) const{
	      return kern.phi_tabulated(r);
	    }

	    __host__ __device__ real delta(real3 r, real3 h) const{
	      return phi(r.x)*phi(r.y)*phi(r.z);
	    }

	  };
	}

      }
    }
  }
}

#endif
