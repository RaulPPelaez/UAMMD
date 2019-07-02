/*Raul P. Pelaez 2019. Quasi2D Integrator.
This integrator solves the Brownian Dynamics with Hydrodynamic interactions equation where
particles are restricted to move in a 2D enviroment.
The hydrodynamic kernel is free to choose as a kernel argument, mainly:
 1- True2D  -> Pure 2D hydrodynamics, using the 2D FCM kernel (just 3D with z=0)
 2- Quasi2D -> 3D hydrodynamics acting on 2D particles. 3D FCM kernel integrated in z.
 3- Saffman -> Saffman hydrodynamic kernel.

Any hydrodynamic kernel can be provided as a functor with the following prototype (i.e for True2D):

struct True2D{
//This function returns the relation between the gaussian kernel variance and the hydrodynamic radius,
// which varies with the kernel
  static inline __host__ __device__ real getGaussianVariance(real a){
    return pow(a*0.66556976637237890625, 2);
  }
//This function returns whether the kernel needs to spread thermal drift or not
  static inline bool hasThermalDrift(){ return false;}

//This function returns f_k and g_k in the Fourier representation of the hydrodynamic Green function:
// \hat{G}_\vec{k} = 1/\eta [ g_k(k*a)  \vec{k_perp}\dyadic\vec{k_perp} + f_k(k*a\vec{k}\dyadic\vec{k} ]
//See eq.25 in [1].

__device__ real2 operator()(real k2, real a){
  //For True2D:
  real f_k = 0;
  real g_k = 1/(k2*k2)
  return {f_k, g_k};
}
};
This file encodes the algorithm described in [1].

The spreading/interpolation is currently done via a Gaussian kernel.

USAGE:

    Create the module as any other integrator with the following parameters:

    auto sys = make_shared<System>();
    auto pd = make_shared<ParticleData>(N,sys);
    auto pg = make_shared<ParticleGroup>(pd,sys, "All");

    using Kernel = BDHI::Quasi2D::q2D;
    using Hydro2D = BDHI::Quasi2D<Kernel>;

    Hydro2D::Parameters par;
    par.temperature = temperature;
    par.viscosity = viscosity;
    par.dt = dt;
    par.hydrodynamicRadius = hydrodynamicRadius;
    //par.cells = make_int2(cells_x, cells_y); //Cell dimensions control accuracy, can be forced
    par.tolerance = 1e-5; //Controls accuracy throguh the number of cells
    par.box = box; //boxSize.z will be ignored if != 0
    //par.hydroKernel = std::make_shared<Kernel>(...) //Optionally you can pass an instance of the hydrodynamic kernel to be used.
    auto q2D = make_shared<Hydro2D>(pd, pg, sys, par);

    //Add any interactor
    q2D->addInteractor(...);
    ...

    //forward simulation 1 dt:
    q2D->forwardTime();



References:
[1] Hydrodynamic fluctuations in quasi-two dimensional diffusion. Raul P. Pelaez, et. al. JSTAT 2018
*/
#ifndef BDHI_QUASI2D_CUH
#define BDHI_QUASI2D_CUH

#include "Integrator/Integrator.cuh"
#include "Integrator/BDHI/BDHI.cuh"
#include "misc/IBM.cuh"
#include "utils/utils.h"
#include "global/defines.h"
#include<thread>
#include"utils/Grid.cuh"
#include"utils/cufftPrecisionAgnostic.h"

#include"utils/cufftComplex2.cuh"
#include"third_party/managed_allocator.h"
namespace uammd{
  namespace BDHI{

    namespace BDHI2D_ns{

      //See eq.21 and beyond in  [1]
      struct True2D{
	static inline __host__ __device__ real getGaussianVariance(real a){
	  return pow(a*0.66556976637237890625, 2);
	}
	static constexpr inline bool hasThermalDrift(){ return false;}
	inline __device__ real2 operator()(real k2, real a){
	  const real fk = 0;
	  const real gk = real(1.0)/(k2*k2);
	  return {fk, gk};
	}
      };
      //See eq. 20 and beyond in [1]
      struct Quasi2D{
	static constexpr inline bool hasThermalDrift(){ return true;}
	static inline __host__ __device__ real getGaussianVariance(real a){
	  return pow(a/sqrt(M_PI), 2);
	}
	inline __device__ real2 operator()(real k2, real a){
	  const real k = sqrt(k2);
	  const real invk3 = real(1.0)/(k*k*k);
	  constexpr real inv_sqrtpi = 0.564189583547756; //1/sqrt(pi)
	  constexpr real invpi = 1.0/M_PI;
	  const real fk = real(0.5)*invpi*invk3*(-k*a +
						 + exp(k2*invpi*a*a)*(k2*a*a + real(M_PI)*real(0.5))*erfc(k*a*inv_sqrtpi)
						 );
	  const real gk = real(0.5)*invk3*erfc(k*a*inv_sqrtpi)*exp(k2*a*a*invpi);
	  return {fk, gk};
	}


      };

      template<bool thermalDrift>
      struct Gaussian{
	int support;
        Gaussian(int support, real width):support(support){
	  this-> prefactor = 1.0/(2.0*M_PI*width);
	  this-> tau = -1.0/(2.0*width);
	  sup = 0.5*support;
	}

	inline __device__ real2 delta(real3 rvec, real3 h) const{
	  const real r2 = dot(rvec, rvec);
	  //if(r2>sup*sup*h.x*h.x) return 0;
	  if(thermalDrift)
	    return -prefactor*exp(tau*r2)*make_real2(rvec);
	  else
	    return make_real2(prefactor*exp(tau*r2));
	}
      private:
	real prefactor;
	real tau;
	real sup;
      };

      using cufftComplex2 = cufftComplex2_t<real>;
      using cufftComplex = cufftComplex_t<real>;
      using cufftReal = cufftReal_t<real>;

    }
    template<class HydroKernel>
    class BDHI2D: public Integrator{
    public:
      //WARNING! the code is specifically tailored for a Gaussian kernel at the moment.
      //It assumes that d\delta(r)/dr = \sigma \delta(r)
      using Kernel = BDHI2D_ns::Gaussian<false>;
      using KernelThermalDrift = BDHI2D_ns::Gaussian<true>;

      using cufftComplex2 = BDHI2D_ns::cufftComplex2;
      using cufftComplex = BDHI2D_ns::cufftComplex;
      using cufftReal = BDHI2D_ns::cufftReal;

      struct Parameters: BDHI::Parameters{
	int2 cells = make_int2(-1, -1); //Number of Fourier nodes in each direction
	std::shared_ptr<HydroKernel> hydroKernel;
      };

      BDHI2D(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par);
      ~BDHI2D();

      virtual void forwardTime() override;


    private:
      ullint seed;

      real temperature;
      real dt;
      real viscosity;
      real hydrodynamicRadius;

      std::shared_ptr<IBM<Kernel>> ibm;
      std::shared_ptr<IBM<KernelThermalDrift>> ibmThermalDrift;
      std::shared_ptr<HydroKernel> hydroKernel;

      Box box;

      Grid grid;

      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea; //Work space for cufft
      // thrust::device_vector<cufftComplex> gridVelsFourier;
      // thrust::device_vector<real2> particleVels;

      managed_vector<cufftComplex> gridVelsFourier;
      managed_vector<real2> particleVels;

      cudaStream_t st, st2;

      void initCuFFT();
      void spreadParticles();
      void convolveFourier();
      void interpolateParticles();

    };
  }
}

#include "BDHI_quasi2D.cu"
namespace uammd{
  namespace BDHI{
    using True2D = BDHI2D<BDHI2D_ns::True2D>;
    using Quasi2D = BDHI2D<BDHI2D_ns::Quasi2D>;

  }
}
#endif
