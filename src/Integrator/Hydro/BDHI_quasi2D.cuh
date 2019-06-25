/*Raul P. Pelaez 2019. Quasi2D Integrator.
This integrator solves the Brownian Dynamics with Hydrodynamic interactions equation where 
particles are restricted to move in a 2D enviroment. 
The hydrodynamic kernel is free to choose as a kernel argument, mainly:
 1- True2D  -> Pure 2D hydrodynamics, using the 2D FCM kernel (just 3D with z=0)
 2- Quasi2D -> 3D hydrodynamics acting on 2D particles. 3D FCM kernel integrated in z.
 3- Saffman -> Saffman hydrodynamic kernel.

Any hydrodynamic kernel can be provided as a functor with the following prototype (i.e for True2D):

//This function returns f_k and g_k in the Fourier representation of the hydrodynamic Green function:
// \hat{G}_\vec{k} = 1/\eta [ g_k(k*a)  \vec{k_perp}\dyadic\vec{k_perp} + f_k(k*a\vec{k}\dyadic\vec{k} ]
//See eq.25 in [1].
struct True2D{
__device__ real2 operator()(real k2){
  //For True2D:
  real f_k = 0;
  real g_k = 1/(k2*k2)
  return {f_k, g_k};
}
};
This file encodes the algorithm described in [1].

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

    namespace Quasi2D_ns{

      struct Gaussian{
	int support;
        Gaussian(int support, real width):support(support){
	  this-> prefactor = 1.0/(2.0*M_PI*width);
	  this-> tau = -1.0/(2.0*width);
	  sup = 0.5*support;
	}
      
	inline __device__ real delta(real3 rvec, real3 h) const{
	  const real r2 = dot(rvec, rvec);
	  //if(r2>sup*sup*h.x*h.x) return 0;
	  return prefactor*exp(tau*r2);
	}
      private:
	real prefactor;
	real tau;
	real sup;
      };

    }

    class Quasi2D: public Integrator{
    public:
      using Kernel = Quasi2D_ns::Gaussian;
      //using Kernel = IBM_kernels::BarnettMagland;
      //using Kernel = IBM_kernels::PeskinKernel::fourPoint;
      //using Kernel = IBM_kernels::PeskinKernel::threePoint;
      //using Kernel = IBM_kernels::GaussianFlexible::sixPoint;

      using cufftComplex2 = cufftComplex2_t<real>;
      using cufftComplex = cufftComplex_t<real>;
      using cufftReal = cufftReal_t<real>;
      struct Parameters: BDHI::Parameters{
	int2 cells = make_int2(-1, -1); //Number of Fourier nodes in each direction
      };     
      real fac;
      Quasi2D(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par);
      ~Quasi2D();
      
      void computeMF();    
      virtual void forwardTime() override;
      
      
    private:
      ullint seed;
      
      real temperature;
      real dt;
      real viscosity;

      shared_ptr<IBM<Kernel>> ibm;
      
      Box box;

     /****Far (wave space) part) ******/
      Grid grid; /*Wave space Grid parameters*/
           
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea; //Work space for cufft     
      //thrust::device_vector<cufftComplex> gridVelsFourier; //Interpolated grid forces/velocities in fourier/real space
      managed_vector<cufftComplex> gridVelsFourier; //Interpolated grid forces/velocities in fourier/real space
      //thrust::device_vector<real2> particleVels;
      managed_vector<real2> particleVels;      
      cudaStream_t st;
     
      void initCuFFT();
      void spreadParticles();
      void convolveFourier();
      void interpolateParticles();

    };
  }
}

#include "BDHI_quasi2D.cu"
#endif
