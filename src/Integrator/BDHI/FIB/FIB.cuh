/*Raul P. Pelaez 2018. Fluctuating Immerse Boundary for Brownian Dynamics with Hydrodynamic Interactions.

This file implements the algorithm described in [1] for PBC using FFT to solve the stokes operator. Fluid properties are stored in a staggered grid for improved translational invariance [2]. This module solves the same regime as the rest of the BDHI modules, but without imposing a mobility kernel. FIB solves the fluctuating Navier-Stokes equation directly.


USAGE:
Use it as any other module:

BDHI::FIB::Parameters par;
par.temperature = 1.0;
par.viscosity = 1.0;
//The hydrodynamic radius is given by ~0.91*Lbox/cellDimension in FIB, so the hydrodynamic radius given here cannot be enforced exactly, rather the most approximate one will be computed by selecting an appropiate cellDimension (which will be an FFT wise number).
par.hydrodynamicRadius = 1.0;
//Instead of the hydrodynamic radius the cell dimensions can be provided directly. Giving rh~0.91·L/cellDim
//par.cells=make_int3(64,64,64); //cells.z == 1 means 2D
//In any case a message will be issued with the used hydrodynamic radius.
par.dt = 0.01;
par.box = Box(64);
//You can choose between two mid point schemes for temporal integration:
//par.scheme = BDHI::FIB::MIDPOINT;
//par.scheme = BDHI::FIB::IMPROVED_MIDPOINT; -> Default
//The MIDPOINT scheme is faster as it requires only one Stokes solve and thus half the FFTs as the improved one. Furthemore the simple midpoint scheme computes particle forces only one time as opposed to two times in the improved scheme. Both algorithms are described in [1].

auto bdhi = make_shared<BDHI::FIB>(pd, sys, par);

//add any interactor

bdhi->forwardTime();

The function getSelfMobility() will return the self mobility of the particles.
The function getHydrodynamicRadius() will return the used hydrodynamic radius. You should expect a variation of +- 1% between this value and the one you measure from simulations.

----------------------------------

A brief summary of the algorithm:

The incompressible fluctuating Navier-Stokes equation in the creeping flow regime can be written like this for the fluid velocity:

  \vec{v_fluid} = 1/\eta\mathcal{L}^-1\vec{g} -> fluid velocity

  Where \mathcal{L} is an operator coming from a proyection method that allows to solve the velocity without computing or storing the pressure and can be written as:
  \mathcal{L}^-1 = -L^{-1}P 
  Where L is the Laplacian operator and P aa projector onto the divergence-free space.
  P = I-L^{-1}·G -> \hat{P} = I-k·k^T/|k|^2 with PBC, G is gradient, \hat represents Fourier space

So \hat{\mathcal{L}^-1} =  1/|k|^2 (I - k·k^T/|k|^2)

The different quantities are stored in a staggered grid, see [2].
Vectors (velocity) is stored at cell faces, scalars at cell centers and tensors at cell nodes (noise).

The fluid forcing can be written as:

\vec{g} = S^n·F^n + sqrt(2·\eta·kT/(dt·dV))·\hat{D}\bf{W}^{n,1} + kT/\delta [ S(q^n + \delta/2\hat{\bf{W}}^n) - S(q^n - \delta/2\hat{\bf{W}}^n)]\hat{\bf{W}}^n = particle forces + random advection + thermal drift
Where
S: Spreading operator.
\hat{D}: Stochastic divergence .
\bf{W}: Collection of independent Wienner processes.
F: Forces acting on particles

v_particles = J·v_fluid

J: Interpolation operator J^T = dV·S

The particles are updated following the mid point (predictor-corrector) schemes developed in [1]. See midPointStep and improvedMidPointStep for more info.
Euler: x^{t+dt} = x^t + v_particles·dt


About J and S:

S_cellj = sum_i=0^N{\delta(ri-r_cellj)} ->particles to fluid
J_i = dV·sum_j=0^ncells{\delta(ri-r_cellj)}->fluid to particles

Where \delta is an spreading kernel (smeared delta).
The default kernel used is the 3-point Peskin kernel, see IBM_kernels.cuh. 
But others can be selected with arbitrary support to better describe or satisfy certain conditions.



REFERENCES:
[1] Brownian Dynamics without Green's Functions. Steve Delong, Florencio Balboa, et. al. 2014
[2] Staggered Schemes for Fluctuating Hydrodynamics. Florencio Balboa, et.al. 2012.

*/

#ifndef INTEGRATORBDHIFIB_CUH
#define INTEGRATORBDHIFIB_CUH
#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"global/defines.h"

#include"Integrator/Integrator.cuh"
#include<thrust/device_vector.h>

#include "utils/utils.h"
#include"Interactor/NeighbourList/CellList.cuh"
#include"utils/cufftPrecisionAgnostic.h"
#include"utils/Grid.cuh"
#include<curand.h>


#include"IBM_kernels.cuh"

namespace uammd{
  namespace BDHI{

    namespace FIB_ns{
      /*A convenient struct to pack 3 complex numbers, that is 6 real numbers*/
      struct cufftComplex3{
	cufftComplex_t<real> x,y,z;
      };

      inline __device__ __host__ cufftComplex3 operator+(cufftComplex3 a, const cufftComplex3 &b){
	return {a.x + b.x, a.y + b.y, a.z + b.z};
      }
      inline __device__ __host__ void operator+=(cufftComplex3 &a, const cufftComplex3 &b){
	a.x += b.x; a.y += b.y; a.z += b.z;
      }
      inline __device__ __host__ cufftComplex3 operator*(const real &a, const cufftComplex3 &b){
	return {a*b.x, a*b.y, a*b.z};
      }
      inline __device__ __host__ cufftComplex3 operator*(const cufftComplex3 &b, const real &a){
	return a*b;
      }


    }
    class FIB: public Integrator{
    public:
      enum Scheme{MIDPOINT, IMPROVED_MIDPOINT};
      struct Parameters{
	real temperature;
	real viscosity;
	real hydrodynamicRadius = -1;
	real dt;
	Box box;
	int3 cells={-1, -1, -1}; //Default is compute the closest number of cells that is a FFT friendly number
	Scheme scheme = Scheme::IMPROVED_MIDPOINT;       
      };

      FIB(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,		       
	  Parameters par);
      FIB(shared_ptr<ParticleData> pd,
	  shared_ptr<System> sys,		       
	  Parameters par):
	FIB(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, par){}
      				     
      ~FIB();

      void forwardTime() override;
      real sumEnergy() override;

      using cufftComplex3 = FIB_ns::cufftComplex3;
      using cufftComplex = cufftComplex_t<real>;
      using cufftReal = cufftReal_t<real>;

      real getSelfMobility(){
	double rh = 0.91*box.boxSize.x/grid.cellDim.x;
	return 1.0/(6*M_PI*viscosity*rh)*(1-2.837297*rh/box.boxSize.x);
      }
      real getHydrodynamicRadius(){
	return 0.91*box.boxSize.x/(real)grid.cellDim.x;
      }
    private:
      using Kernel = IBM::PeskinKernel::threePoint;
      real temperature, viscosity;      
      
      shared_ptr<CellList> cl;
      
      Grid grid;
      Box box;
      
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<real> cufftWorkArea; //Work space for cufft

      //Grid forces/velocities in fourier/real space
      thrust::device_vector<cufftComplex> gridVels; 

      curandGenerator_t curng;
      thrust::device_vector<real> random;
      
      //Temporal integration variables
      real dt;
      Scheme scheme; //Temporal integration mode
      
      thrust::device_vector<real4> posOld; //q^n
      real deltaRFD; //Random finite diference step      
      ullint step = 0;
      
      ullint seed = 1234;


      void spreadParticleForces();
      void randomAdvection(real prefactor);
      void thermalDrift();

      void applyStokesSolutionOperator();
    
      void predictorStep();
      void correctorStep();
      void eulerStep();

      void forwardMidpoint();
      void forwardImprovedMidpoint();
      
    };

  }
}
#include"FIB.cu"
#endif