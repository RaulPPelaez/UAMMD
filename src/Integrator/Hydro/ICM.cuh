/*Raul P. Pelaez 2018-2021. Inertial Coupling Method for particles in an incompressible fluctuating fluid.

This file implements the algorithm described in [1] for PBC using FFT to solve the stokes operator. Fluid properties are stored in a staggered grid for improved translational invariance [2]. ICM solves the incompressible fluctuating Navier-Stokes equation and couples the resulting fluid with immersed particles particles with the Immerse Boundary Method.
Currently the case with excess mass 0 is encoded.

USAGE:

Use it as any other module:

Hydro::ICM::Parameters par;
par.temperature = 1.0;
par.density = 1.0;
par.viscosity = 1.0;
//The hydrodynamic radius is given by ~0.91*Lbox/cellDimension in ICM, so the hydrodynamic radius given here cannot be enforced exactly, rather the most approximate one will be computed by selecting an appropiate cellDimension (which will be an FFT wise number).
par.hydrodynamicRadius = 1.0;
//Instead of the hydrodynamic radius the cell dimensions can be provided directly. Giving rh~0.91·L/cellDim
//par.cells=make_int3(64,64,64); //cells.z == 1 means 2D
//In any case a message will be issued with the used hydrodynamic radius.
par.dt = 0.01;
par.box = Box(64);
//par.sumThermalDrift = false; //Default is false, controls if the thermal drift is taken into account or not.

auto bdhi = make_shared<BDHI::FIB>(pd, sys, par);

//add any interactor

bdhi->forwardTime();

The function getSelfMobility() will return the self mobility of the particles.
The function getHydrodynamicRadius() will return the used hydrodynamic radius. You should expect a variation of +- 1% between this value and the one you measure from simulations.
---------------

A brief summary of the algorithm:

eq. 36 in [1] can be rewritten as:

\tilde{\vec{v}}^{n+1} = \mathcal{L}^{-1}·\vec{g} -> fluid velocity

Where:

\mathcal{L}^{-1} = (\rho/dt·\bf{I} - \eta/2 L)^{-1} · P
\vec{g} = (\rho/dt\bf{I} + \eta/2 L)\vec{v}^n + DW^n + SF^{n+1/2} - D(\rho \vec{v}\vec{v}^T)^n+1/2 -> fluid forcing
                                                 ^ noise                  ^ advection
P = \bf{I} - G(DG)^-1 D -> projection to divergence free space

D: Divergence operator
G: Gradient operator
L: Laplacian operator
S: Spreading operator (transmits particle properties to fluid)
W: Symmetric noise tensor, Gaussian numbers with mean 0 and std 1

On the other hand the particles and fluid are coupled with:
\vec{u}^n = J·\vec{v}
Where J is the interpolation operator defined as J^T = dV·S
Where dV is the volume of a grid cell.

The particles are updated following the mid point (predictor-corrector) scheme developed in [1].

About J and S:

S_cellj = sum_i=0^N{\delta(ri-r_cellj)} ->particles to fluid
J_i = dV·sum_j=0^ncells{\delta(ri-r_cellj)}->fluid to particles

Where \delta is an spreading kernel (smeared delta).
The default kernel used is the 3-point Peskin kernel, see IBM_kernels.cuh.
But others can be selected with arbitrary support to better describe or satisfy certain conditions.

The thermal drift is computed using random finite diferences as explained in [3].

REFERENCES:

[1] Inertial coupling method for particles in an incompressible fluctuating fluid. Florencio Balboa, Rafael Delgado-Buscalioni, Boyce E. Griffith and Aleksandar Donev. 2014 https://doi.org/10.1016/j.cma.2013.10.029.
[2] Staggered Schemes for Fluctuating Hydrodynamics. Florencio Balboa, et.al. 2012.
[3] Brownian Dynamics without Green's Functions. Steve Delong, Florencio Balboa, et. al. 2014

 */

#ifndef INTEGRATORHYDROICM_CUH
#define INTEGRATORHYDROICM_CUH
#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"global/defines.h"
#include"utils/cufftComplex3.cuh"
#include"Integrator/Integrator.cuh"
#include <stdexcept>
#include<thrust/device_vector.h>
#include "utils/utils.h"
#include"utils/cufftPrecisionAgnostic.h"
#include"utils/Grid.cuh"
#include<curand.h>
#include"misc/IBM_kernels.cuh"

namespace uammd{
  namespace Hydro{

    class ICM: public Integrator{
    public:
      struct Parameters{
	real temperature = 0;
	real viscosity = -1;
	real density = -1;
	real hydrodynamicRadius = -1;
	real dt;
	Box box;
	int3 cells={-1, -1, -1}; //Default is compute the closest number of cells that is a FFT friendly number
	bool sumThermalDrift = false; //Thermal drift has a neglegible contribution in ICM
	bool removeTotalMomemtum = true; //Set the total fluid momentum to zero in each step
      };

      ICM(shared_ptr<ParticleData> pd,
	  shared_ptr<System> sys,
	  Parameters par);

      ~ICM();

      void forwardTime() override;
      real sumEnergy() override{return 0;}

      using cufftComplex3 = cufftComplex3_t<real>;
      using cufftComplex = cufftComplex_t<real>;
      using cufftReal = cufftReal_t<real>;

      real getSelfMobility(){
	double rh = getHydrodynamicRadius();
	return 1.0/(6*M_PI*viscosity*rh)*(1-2.837297*rh/box.boxSize.x);
      }
      real getHydrodynamicRadius(){
	return 0.91*box.boxSize.x/(real)grid.cellDim.x;
      }

      const real3* getFluidVelocities(access::location dev){
	real3* ptr = nullptr;
	switch(dev){
	case access::gpu:
	  ptr = thrust::raw_pointer_cast(gridVels.data());
	  break;
	case access::cpu:
	  h_gridVels = gridVels;
	  ptr = thrust::raw_pointer_cast(h_gridVels.data());
	  break;
	default:
	  System::log<System::EXCEPTION>("Invalid device in ICM::getFluidVelocities");
	  throw std::runtime_error("Invalid device");
	}
	return ptr;
      }

      int3 getNumberFluidCells(){
	return grid.cellDim;
      }
      
    private:
      using Kernel = IBM_kernels::Peskin::threePoint;
      real temperature, viscosity, density;
      bool sumThermalDrift;
      bool removeTotalMomemtum;
      Grid grid;
      Box box;

      cudaStream_t st;
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea; //Work space for cufft
      //Grid forces/velocities in fourier/real space
      thrust::device_vector<real3> gridVels;
      thrust::device_vector<cufftComplex> gridVelsPredictionF;
      //Host version of grid velocities for getFluidVelocities
      thrust::host_vector<real3> h_gridVels;
      thrust::device_vector<real3> cellAdvection;
      curandGenerator_t curng;
      thrust::device_vector<real> random;
      real deltaRFD;
      real dt;

      thrust::device_vector<real4> posOld; //q^n

      uint step = 0;

      uint seed = 1234;

      void initializeGrid(Parameters par);
      void printMessages(Parameters par);
      void initCuFFT();
      void initFluid();
      void initCuRAND();
      void resizeContainers();
      void spreadParticleForces();
      void thermalDrift();
      void unperturbedFluidForcing();

      void applyStokesSolutionOperator();

      void predictorStep();
      void correctorStep();

    };

  }
}
#include"ICM.cu"
#endif

