/* Raul P. Pelaez 2018. Fluctuating Immerse Boundary (FIB) method.



References:
[1] Brownian dynamics without Green's functions. Delong et al. https://aip.scitation.org/doi/pdf/10.1063/1.4869866?class=pdf


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
#include<utils/cufftPrecisionAgnostic.h>
#include"utils/Grid.cuh"


namespace uammd{
  namespace BDHI{

    class FIB: public Integrator{
    public:
      struct Parameters{
	real temperature;
	real viscosity;
	real hydrodynamicRadius;
	real dt;
	Box box;
	int3 cells={-1, -1, -1}; //Default is compute the closest number of cells that is a FFT friendly number
      };

      FIB(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,		       
	  Parameters par);
      FIB(shared_ptr<ParticleData> pd,
	  shared_ptr<System> sys,		       
	  Parameters par):
	FIB(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), par){}
      				     
      ~FIB();

      void forwardTime() override;
      real sumEnergy() override;

    private:
      using cufftComplex = cufftComplex_t<real>;
      using cufftReal = cufftReal_t<real>;
      real temperature;
      real dt;
      real M0;

      shared_ptr<CellList> cl;
      
      Grid grid; /*Wave space Grid parameters*/
      
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<real> cufftWorkArea; //Work space for cufft
      
      thrust::device_vector<cufftComplex> gridVelsFourier; //Interpolated grid forces/velocities in fourier space

      thrust::device_vector<real> random;
      //Temporal integration variables
      thrust::device_vector<real4> posPrediction;
      real deltaRFD; //Random finite diference step
      ullint step = 0;

      
      /*A convenient struct to pack 3 complex numbers, that is 6 real numbers*/
      struct cufftComplex3{
	cufftComplex x,y,z;
      };

      inline __device__ __host__ cufftComplex3 operator+(const cufftComplex3 &a, const cufftComplex3 &b){
	return {a.x + b.x, a.y + b.y, a.z + b.z};
      }
      inline __device__ __host__ void operator+=(cufftComplex3 &a, const cufftComplex3 &b){
	a.x += b.x; a.y += b.y; a.z += b.z;
      }

      
    };

  }
}
#endif