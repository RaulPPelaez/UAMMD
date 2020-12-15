/*Raul P. Pelaez 2017-2020. Positively Split Edwald Rotne-Prager-Yamakawa Brownian Dynamics with Hydrodynamic interactions.

Utils

*/

#ifndef BDHI_PSE_UTILS_CUH
#define BDHI_PSE_UTILS_CUH 
#include "global/defines.h"
#include "utils/vector.cuh"
#include "Integrator/BDHI/BDHI.cuh"
namespace uammd{
  namespace BDHI{
    namespace pse_ns{

      struct Parameters: BDHI::Parameters{
	//Splitting parameter, works best between 0.5 and 1.0
	//lower values will give more importance to the near part (neighbour list) and higher values will
	// put the weight of the computation in the far part (FFT).
	real psi = 0.5;
	real shearStrain = 0;
      };

      __device__ int3 indexToWaveNumber(int i, int3 nk){
	int ikx = i%(nk.x/2+1);
	int iky = (i/(nk.x/2+1))%nk.y;
	int ikz = i/((nk.x/2+1)*nk.y);
	ikx -= nk.x*(ikx >= (nk.x/2+1));
	iky -= nk.y*(iky >= (nk.y/2+1));
	ikz -= nk.z*(ikz >= (nk.z/2+1));
	return make_int3(ikx, iky, ikz);
      }

      __device__ real3 waveNumberToWaveVector(int3 ik, real3 L, real shearStrain){
	auto kvec = (real(2.0)*real(M_PI)/L)*make_real3(ik.x, ik.y, ik.z);
	kvec.y -= shearStrain*kvec.x;
	return kvec;
      }
    }
  }
}      
#endif
 
