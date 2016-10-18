
/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
  takes a thrust::Tuple containing positions, velocities and forces on each particle. 

  TODO:
  100-Optimize, see .cpp
*/
#include "globals/defines.h"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>




using namespace thrust;

#define TPB 128
namespace brownian_hy_euler_maruyama_ns{
  __constant__ Params params;


  void initGPU(Params m_params){
    m_params.sqrt2dt = sqrt(2.0f)*m_params.sqrtdt;
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }


  __global__ void real4_to_real3D(const real4 __restrict__ *v4, real3 __restrict__ *v3, uint N){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    v3[i] = make_real3(v4[i]);
  
  }

  void real4_to_real3GPU(real4 *v4, real3* v3, uint N){
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0); 
    
    real4_to_real3D<<<nthreads,nblocks>>>(v4, v3, N);
  }


  __global__ void integrateGPUD(real4 __restrict__  *pos,
				const real3 __restrict__  *DF,
				const real3 __restrict__ *BdW,
				const real3 __restrict__ *K){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=params.N) return;
    /*Half step velocity*/
    real3 p = make_real3(pos[i]);
    real3 KR;
    KR.x = params.dt*dot(K[0], p);
    KR.y = params.dt*dot(K[1], p);
    KR.z = params.dt*dot(K[2], p);

    pos[i] += make_real4( KR + DF[i] + params.sqrt2dt*BdW[i]);
  }

  //Update the positions
  void integrateGPU(real4 *pos,
		    real3* DF, real3* BdW,
		    real3* K, uint N){
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0); 
    integrateGPUD<<<nblocks, nthreads>>>(pos, DF, BdW, K);    
  }







  __global__ void rotneGPU_prev(real *D, real4 *R, uint N){
    int i_id = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = 3*N;
    for(int id = i_id; id<n*n; id += blockDim.x*gridDim.x){
      /*Compute one pair per thread in the gride-stride loop*/
	/*Get the pair*/
	int j = id/N;
	int i = id%N;
  
	/*Fix the Diagonal boxes of D*/
	real D0 = params.D0;
	if(i >= N || j>=N ||  j<i) continue;
	else if(j==i){
	  for(int k = 0; k < 3; k++)
	    for(int l = 0; l < 3; l++){
	      D[3*i + k + n*(3*i + l)] =  0.0f;
	    }
	  D[3*i + 0 + n*(3*i + 0)] = D0;
	  D[3*i + 1 + n*(3*i + 1)] = D0;
	  D[3*i + 2 + n*(3*i + 2)] = D0;
	  continue;
	}
	real rh = params.rh;

  
	real4 rij;
	real *rijp = &(rij.x);
    

	real c1, c2;
  
	rij = R[j] - R[i];
	rij.w = real(0.0);
  
	real invr2 = real(1.0)/dot(rij, rij);
	real invr = sqrt(invr2);

   
	/*Rotne-Prager-Yamakawa tensor */
	real r = real(1.0)/invr;
	if(r >= real(2.0)*rh){
	  c1 = real(0.75)*rh*invr*(real(1.0) + real(2.0)*invr2*rh*rh/real(3.0));
	  c2 = real(0.75)*rh*invr*(real(1.0) - real(2.0)*rh*rh*invr2);
	}
	else{
	  c1 = real(1.0) - real(9.0)*r/(real(32.0)*rh);
	  c2 = real(3.0)*r/(real(32.0)*rh);
	}

	/*Oseen tensor*/
	// c1 = 0.75f*invr*rh;
	// c2 = c1;

	for(int k = 0; k < 3; k++)
	  for(int l = 0; l < 3; l++)
	    D[3*i + k + n*(3*j + l)] = D0*c2*rijp[k]*rijp[l]*invr2;

	for(int k = 0; k<3; k++)  D[3*i + k + n*(3*j + k)] += D0*c1;

      }
    }


    void rodne_callGPU(real *d_D, real4 *d_R, cudaStream_t stream, uint N){
      rotneGPU_prev<<<N*N/128/2+1, 128, 0 ,stream>>>(d_D, d_R, N);
    }








    // __global__ void fix_B(real *B, uint n){
  
    //   uint ii = blockIdx.x*blockDim.x + threadIdx.x;
    //   if(ii >= n*n) return;
    //   uint i = ii%n;
    //   uint j = ii/n;
    //   if(i<=j) return;  
    //   B[ii] = 0.0f;
    // }


    // void fix_B_call(real *B, uint N, cudaStream_t stream){
    //   fix_B<<<(3*N*3*N)/128+1,128,0, stream>>>(B, 3*N);
    // }







    // __global__ void rotneGPU(real *D, real3 *R2, uint N){
    //   int id = blockIdx.x*blockDim.x + threadIdx.x;
  
    //   int j = id/N;
    //   int i = id%N;
    //   real D0 = 1.0f;
    //     uint n = 3*N;
    //   if(i >= N || j>=N) return;
    //   else if(j==i){
    //     for(int k = 0; k < 3; k++)
    //       for(int l = 0; l < 3; l++){
    // 	D[3*i + k + n*(3*i + l)] =  k==l?D0:0.0f;
    //       }
    //     return;
    //   }
    //   real rh = 1;
    //   real *R = (real*)R2;

    //   real rij[3];
    //   real r2 = 0.0f;
    //   real r;
    //   real c1, c2;
    //   for(int k = 0; k<3; k++){
    //     rij[k] = R[3*j + k] - R[3*i+k];
    //     r2 += rij[k]*rij[k];
    //   }
    //   r = sqrt(r2);
    //   if(r>=2*rh){
    //     c1 = 0.75*rh/r*(1.0f + 2.0f*rh*rh/(3.0f*r2));
    //     c2 = 0.75*rh/r*(1.0f - 2.0f*rh*rh/r2);
    //   }
    //   else{
    //     c1 = 1.0f - 9.0f*r/(32.0f*rh);
    //     c2 = 3.0f*r/(32.0f*rh);
    //   }

    //   for(int k = 0; k < 3; k++)
    //     for(int l = 0; l < 3; l++)
    //       D[3*i + k + n*(3*j + l)] = D0*c1*(k==l?1.0f:0.0f) + c2*rij[k]*rij[l]/r2;

    // }

  }
