
/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
  takes a thrust::Tuple containing positions, velocities and forces on each particle. 

  TODO:
  80- Make Diffusion into a virtual class when implementing FMM. It is better than a class with switchs...
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

#define TPB 512
namespace brownian_hy_euler_maruyama_ns{
  __constant__ Params params;
  __constant__ RPYParams RPYparams;


  void initGPU(Params m_params){
    m_params.sqrt2dt = sqrt(2.0)*m_params.sqrtdt;    
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }

  void initRPYGPU(RPYParams m_params){
    m_params.inv32rhtimes3   = 3.0/(32.0*m_params.rh);
    m_params.rhrh2div3 = m_params.rh*m_params.rh*2.0/3.0;
    gpuErrchk(cudaMemcpyToSymbol(RPYparams, &m_params, sizeof(RPYParams)));
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

    pos[i] += make_real4( KR + DF[i]*params.dt + params.sqrt2dt*BdW[i]);
  }

  //Update the positions
  void integrateGPU(real4 *pos,
		    real3* DF, real3* BdW,
		    real3* K, uint N){
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0); 
    integrateGPUD<<<nblocks, nthreads>>>(pos, DF, BdW, K);    
  }

  /*RPY tensor as a function of distance, r*/
  /*This is a critical function and is insanely optimized to perform the least FLOPS possible*/
  inline __device__  real2  RPY(const real &r){
    const real invr  = real(1.0)/r;
    
    real2 c12;
    /*c12.y = c2 -> c2*invr2*/
    if(r > real(2.0)*RPYparams.rh){
      const real A = real(0.75)*RPYparams.rh*invr;
      const real invr2 = invr*invr;
      
      c12.x = A*(real(1.0) + RPYparams.rhrh2div3*invr2);
      
      c12.y = A*invr2*(real(1.0) - real(3.0)*RPYparams.rhrh2div3*invr2);      
    }
    else{      
      c12.x = real(1.0) - real(3.0)*r*RPYparams.inv32rhtimes3;
      c12.y = RPYparams.inv32rhtimes3*invr;
    }
    
    return c12;
  }

  __global__ void computeDiffusionRPYD(real *D, const __restrict__ real4 *R, uint N){
    int i_id = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = 3*N;
    for(int id = i_id; id<n*n; id += blockDim.x*gridDim.x){
      /*Compute one pair per thread in the gride-stride loop*/
	/*Get the pair*/
	int j = id/N;
	int i = id%N;
  
	/*Self Diffusion*/
	real D0 = RPYparams.D0;
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
	
	
	real3 rij;
	real *rijp = &(rij.x);    
	real c1, c2;
	
	rij = make_real3(R[j]) - make_real3(R[i]);
	real r = sqrt(dot(rij, rij));
	
	/*Rotne-Prager-Yamakawa tensor */
	real2 c12 = RPY(r);
	c1 = c12.x;
	c2 = c12.y;
	/*Oseen tensor*/
	// c1 = 0.75f*invr*rh;
	// c2 = c1;

	/*3x3 Matrix for each particle pair*/
	for(int k = 0; k < 3; k++)
	  for(int l = 0; l < 3; l++)
	    D[3*i + k + n*(3*j + l)] = D0*c2*rijp[k]*rijp[l];
	/*Diagonal*/
	for(int k = 0; k<3; k++)  D[3*i + k + n*(3*j + k)] += D0*c1;

      }
    }

  /*Fill the 3Nx3N matrix D using the RPY tensor, the diagonal boxes remain as D0*/
  void computeDiffusionRPYGPU(real *d_D, real4 *d_R, cudaStream_t stream, uint N){
    computeDiffusionRPYD<<<N*N/128/2+1, 128, 0 ,stream>>>(d_D, d_R, N);
  }

  /*Compute the product Dv = D·v, computing D on the fly when needed, without storing it*/
  __global__ void diffusionDotGPUD(const __restrict__ real4  *pos,
				   const __restrict__ real3 *v,
				   __restrict__ real3 *Dv,
				   int numTiles, /*Thread paralellism level, 
						   controls how many elements are stored in 
						   shared memory and
						   computed in parallel between synchronizations*/
				   uint N){
    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    uint id = blockIdx.x*blockDim.x+threadIdx.x;
    /*Each thread handles one particle with the other N, including itself*/
    /*That is 3 full lines of D, or 3 elements of D·v per thread, being the x y z of ij with j=0:N-1*/
    /*In other words. D is made of NxN boxes of size 3x3, defining the x,y,z mobility between particle pairs, each thread handles a row of boxes and multiplies it by three elements of v*/
    extern __shared__ real shMem[];
    /*Store a group of positions and elements of v in shared memory,
      for all threads will be accesing them in parallel*/
    real4 *shPos = (real4*)&shMem[0];
    real3 *shV   = (real3*)&shMem[4*TPB];
    /*TODO: shV being real3 probably causes a bad accessing pattern, 
      not much can be done about it though*/
    
    const real3 pi = make_real3(pos[id]); /*My position*/
    real3 Dv_t = make_real3(real(0.0)); /*The three elements of the result D·v I compute*/
    
    real3 vj; /*Take the three elements in v, x y z in the current particle*/
    
    /*Distribute the N particles in numTiles tiles.
      Storing in each tile blockDim.x positions and elements of v in shared memory*/
    /*This way all threads are accesing the same memory addresses at the same time*/
    for(int tile = 0; tile<numTiles; tile++){
      //real3 Dijtemp;
      /*Save this tile pos and v to shared memory*/
      shPos[threadIdx.x] = pos[tile*blockDim.x+threadIdx.x];
      shV[threadIdx.x]   = v[tile*blockDim.x+threadIdx.x];
      __syncthreads();
      /*Go through all the particles in the current tile*/
      #pragma unroll 8
      for(uint counter = 0; counter<blockDim.x; counter++){
	vj = shV[counter]; /*Take the three elements in v, x y z*/
	if(id==tile*blockDim.x+counter){/*Diagonal boxes, i==j*/
	  /*This assumes that the self diffusion is just D0*I*/
	  Dv_t += vj;
	}
	else{ /*Any other box, i!=j*/	  
	  
	  const real3 rij = pi-make_real3(shPos[counter]); /*Take the ri-rj*/

	  const real  r   = sqrtf(dot(rij, rij));  /*Distance between ri and rj*/	  
	  
	  const real2 c12 = RPY(r); /*Compute the Rotne Prager Yamakawa coefficients*/


	  const real c1 = c12.x;
	  const real c2 = c12.y;
	  /*Precompute this two quantities*/
	  const real3 c2rij = c2*rij;
	  //real d0c1 = RPYparams.D0*c1;
	  
	  /*Update the result with Dij·vj, the current box dot the current three elements of v*/
	  //[51 FLOPS]
	   Dv_t.x += dot( c2rij*rij.x + make_real3(c1, 0, 0), vj);
	   Dv_t.y += dot( c2rij*rij.y + make_real3(0, c1, 0), vj);
	   Dv_t.z += dot( c2rij*rij.z + make_real3(0, 0, c1), vj);
	   // Dijtemp    = c2rij*rij.x;
	   // Dijtemp.x += c1;
	   // Dv_t.x += dot( Dijtemp, vj);
	  
	   // Dijtemp    = c2rij*rij.y;
	   // Dijtemp.y += c1;
	   // Dv_t.y += dot( Dijtemp, vj);

	   // Dijtemp    = c2rij*rij.z;
	   // Dijtemp.z += c1;
	   // Dv_t.z += dot( Dijtemp, vj);
	  // const real av = dot(c2rij, vj);	  

	  // Dv_t.x += av + c1*vj.x + (rij.x-real(1.0))*vj.x*c2rij.x;
	  // Dv_t.y += av + c1*vj.y + (rij.y-real(1.0))*vj.y*c2rij.y;
	  // Dv_t.z += av + c1*vj.z + (rij.z-real(1.0))*vj.z*c2rij.z;
	  
	}
      }/*End of particles in tile loop*/
      __syncthreads();    

    }/*End of tile loop*/
    /*Write the result to global memory*/
    Dv[id] = RPYparams.D0*Dv_t;
  }
  
  /*GPU kernel caller*/
  /*Compute the product D·v, computing D on the fly when needed, without storing it*/
  void diffusionDotGPU(real4 *pos, real3 *v, real3 *Dv, uint N){
    int Nblocks  = (N+TPB-1)/TPB;
    int numtiles = (N + TPB-1)/TPB;
    
    diffusionDotGPUD<<<Nblocks, TPB, TPB*(sizeof(real4)+sizeof(real3))>>>(pos, v, Dv, numtiles, N);
    
  }
  
  
}