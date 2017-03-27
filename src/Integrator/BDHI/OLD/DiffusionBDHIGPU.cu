/*Raul P. Pelaez 2016. Kernels to compute the mobility tensor and the product M·v -> any vector
  Using the Rodne-Prager-Yamakawa tensor.
*/ 
#include "globals/defines.h"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"DiffusionBDHIGPU.cuh"

#define TPB 1024

namespace brownian_hy_euler_maruyama_ns{
  __constant__ RPYParams RPYparams;


  void initRPYGPU(RPYParams m_params){
    m_params.inv32rhtimes3   = 3.0/(32.0*m_params.rh);
    m_params.rhrh2div3 = m_params.rh*m_params.rh*2.0/3.0;
    gpuErrchk(cudaMemcpyToSymbol(RPYparams, &m_params, sizeof(RPYParams)));
  }


  /*RPY tensor as a function of distance, r*/
  /*M(r) = 0.75*M0*( f(r)*I + g(r)*r(diadic)r )*/
  /*c12.x = f(r) * 0.75*M0
    c12.y = g(r) * 0.75*M0*/
  /*This is a critical function and is insanely optimized to perform the least FLOPS possible*/
  inline __device__  real2  RPY(const real &r){
    
    const real invr  = real(1.0)/r;
    
    real2 c12 = {0,0};

    /*Oseen tensor*/
    //return RPYparams.rh*make_real2(real(0.75)*invr, real(0.75)*invr*invr*invr);
    const real rh = RPYparams.rh;
    /*c12.y = c2 -> c2*invr2*/
    if(r >= real(2.0)*rh){
      const real A = real(0.75)*rh*invr;
      const real invr2 = invr*invr;

      // c12.x = real(0.75)*(invr+(2.0f/3.0f)*rh*rh*invr2*invr)*rh;
      // c12.y = real(0.75)*(invr-2.0f*invr2*invr*rh*rh)*rh;
      
       c12.x = A*(real(1.0) + RPYparams.rhrh2div3*invr2);      
       c12.y = A*invr2*(real(1.0) - real(3.0)*RPYparams.rhrh2div3*invr2);      
    }
    else{
       c12.x = 1.0f-(9.0f/32.0f)*r/rh;
       if(r>real(0.0))
       	c12.y = (3.0f/32.0f)*invr/rh;      
      // c12.x = real(1.0) - real(3.0)*r*RPYparams.inv32rhtimes3;
      // if(r>real(0.0))
      // 	c12.y = RPYparams.inv32rhtimes3*invr;
      
    }
    
    return c12;
  }

  /*Helper function for divergence mode in diffusionDot, 
    computes {f(r+dw)-f(r), g(r+dw)-g(r)}
    See diffusionDot for more info
   */
  inline __device__ real2  RPYDivergence(real3 rij, real3 dwij){

    const real r    = sqrtf(dot(rij, rij));
    const real3 rpdwij = rij+dwij;
    const real rpdw = sqrtf(dot(rpdwij, rpdwij));
    
    return RPY(rpdw)-RPY(r);

  }

  /*Fills the 3Nx3N Mobility matrix with Mij = RPY(rij) in blocks of 3x3 matrices*/
  __global__ void computeDiffusionRPYD(real *D, const __restrict__ real4 *R, uint N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;
    uint n = 3*N;
    real D0 = RPYparams.D0;

    /*Self Diffusion*/
    
    for(int k = 0; k < 3; k++)
      for(int l = 0; l < 3; l++){
	D[3*i + k + n*(3*i + l)] =  0.0f;
      }
    D[3*i + 0 + n*(3*i + 0)] = D0;
    D[3*i + 1 + n*(3*i + 1)] = D0;
    D[3*i + 2 + n*(3*i + 2)] = D0;
       

    real3 rij;
    real* rijp = &(rij.x);    
    real c1, c2;
    for(int j=i+1; j<N; j++){            	
	
      rij = make_real3(R[j]) - make_real3(R[i]);
      const real r = sqrt(dot(rij, rij));
	
      /*Rotne-Prager-Yamakawa tensor */
      const real2 c12 = RPY(r);
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
    int GPU_Nthreads = TPB<N?TPB:N;
    int GPU_Nblocks  =  N/GPU_Nthreads +  ((N%GPU_Nthreads!=0)?1:0); 
    
    computeDiffusionRPYD<<<GPU_Nblocks, GPU_Nthreads, 0 ,stream>>>(d_D, d_R, N);
  }

  
  /*Compute the product Dv = D·v, computing D on the fly when needed, without storing it*/
  /*This critital kernel is the 99% of the execution time in a BDHI simulation*/
  
  /*DIVERGENCE MODE:
    -Divergence mode computes divM = (M(q+dw) - M(q))·dw/d in a single pass.
  */

  /*TODO: Template this the same as transverseList.
    This is like MatrixVector product, but the matrix has a generating function f(i,j,...)
  General minimum info: pos, v
  Restricted to result as real3(N)?
  Maybe general like MVdot(T transverser, int numTiles, uint N) and transverser holds pos, v and Dv.
  Probably real3 is ok.

  marked with / *! 
  */
  
  template<bool divergence_mode>
  __global__ void diffusionDotGPUD(const __restrict__ real4  *pos,
				   const __restrict__ real3 *v, /*This is dw in divergence mode*/
				   __restrict__ real3 *Dv,
				   int numTiles, /*Thread paralellism level, 
						   controls how many elements are stored in 
						   shared memory and
						   computed in parallel between synchronizations*/
				   uint N){
    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    uint id = blockIdx.x*blockDim.x+threadIdx.x;
    bool active = true;
    if(id >= N) active = false;
    /*Each thread handles one particle with the other N, including itself*/
    /*That is 3 full lines of D, or 3 elements of D·v per thread, being the x y z of ij with j=0:N-1*/
    /*In other words. D is made of NxN boxes of size 3x3, defining the x,y,z mobility between particle pairs, each thread handles a row of boxes and multiplies it by three elements of v*/
    extern __shared__ real shMem[];
    /*Store a group of positions and elements of v in shared memory,
      for all threads will be accesing them in parallel*/
    real3 *shPos = (real3*)&shMem[0];
    real3 *shV   = (real3*)&shMem[3*TPB];
    /*TODO: shV being real3 probably causes a bad accessing pattern, 
      not much can be done about it though*/

    real3 pi;
    if(active)
      pi = make_real3(pos[id]); /*My position*/

    real3 vi;
    if(divergence_mode)
      if(active)      
	vi = v[id];

    real3 Dv_t = {0,0,0}; /*The three elements of the result D·v I compute*/

    real3 vj; /*Take the three elements in v, x y z in the current particle*/
    
    /*Distribute the N particles in numTiles tiles.
      Storing in each tile blockDim.x positions and elements of v in shared memory*/
    /*This way all threads are accesing the same memory addresses at the same time*/
    for(int tile = 0; tile<numTiles; tile++){
      //real3 Dijtemp;
      /*Save this tile pos and v to shared memory*/
      if(active && tile*blockDim.x+threadIdx.x < N){
	shPos[threadIdx.x] = make_real3(pos[tile*blockDim.x+threadIdx.x]);
	shV[threadIdx.x]   = v[tile*blockDim.x+threadIdx.x];
      }
      else{
	shPos[threadIdx.x] = make_real3(0);
	shV[threadIdx.x] = make_real3(0);
      }
      __syncthreads();
      if(active){
	/*Go through all the particles in the current tile*/
#pragma unroll 8
	for(int counter = 0; counter<blockDim.x; counter++){
	  if(tile*blockDim.x+counter >= N) break;
	  vj = shV[counter]; /*Take the three elements in v, x y z*/	 
	  if(id==tile*blockDim.x+counter){/*Diagonal boxes, i==j*/
	    /*This assumes that the self diffusion is just D0*I*/
	    if(!divergence_mode)
	      Dv_t += vj; //make_real3(vj.x, vj.y, 0);
	  }
	  else{ /*Any other box, i!=j*/	  
	  
	    const real3 rij = pi-shPos[counter]; /*Take the ri-rj*/
	    
	    real2 c12;
	    if(divergence_mode){
	      /*
		M(q+dw)-M(q) =  [f(r+dw)-f(r)]·I + [g(r+dw)-g(r)]·r(diadic)r = 
		 = f_eff·I + g_eff·r(diadic)r
		 So the rest of the function remains the same as in normal diffusion dot mode.
	      */
	      /*r+dw = ri+dwi - (rj+dwj) = ri-rj + dwi-dwj = rij + dwij*/
	      const real3 vij = vi - vj; /* dwij */
	      c12 = RPYDivergence(rij, vij);
	    }
	    else{
	      const real  r   = sqrtf(dot(rij, rij));  /*Distance between ri and rj*/	  
	      c12 = RPY(r); /*Compute the Rotne Prager Yamakawa coefficients*/
	    }
	    const real f = c12.x;
	    const real g = c12.y;

	    /*Update the result with Dij·vj, the current box dot the current three elements of v*/
	    /*This expression is a little obfuscated, Dij*vj*/
	    /*
	      M = f(r)*I+g(r)*r(diadic)r - > (M·v)_ß = f(r)·v_ß + g(r)·v·(r(diadic)r)
	     */
	    const real gv = dot(g*rij, vj);
	    /*gv = g(r)·( vx·rx + vy·ry + vz·rz )*/
	    /*(g(r)·v·(r(diadic)r) )_ß = gv·r_ß*/
	    Dv_t.x += f*vj.x + gv*rij.x;
	    Dv_t.y += f*vj.y + gv*rij.y;
	    Dv_t.z += f*vj.z + gv*rij.z;
	  }
	}/*End of particles in tile loop*/
      }
      __syncthreads();    

    }/*End of tile loop*/
    /*Write the result to global memory*/
    if(active)
      Dv[id] = RPYparams.D0*Dv_t;
    
  }

  /*GPU kernel caller*/
  /*Compute the product D·v, computing D on the fly when needed, without storing it*/
  void diffusionDotGPU(real4 *pos, real3 *v, real3 *Dv, uint N, cudaStream_t st, bool divergence_mode){
    int Nblocks  = (N+TPB-1)/TPB;
    int numtiles = (N + TPB-1)/TPB;
    
    if(!divergence_mode)
      diffusionDotGPUD<false><<<Nblocks, TPB, TPB*(sizeof(real3)+sizeof(real3)), st>>>(pos, v, Dv, numtiles, N);
    else
      diffusionDotGPUD<true><<<Nblocks, TPB, TPB*(sizeof(real3)+sizeof(real3)), st>>>(pos, v, Dv, numtiles, N); 
  }

  /*Analytic divergence term*/
  inline __device__ real3 divij(const real4 &posi, const real4 &posj){
    real3 r12 = make_real3(posi)-make_real3(posj);
    real r2 = dot(r12, r12);
    if(r2==real(0.0))
      return make_real3(0.0);
    real invr = rsqrtf(r2);
    if(r2>real(1.0)){
      r2 *= real(4.0);
      return make_real3(real(0.75)*real(2.0)*(r2-real(2.0))/(r2*r2)*r12*invr);
    }
    else{
      //return make_real3(0.0);
      return make_real3(real(0.09375)*real(2.0)*r12*invr);
    }
    
    
  }


  /*Nbody kernel adapted to compute the analytic divergence term in 2D*/
  __global__ void NBodyGPUD(const __restrict__ real4  *pos,
				 __restrict__ real3 *force,
				 int numTiles, /*Thread paralellism level, 
						 controls how many elements are stored in 
						   shared memory and
						   computed in parallel between synchronizations*/
				   uint N){
    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    uint id = blockIdx.x*blockDim.x+threadIdx.x;
    bool active = true;
    if(id>=N) active = false;
    /*Each thread handles the interaction between particle id and all the others*/
    /*Storing blockDim.x positions in shared memory and processing all of them in parallel*/
    extern __shared__ real4 shPos[];
    
    real4 pi;
    if(active)
      pi = pos[id]; /*My position*/
    
    real3 f = make_real3(real(0.0)); /*The three elements of the result D·v I compute*/

    /*Distribute the N particles in numTiles tiles.
      Storing in each tile blockDim.x positions and elements of v in shared memory*/
    /*This way all threads are accesing the same memory addresses at the same time*/
    for(int tile = 0; tile<numTiles; tile++){
      /*Save this tile pos to shared memory*/
      if(active)
	shPos[threadIdx.x] = pos[tile*blockDim.x+threadIdx.x];
      __syncthreads();
      /*Go through all the particles in the current tile*/
      #pragma unroll 8
      for(uint counter = 0; counter<blockDim.x; counter++){
	if(!active) break;
	if(id != tile*blockDim.x+counter && (tile*blockDim.x+counter)<N)
	  f += divij(pi,shPos[counter]);
      }/*End of particles in tile loop*/
      __syncthreads();

    }/*End of tile loop*/
    /*Write the result to global memory*/
    if(active)
      force[id] = RPYparams.D0*f;
  }

  void divergenceGPU(real4 *pos, real3 *divM, uint N){

    int Nblocks  = (N+TPB-1)/TPB;
    int numtiles = (N + TPB-1)/TPB;

    NBodyGPUD<<<Nblocks, TPB, TPB*sizeof(real4)>>>(pos, divM, numtiles, N);
  }

  
  

}