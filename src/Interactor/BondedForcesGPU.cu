/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor GPU implementation.
    Currently Implemented: 
       1-Two body harmonic springs
       2-Three body angle springs

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  This module implements an algorithm to compute the force between particles joined by springs.

TODO:
100- Set TPB to mean number of bonds per particle with bond
100- Grid stride computeBondedForceD
100- Implement FP without thrust
*/

#include"BondedForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include<thrust/device_ptr.h>
#include<thrust/reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>

using namespace thrust;
namespace bonded_forces_ns{
  //Parameters in constant memory, super fast access
  __constant__ Params params; 

  void initGPU(Params m_params){
    m_params.invL = 1.0f/m_params.L;
    /*Upload parameters to constant memory*/
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }


  //MIC algorithm
  template<typename vecType>
  inline __device__ void apply_pbc(vecType &r){
    float3 r3 = make_float3(r.x, r.y, r.z);
    float3 shift = (floorf(r3*params.invL+0.5f)*params.L); //MIC Algorithm
    r.x -= shift.x;
    r.y -= shift.y;
    r.z -= shift.z;
  }


#define TPB 64

  //Custom kernel to compute and sum the force
  __global__ void computeBondedForceD(float4* __restrict__ force, const float4* __restrict__ pos,
				      const uint* __restrict__ bondStart,
				      const uint* __restrict__ bondEnd,
				      const uint* __restrict__ bondedParticleIndex,
				      const Bond* __restrict__ bondList){
    __shared__ float4 forceTotal[TPB]; /*Each thread
    /*A block per particle*/
    /*Instead of launching a thread per particle and discarding those without any bond,
      I store an additional array of size N_particles_with_bonds that contains the indices
      of the particles that are involved in at least one bond. And only launch N_particles_with_bonds blocks*/
    /*Each thread in a block computes the force on particle p due to one (or several) bonds*/
    uint p = bondedParticleIndex[blockIdx.x]; //This block handles particle p
    float4 posi = pos[p]; //Get position

    /*First and last bond indices of p in bondList*/
    uint first = bondStart[p]; 
    uint last = bondEnd[p];
   
    float4 f = make_float4(0.0f);

    Bond bond; //The current bond
    
    int j; float4 posj; //The other particle
    float r0, k; //The bond info
      
    float3 r12;

    
    for(int b = first+threadIdx.x; b<last; b+=blockDim.x){
      /*Read bond info*/
      bond = bondList[b];
      j = bond.j;
      r0 = bond.r0;
      k = bond.k;
      /*Bring pos of other particle*/
      posj = pos[j];
    
      /*Compute force*/
      r12 =  make_float3(posi-posj);
      apply_pbc(r12);
    
      float invr = rsqrt(dot(r12, r12));
    
      float fmod = -k*(1.0f-r0*invr); //F = -k·(r-r0)·rvec/r
      
      f += make_float4(fmod*r12);
    }

    /*The first thread sums all the contributions*/
    forceTotal[threadIdx.x] = f;
    __syncthreads();
    //TODO Implement a warp reduction
    if(threadIdx.x==0){
      float4 ft = make_float4(0.0f);
      for(int i=0; i<TPB; i++){
	ft += forceTotal[i];
      }
      /*Write to global memory*/
      force[p] += ft;
    }

  }


  void computeBondedForce(float4 *force, float4 *pos,
			  uint *bondStart, uint *bondEnd, uint *bondedParticleIndex, 
			  Bond* bondList, uint N, uint Nparticles_with_bonds, uint nbonds){
    computeBondedForceD<<<Nparticles_with_bonds, TPB>>>(force, pos,
							bondStart, bondEnd,
							bondedParticleIndex,  bondList);
  }




  struct bondedForcesFP_functor{
    BondFP   *bondListFP;
    __host__ __device__ bondedForcesFP_functor(BondFP *bondList):
      bondListFP(bondList){}
    //The operation is performed on creation
    template <typename Tuple>
    __device__  void operator()(Tuple t){
      /*Retrive the data*/
      float4 pos = get<0>(t);
      float4 force = get<1>(t);
      uint first = get<2>(t); //bondStart
      uint last = get<3>(t);  //bondEnd

      /*If I am connected to some particle*/
      if(first!=0xffffffff){

	BondFP bond;
	float4 posFP; //The other particle
	float r0, k; //The bond info
      
	float3 r12;
	/*For all particles connected to me*/
	for(int b = first; b<last; b++){
	  /*Retrieve bond*/
	  bond = bondListFP[b];
	  r0 = bond.r0;
	  k = bond.k;
	  posFP = make_float4(bond.pos);

	  /*Compute force*/
	  r12 =  make_float3(pos-posFP);
	  apply_pbc(r12);

	  float invr = 0.0f;
	  if(r0!=0.0f) invr = rsqrt(dot(r12, r12));
	
	  float fmod = -k*(1.0f-r0*invr); //F = -k·(r-r0)·rvec/r
	  force += make_float4(fmod*r12);
	}
      }
      get<1>(t) = force;
    }
  };



  void computeBondedForceFixedPoint(float4 *force, float4 *pos,
				    uint *bondStartFP, uint *bondEndFP, BondFP* bondListFP,
				    uint N, uint nbonds){

    device_ptr<float4> d_pos4(pos);
    device_ptr<float4> d_force4(force);
    device_ptr<uint> d_bondStart(bondStartFP);
    device_ptr<uint> d_bondEnd(bondEndFP);

    /**Thrust black magic to perform a multiple transformation, see the functor description**/
    for_each(
	     make_zip_iterator( make_tuple( d_pos4, d_force4, d_bondStart, d_bondEnd)),
	     make_zip_iterator( make_tuple( d_pos4 + N, d_force4 +N, d_bondStart+N, d_bondEnd+N)),
	     bondedForcesFP_functor(bondListFP)); 

  }


  /***********************THREE BONDED FORCES*****************************/



  //Custom kernel to compute and sum the force in a three particle angle spring
  /*
    Computes the potential: V(theta) = 0.5 K(theta-theta_0)^2
    F(\vec{ri}) = d(V(theta))/d(cos(theta))·d(cos(theta))/d(\vec{ri})
   */
  __global__ void computeThreeBondedForceD(float4* __restrict__ force, const float4* __restrict__ pos,
					   const uint* __restrict__ bondStart,
					   const uint* __restrict__ bondEnd,
					   const uint* __restrict__ bondedParticleIndex,
					   const ThreeBond* __restrict__ bondList){
    __shared__ float4 forceTotal[TPB];
    /*A block per particle, as in computeBondedForcesD*/
    uint p = bondedParticleIndex[blockIdx.x];
  
    float4 posp = pos[p];
  
    uint first = bondStart[p];
    uint last = bondEnd[p];
   
    float4 f = make_float4(0.0f);

    ThreeBond bond;//The current bond
    uint i,j,k;             //The bond indices
    float4 posi,posj, posk; //The bond particles
    float r0, kspring, ang0; //The bond info

    /*         i -------- j -------- k*/
    /*             rij->      <-rkj  */
    
    float3 rij, rkj; //rij = ri - rj
  
    float invsqrij, invsqrkj; //1/|rij|
    float rij2, rkj2;  //|rij|^2

    
    float a2; 
    float cijk, sijk;
    float a, a11, a12, a22;
    float ampli;

    /*Go through my bonds*/
    for(int b = first+threadIdx.x; b<last; b+=blockDim.x){
      /*Recover bond info*/
      bond = bondList[b];
      i = bond.i;
      j = bond.j;
      k = bond.k;

      kspring = bond.kspring;
      r0 = bond.r0;
      ang0 = bond.ang;

      
      //TODO Texture memory target
      /*Store the positions of the three particles*/
      /*We already got one of them, p*/
      /*Differentiate between the three particles in the bond*/
      if(p==i){
	posi = posp;
	posj = pos[j];
	posk = pos[k];
      }
      else if(p==j){
	posi = pos[i];
	posj = posp;
	posk = pos[k];
      }
      else{
	posi = pos[i];
	posj = pos[j];
	posk = posp;
      }

      /*Compute distances and vectors*/
      /***rij***/
      rij =  make_float3(posi-posj);
      apply_pbc(rij);
      rij2 = dot(rij, rij);
      invsqrij = rsqrt(rij2);
      /***rkj***/
      rkj =  make_float3(posk-posj);
      apply_pbc(rkj);
      rkj2 = dot(rkj, rkj);
      invsqrkj = rsqrt(rkj2);
      /********/
      
      a2 = invsqrij * invsqrkj;
      cijk = dot(rij, rkj)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)

      /*Cos must stay in range*/
      if(cijk>1.0f) cijk = 1.0f;
      else if (cijk<-1.0f) cijk = -1.0f;
      
      sijk = sqrt(1.0f-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
      /*sijk cant be zero to avoid division by zero*/
      if(sijk<0.000001f) sijk = 0.000001f;

      ampli = -kspring * (acos(cijk) - ang0); //The force amplitude -k·(theta-theta_0)

      //Magical trigonometric relations to infere the direction of the force
      a = ampli/sijk;
      a11 = a*cijk/rij2;
      a12 = -a*a2;
      a22 = a*cijk/rkj2;
      
      /*Sum according to my position in the bond*/
      // i ----- j ------ k
      if(p==i){
	f += make_float4(a11*rij + a12*rkj); //Angular spring
	
	//f += make_float4(-kspring*(1.0f - r0*invsqrij)*rij ); //Harmonic spring
	f += make_float4((-kspring/(1.0f-rij2/(r0*r0)))*rij); //fene spring
      }
      else if(p==j){
	//Angular spring
	f -= make_float4(a11*rij + a12*rkj + a22*rkj + a12*rij);
	

	// f += make_float4(kspring*(1.0f - r0*invsqrij)*rij); //First harmonic spring
	// f += make_float4(kspring*(1.0f - r0*invsqrkj)*rkj); //Second harmonic spring

	f -= make_float4((-kspring/(1.0f-rij2/(r0*r0)))*rij); // first fene spring
	f -= make_float4((-kspring/(1.0f-rkj2/(r0*r0)))*rkj); //second fene spring
      }
      else if(p==k){
	//Angular spring
	f += make_float4(a22*rkj + a12*rij);
	//Harmonic spring
	//f += make_float4(-kspring*(1.0f-r0*invsqrkj)*rkj);
	f += make_float4((-kspring/(1.0f-rkj2/(r0*r0)))*rkj); //fene spring
      }
    }

    //The fisrt thread sums all the contributions
    forceTotal[threadIdx.x] = f;
    __syncthreads();
    //TODO Implement a warp reduction
    if(threadIdx.x==0){
      float4 ft = make_float4(0.0f);
      for(int i=0; i<TPB; i++){
	ft += forceTotal[i];
      }
      force[p] += ft;
    }

  }








  void computeThreeBondedForce(float4 *force, float4 *pos,
			       uint *bondStart, uint *bondEnd, uint *bondedParticleIndex, 
			       ThreeBond* bondList, uint N, uint Nparticles_with_bonds, uint nbonds){
  
    computeThreeBondedForceD<<<Nparticles_with_bonds, TPB>>>(force, pos,
							     bondStart, bondEnd,
							     bondedParticleIndex,  bondList);



  }


}
