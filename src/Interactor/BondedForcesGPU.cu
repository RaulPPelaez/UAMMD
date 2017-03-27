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
#include"globals/defines.h"
#include"BondedForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include<thrust/device_ptr.h>
#include<thrust/reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include"utils/GPUutils.cuh"

using namespace thrust;
namespace bonded_forces_ns{
  //Parameters in constant memory, super fast access
  __constant__ Params params; 

  void initGPU(Params m_params){
    m_params.invL = 1.0/m_params.L;
    /*Upload parameters to constant memory*/
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }


  //MIC algorithm
  // template<typename vecType>
  // inline __device__ void apply_pbc(vecType &r){
  //   real3 r3 = make_real3(r.x, r.y, r.z);
  //   real3 shift = (floorf(r3*params.invL+real(0.5))*params.L); //MIC Algorithm
  //   r.x -= shift.x;
  //   r.y -= shift.y;
  //   r.z -= shift.z;
  // }


#define TPB 64

  //Custom kernel to compute and sum the force
  __global__ void computeBondedForceD(real4* __restrict__ force, const real4* __restrict__ pos,
				      const uint* __restrict__ bondStart,
				      const uint* __restrict__ bondEnd,
				      const uint* __restrict__ bondedParticleIndex,
				      const Bond* __restrict__ bondList){
    __shared__ real4 forceTotal[TPB]; /*Each thread*/
    /*A block per particle*/
    /*Instead of launching a thread per particle and discarding those without any bond,
      I store an additional array of size N_particles_with_bonds that contains the indices
      of the particles that are involved in at least one bond. And only launch N_particles_with_bonds blocks*/
    /*Each thread in a block computes the force on particle p due to one (or several) bonds*/
    uint p = bondedParticleIndex[blockIdx.x]; //This block handles particle p
    real4 posi = pos[p]; //Get position

    /*First and last bond indices of p in bondList*/
    uint first = bondStart[p]; 
    uint last = bondEnd[p];
   
    real4 f = make_real4(real(0.0));

    Bond bond; //The current bond
    
    int j; real4 posj; //The other particle
    real r0, k; //The bond info
      
    real3 r12;

    
    for(int b = first+threadIdx.x; b<last; b+=blockDim.x){
      /*Read bond info*/
      bond = bondList[b];
      j = bond.j;
      r0 = bond.r0;
      k = bond.k;
      /*Bring pos of other particle*/
      posj = pos[j];
    
      /*Compute force*/
      r12 =  make_real3(posi-posj);
      apply_pbc(r12);
    
      real invr = rsqrt(dot(r12, r12));
    
      real fmod = -k*(real(1.0)-r0*invr); //F = -k·(r-r0)·rvec/r
      
      f += make_real4(fmod*r12);
    }

    /*The first thread sums all the contributions*/
    forceTotal[threadIdx.x] = f;
    __syncthreads();
    //TODO Implement a warp reduction
    if(threadIdx.x==0){
      real4 ft = make_real4(0.0f);
      for(int i=0; i<TPB; i++){
	ft += forceTotal[i];
      }
      /*Write to global memory*/
      force[p] += ft;
    }

  }


  void computeBondedForce(real4 *force, real4 *pos,
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
      /*Retrive the.data*/
      real4 pos = get<0>(t);
      real4 force = get<1>(t);
      uint first = get<2>(t); //bondStart
      uint last = get<3>(t);  //bondEnd

      /*If I am connected to some particle*/
      if(first!=0xffffffff){

	BondFP bond;
	real4 posFP; //The other particle
	real r0, k; //The bond info
      
	real3 r12;
	/*For all particles connected to me*/
	for(int b = first; b<last; b++){
	  /*Retrieve bond*/
	  bond = bondListFP[b];
	  r0 = bond.r0;
	  k = bond.k;
	  posFP = make_real4(bond.pos);

	  /*Compute force*/
	  r12 =  make_real3(pos-posFP);
	  apply_pbc(r12);

	  real invr = real(0.0);
	  if(r0!=real(0.0)) invr = rsqrt(dot(r12, r12));
	
	  real fmod = -k*(real(1.0)-r0*invr); //F = -k·(r-r0)·rvec/r
	  force += make_real4(fmod*r12);
	}
      }
      get<1>(t) = force;
    }
  };



  void computeBondedForceFixedPoint(real4 *force, real4 *pos,
				    uint *bondStartFP, uint *bondEndFP, BondFP* bondListFP,
				    uint N, uint nbonds){

    device_ptr<real4> d_pos4(pos);
    device_ptr<real4> d_force4(force);
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
  __global__ void computeThreeBondedForceD(real4* __restrict__ force, const real4* __restrict__ pos,
					   const uint* __restrict__ bondStart,
					   const uint* __restrict__ bondEnd,
					   const uint* __restrict__ bondedParticleIndex,
					   const ThreeBond* __restrict__ bondList){
    __shared__ real4 forceTotal[TPB];
    /*A block per particle, as in computeBondedForcesD*/
    uint p = bondedParticleIndex[blockIdx.x];
  
    real4 posp = pos[p];
  
    uint first = bondStart[p];
    uint last = bondEnd[p];
   
    real4 f = make_real4(real(0.0));

    ThreeBond bond;//The current bond
    uint i,j,k;             //The bond indices
    real4 posi,posj, posk; //The bond particles
    real r0, kspring, ang0; //The bond info

    /*         i -------- j -------- k*/
    /*             rij->      <-rkj  */
    
    real3 rij, rkj; //rij = ri - rj
  
    real invsqrij, invsqrkj; //1/|rij|
    real rij2, rkj2;  //|rij|^2

    
    real a2; 
    real cijk, sijk;
    real a, a11, a12, a22;
    real ampli;

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
      rij =  make_real3(posi-posj);
      apply_pbc(rij);
      rij2 = dot(rij, rij);
      invsqrij = rsqrt(rij2);
      /***rkj***/
      rkj =  make_real3(posk-posj);
      apply_pbc(rkj);
      rkj2 = dot(rkj, rkj);
      invsqrkj = rsqrt(rkj2);
      /********/
      
      a2 = invsqrij * invsqrkj;
      cijk = dot(rij, rkj)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)

      /*Cos must stay in range*/
      if(cijk>1.0f) cijk = real(1.0);
      else if (cijk<-1.0f) cijk = -real(1.0);
      
      sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
      /*sijk cant be zero to avoid division by zero*/
      if(sijk<real(0.000001)) sijk = real(0.000001);

      ampli = -real(100.0)*kspring * (acos(cijk) - ang0); //The force amplitude -k·(theta-theta_0)

      //Magical trigonometric relations to infere the direction of the force
      a = ampli/sijk;
      a11 = a*cijk/rij2;
      a12 = -a*a2;
      a22 = a*cijk/rkj2;
      
      /*Sum according to my position in the bond*/
      // i ----- j ------ k
      if(p==i){
	f += make_real4(a11*rij + a12*rkj); //Angular spring

	f += make_real4(-kspring*(real(1.0) - r0*invsqrij)*rij ); //Harmonic spring
	
	// real rep = 0;
	// if(1.0f/invsqrij >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrij,14) + 24.0f*pow(invsqrij,8);
	//f += make_real4((-kspring/(1.0f-rij2/(r0*r0)) )*rij); //fene spring
      }
      else if(p==j){
	//Angular spring
	f -= make_real4(a11*rij + a12*rkj + a22*rkj + a12*rij);
	

	f += make_real4(kspring*(real(1.0) - r0*invsqrij)*rij); //First harmonic spring
	f += make_real4(kspring*(real(1.0) - r0*invsqrkj)*rkj); //Second harmonic spring

	// real rep = 0;
	// if(1.0f/invsqrij >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrij,14) + 24.0f*pow(invsqrij,8);
	// f -= make_real4((-kspring/(1.0f-rij2/(r0*r0)) + rep)*rij); //fene spring

	// if(1.0f/invsqrkj >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrkj,14) + 24.0f*pow(invsqrkj,8);
	// f -= make_real4((-kspring/(1.0f-rkj2/(r0*r0)) + rep)*rkj); //fene spring


	
	// f -= make_real4((-kspring/(1.0f-rij2/(r0*r0)))*rij); // first fene spring
	//f -= make_real4((-kspring/(1.0f-rkj2/(r0*r0)))*rkj); //second fene spring
      }
      else if(p==k){
	//Angular spring
	f += make_real4(a22*rkj + a12*rij);
	//Harmonic spring
	f += make_real4(-kspring*(real(1.0)-r0*invsqrkj)*rkj);

	// real rep = 0;
	// if(1.0f/invsqrkj >= pow(2.0f,1.0f/6.0f))
	//   rep = -48.0f*pow(invsqrkj,14) + 24.0f*pow(invsqrkj,8);

	//f += make_real4((-kspring/(1.0f-rkj2/(r0*r0)))*rkj); //fene spring
      }
    }

    //The fisrt thread sums all the contributions
    forceTotal[threadIdx.x] = f;
    __syncthreads();
    //TODO Implement a warp reduction
    if(threadIdx.x==0){
      real4 ft = make_real4(real(0.0));
      for(int i=0; i<TPB; i++){
	ft += forceTotal[i];
      }
      force[p] += ft;
    }

  }








  void computeThreeBondedForce(real4 *force, real4 *pos,
			       uint *bondStart, uint *bondEnd, uint *bondedParticleIndex, 
			       ThreeBond* bondList, uint N, uint Nparticles_with_bonds, uint nbonds){
  
    computeThreeBondedForceD<<<Nparticles_with_bonds, TPB>>>(force, pos,
							     bondStart, bondEnd,
							     bondedParticleIndex,  bondList);



  }


}
