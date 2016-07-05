
#include"BondedForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include<thrust/device_ptr.h>
#include<thrust/reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>

using namespace thrust;

//Parameters in constant memory, super fast access
__constant__ BondedForcesParams bondedForcesParamsGPU; 



void initBondedForcesGPU(BondedForcesParams m_params){
  m_params.invL = 1.0f/m_params.L;
  /*Upload parameters to constant memory*/
  gpuErrchk(cudaMemcpyToSymbol(bondedForcesParamsGPU, &m_params, sizeof(BondedForcesParams)));

}



//MIC algorithm
inline __device__ void apply_pbc(float3 &r){
  r -= floorf(r*bondedForcesParamsGPU.invL+0.5f)*bondedForcesParamsGPU.L; 
}
inline __device__ void apply_pbc(float4 &r){
  r -= floorf(r*bondedForcesParamsGPU.invL+0.5f)*bondedForcesParamsGPU.L; //MIC algorithm
}


//Thrust trick to apply a transformation to each element of an array, in parallel
struct bondedForces_functor{
  Bond   *bondList;
  float4 *d_pos;
  __host__ __device__ bondedForces_functor(Bond *bondList, float4* pos):
    bondList(bondList), d_pos(pos){}
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

      Bond bond;
      int j; float4 posj; //The other particle
      float r0, k; //The bond info
      
      float3 r12;
      /*For all particles connected to me*/
      for(int b = first; b<last; b++){
	/*Retrieve bond*/
	bond = bondList[b];
	j = bond.j;
	r0 = bond.r0;
	k = bond.k;
	posj = d_pos[j];

	/*Compute force*/
	r12 =  make_float3(pos-posj);
	apply_pbc(r12);

	float invr = rsqrt(dot(r12, r12));
	
	float fmod = -k*(1.0f-r0*invr); //F = -k·(r-r0)·rvec/r
	force += make_float4(fmod*r12);
      }
    }
    get<1>(t) = force;
  }
};


void computeBondedForce(float4 *force, float4 *pos,
			uint *bondStart, uint *bondEnd, Bond* bondList, uint N, uint nbonds){

  device_ptr<float4> d_pos4(pos);
  device_ptr<float4> d_force4(force);
  device_ptr<uint> d_bondStart(bondStart);
  device_ptr<uint> d_bondEnd(bondEnd);

  /**Thrust black magic to perform a multiple transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_force4, d_bondStart, d_bondEnd)),
	   make_zip_iterator( make_tuple( d_pos4 + N, d_force4 +N, d_bondStart+N, d_bondEnd+N)),
	   bondedForces_functor(bondList, pos)); 

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
