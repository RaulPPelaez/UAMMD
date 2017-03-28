/*Raul P. Pelaez 2017*/
/*WARNING!!! THIS FILE CAN CONTAIN ONLY HEADERS, EXTERNS AND INLINES,
  Any definition of a function outside this terms will result in 'Multiple definitions' erros.*/
/*Any declaration of an extern item will be placed in main.cpp, initialization will take place after Driver::setParameters is called, in globals/initGPU.cu::initGPU*/
/*This file is a trick so the compiler inlines device functions across compilation units*/
/*This file can only be included once per C.U., so if a file includes a file that includes this, you dont have, and cant include it.*/

#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"globals/globals.h"

//MIC algorithm
// template<typename vecType>
// inline __device__ void apply_pbc(vecType &r){
//   real3 r3 = make_real3(r.x, r.y, r.z);
//   real3 shift = (floorf(r3*params.invL+0.5f)*params.L); //MIC Algorithm
//   r.x -= shift.x;
//   r.y -= shift.y;
//   r.z -= shift.z;    
// }
#ifndef GPUUTILS_CUH
#define GPUUTILS_CUH

inline __device__ void apply_pbc(real3 &r){    
  r -= floorf(r*gcnfGPU.invL+real(0.5))*gcnfGPU.L; //MIC Algorithm
}

struct BoxUtils{
  real3 L, invL;
  BoxUtils(real3 L): L(L), invL(1.0/L){
    if(L.z==real(0.0))
      invL.z = real(0.0);
  }
  inline __device__ void apply_pbc(real3 &r){    
    r -= floorf(r*invL+real(0.5))*L; //MIC Algorithm
  }
};

#endif