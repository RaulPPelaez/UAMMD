/*Raul P. Pelaez 2017*/
/*WARNING!!! THIS FILE CAN CONTAIN ONLY HEADERS, EXTERNS AND INLINES,
  Any definition of a function outside this terms will result in 'Multiple definitions' erros.*/
/*Any declaration of an extern item will be placed in main.cpp, initialization will take place after Driver::setParameters is called, in globals/initGPU.cu::initGPU*/
/*This file is a trick so the compiler inlines device functions across compilation units*/
/*This file can only be included once per C.U., so if a file includes a file that includes this, you dont have, and cant include it.*/

#include"utils/vector_overloads.h"
#include"utils/helper_gpu.cuh"
#include"globals/globals.h"

#ifndef GPUUTILS_CUH
#define GPUUTILS_CUH


struct BoxUtils{
  real3 L, invL;
  BoxUtils(): BoxUtils(gcnf.L){}
  BoxUtils(real3 L): L(L), invL(1.0/L){
    if(L.z==real(0.0))
      invL.z = real(0.0);
  }
  inline __device__ void apply_pbc(real3 &r) const{    
    r -= floorf(r*invL+real(0.5))*L; //MIC Algorithm
  }
};

#endif