/*Raul P. Pelaez 2016. Custom texture fetches for double and TexReference*/
#ifndef GPUUTILS_H
#define GPUUTILS_H

#include"Texture.h"

template<>
inline __device__ double tex1Dfetch<double>(cudaTextureObject_t t, int i){

  int2 v = tex1Dfetch<int2>(t, i);

  return __hiloint2double(v.y, v.x);
}


template<>
inline __device__ double4 tex1Dfetch<double4>(cudaTextureObject_t t, int i){
  
 int4 v1 = tex1Dfetch<int4>(t,2*i);
 int4 v2 = tex1Dfetch<int4>(t,2*i+1);

 return make_double4(
 		      __hiloint2double(v1.y, v1.x),
 		      __hiloint2double(v1.w, v1.z),
 		      __hiloint2double(v2.y, v2.x),
 		      __hiloint2double(v2.w, v2.z));		      
}


template<class T>
inline __device__ T tex1Dfetch(TexReference t, int i){
#if __CUDA_ARCH__>210
  return tex1Dfetch<T>(t.tex, i);
#else
  return ((T*)t.d_ptr)[i];
#endif
}

template<>
inline __device__ double4 tex1Dfetch<double4>(TexReference t, int i){

#if __CUDA_ARCH__>=350
  double2 a= __ldg((double2*)t.d_ptr+i*2);
  double2 b= __ldg((double2*)t.d_ptr+i*2+1);
  return make_double4(a.x, a.y, b.x, b.y);
#else
  return *((double4*)t.d_ptr+i);
#endif    
}


#endif



