/* Raul P. Pelaez 2016. cudaTextureObject_t wrapper

  See GPUUtils.h for texture fetch overloads

TODO:
100- Code a texture object that takes into account the offset when using texture interpolation. ( you have to sum 0.5/N when fetching with tex1D. Do this in Potential, something like PotentialTableLookup
 */

#ifndef TEXTURE_H
#define TEXTURE_H
#include"globals/defines.h"
#include<cuda_runtime.h>
#include"utils/helper_gpu.cuh"
#include<cstring>

struct TexReference{
  /*Raw pointer to device memory, 
    in order to avoid templating this and mimic the typeless cudaTextureObject_t...*/
  void *d_ptr;
  /*Texture reference*/
  cudaTextureObject_t tex;
};


struct Texture{
public:
  // cudaTextureObject_t tex;
  // uint n;
  // T *d_m;
  
  // Texture(): tex(0), n(0), d_m(nullptr){}
  // //Texture(uint i): tex(i), n(0), d_m(nullptr){}
  // ~Texture(){
  //   if(tex!=0)
  //     cudaDestroyTextureObject(tex);
  // }
  
  static void destroy(cudaTextureObject_t &tex){
    if(tex!=0)
      cudaDestroyTextureObject(tex);
  }
  // Texture(T *d_m, uint n): d_m(d_m), n(n){
  //   this->init(d_m, n);
  // }
  template<typename T>
  static void init(T *d_m, cudaTextureObject_t &tex, uint n){
    // this->d_m = d_m;
    // this->n = n;
    
    if(sysInfo.cuda_arch<=210) return;
    cudaResourceDesc resDesc;					  
    memset(&resDesc, 0, sizeof(resDesc));				  
    resDesc.resType = cudaResourceTypeLinear;			  
    resDesc.res.linear.devPtr = (void *)d_m;
    resDesc.res.linear.desc = cudaCreateChannelDesc<T>();		  
    resDesc.res.linear.sizeInBytes = n*sizeof(T);			  
                                                                       
    cudaTextureDesc texDesc;						  
    memset(&texDesc, 0, sizeof(texDesc));				  
    texDesc.readMode = cudaReadModeElementType;			  
                                                                       
    gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

  }
  
  // operator cudaTextureObject_t(){ return this->tex;}
  // operator TexReference(){ return {(void *)this->d_m, this->tex};}

};

/*inline to avoid multiple definition conflicts*/

//Double is encoded in an int2
template<>
inline void Texture::init<double>(double *d_m, cudaTextureObject_t &tex, uint n){
    // this->d_m = d_m;
    // this->n = n;
    
    if(sysInfo.cuda_arch<=210) return;
    cudaResourceDesc resDesc;					  
    memset(&resDesc, 0, sizeof(resDesc));				  
    resDesc.resType = cudaResourceTypeLinear;			  
    resDesc.res.linear.devPtr = (int2 *)d_m;					  
    resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();		  
    resDesc.res.linear.sizeInBytes = n*sizeof(int2);  
                                                                       
    cudaTextureDesc texDesc;						  
    memset(&texDesc, 0, sizeof(texDesc));				  
    texDesc.readMode = cudaReadModeElementType;			  
                                                                       
    gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}

//Double4 is encoded in two int4
template<>
inline void Texture::init<double4>(double4 *d_m,cudaTextureObject_t &tex, uint n){
    // this->d_m = d_m;
  //this->n = 2*n;
    
  if(sysInfo.cuda_arch<=210) return;
  cudaResourceDesc resDesc;					  
  memset(&resDesc, 0, sizeof(resDesc));				  
  resDesc.resType = cudaResourceTypeLinear;			  
  resDesc.res.linear.devPtr = (uint4 *)d_m;					  
  resDesc.res.linear.desc = cudaCreateChannelDesc<uint4>();		  
  resDesc.res.linear.sizeInBytes = 2*n*sizeof(uint4);  
                                                                       
  cudaTextureDesc texDesc;						  
  memset(&texDesc, 0, sizeof(texDesc));				  
  texDesc.readMode = cudaReadModeElementType;			  
                                                                       
  gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}



/*************************FETCH OVERLOADS*****************************/
template<>
inline __device__ double tex1Dfetch<double>(cudaTextureObject_t t, int i){

  uint2 v = tex1Dfetch<uint2>(t, i);

  return __hiloint2double(v.y, v.x);
}


template<>
inline __device__ double4 tex1Dfetch<double4>(cudaTextureObject_t t, int i){
  
  uint4 v1 = tex1Dfetch<uint4>(t,2*i);
  uint4 v2 = tex1Dfetch<uint4>(t,2*i+1);

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
