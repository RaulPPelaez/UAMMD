/*

  See GPUUtils.h for texture fetch overloads

 */

#ifndef TEXTURE_H
#define TEXTURE_H
#include"globals/defines.h"
#include<cuda_runtime.h>
#include"utils/helper_gpu.cuh"
#include<cstring>
typedef unsigned int uint;


struct TexReference{
  /*Raw pointer to device memory, 
    in order to avoid templating this and mimic the typeless cudaTextureObject_t...*/
  void *d_ptr;
  /*Texture reference*/
  cudaTextureObject_t tex;
};


template<typename T>
class Texture{
public:
  cudaTextureObject_t tex;
  uint n;
  T *d_m;
  
  Texture(): tex(0), n(0), d_m(nullptr){}
  //Texture(uint i): tex(i), n(0), d_m(nullptr){}
  ~Texture(){
    if(tex!=0)
      cudaDestroyTextureObject(tex);
  }
  void destroy(){
    if(tex!=0)
      cudaDestroyTextureObject(tex);
  }
  Texture(T *d_m, uint n): d_m(d_m), n(n){
    this->init(d_m, n);
  }
  void init(T *d_m, uint n){    
    this->d_m = d_m;
    this->n = n;
    
    if(sysInfo.cuda_arch<=210) return;
    cudaResourceDesc resDesc;					  
    memset(&resDesc, 0, sizeof(resDesc));				  
    resDesc.resType = cudaResourceTypeLinear;			  
    resDesc.res.linear.devPtr = d_m;					  
    resDesc.res.linear.desc = cudaCreateChannelDesc<T>();		  
    resDesc.res.linear.sizeInBytes = n*sizeof(T);			  
                                                                       
    cudaTextureDesc texDesc;						  
    memset(&texDesc, 0, sizeof(texDesc));				  
    texDesc.readMode = cudaReadModeElementType;			  
                                                                       
    gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

  }
  
  operator cudaTextureObject_t(){ return this->tex;}
  operator TexReference(){ return {(void *)this->d_m, this->tex};}

};

//Double is encoded in an int2
template<>
inline void Texture<double>::init(double *d_m, uint n){
    this->d_m = d_m;
    this->n = n;
    
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
inline void Texture<double4>::init(double4 *d_m, uint n){
    this->d_m = d_m;
    this->n = 2*n;
    
    if(sysInfo.cuda_arch<=210) return;
    cudaResourceDesc resDesc;					  
    memset(&resDesc, 0, sizeof(resDesc));				  
    resDesc.resType = cudaResourceTypeLinear;			  
    resDesc.res.linear.devPtr = (int4 *)d_m;					  
    resDesc.res.linear.desc = cudaCreateChannelDesc<int4>();		  
    resDesc.res.linear.sizeInBytes = n*sizeof(int4);  
                                                                       
    cudaTextureDesc texDesc;						  
    memset(&texDesc, 0, sizeof(texDesc));				  
    texDesc.readMode = cudaReadModeElementType;			  
                                                                       
    gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}

#endif
