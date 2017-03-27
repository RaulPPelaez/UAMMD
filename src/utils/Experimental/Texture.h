#ifndef TEXTURE_CUH
#define TEXTURE_CUH
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


namespace Texture{
  template<class T>
  inline cudaTextureObject_t create(T* data, int n){
    //if(sysInfo.cuda_arch<=210) return 0;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
    resDesc.res.linear.sizeInBytes = n*sizeof(T);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex;
    gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    if(!tex) std::cerr<<"Error setting up texture!!!"<<std::endl;
    return tex;
  }
  template<class T>
  inline void destroy(cudaTextureObject_t &tex){
    if(tex!=0){cudaDestroyTextureObject(tex);    tex = 0;}
  }
    


  /*Double is encoded as an int2*/
  template<>
  inline cudaTextureObject_t create<double>(double* data, int n){
    //if(sysInfo.cuda_arch<=210) return 0;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (int2*)data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
    resDesc.res.linear.sizeInBytes = n*sizeof(int2);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex;
    gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    return tex;
  }

  /*Double4 is encoded as two int4*/
  template<>
  inline cudaTextureObject_t create<double4>(double4* data, int n){
    //if(sysInfo.cuda_arch<=210) return 0;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (int4*)data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int4>();
    resDesc.res.linear.sizeInBytes = 2*n*sizeof(int4);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex;
    gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    return tex;
  }
}
/*************************FETCH OVERLOADS*****************************/
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
