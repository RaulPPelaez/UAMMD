#ifndef GPUUTILS_H
#define GPUUTILS_H

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

#endif



