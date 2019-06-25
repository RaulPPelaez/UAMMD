#ifndef CUFFTCOMPLEX2_CUH
#define CUFFTCOMPLEX2_CUH

#include"global/defines.h"
#include"utils/cufftPrecisionAgnostic.h"

namespace uammd{
  /*A convenient struct to pack 2 complex numbers, that is 4 real numbers*/

  template<class T>
  struct cufftComplex2_t{
    cufftComplex_t<T> x,y;
    using cufftComplex2 = cufftComplex2_t<T>;    
    friend inline  __device__ __host__ cufftComplex2 operator+(const cufftComplex2 &a, const cufftComplex2 &b){
      return {a.x + b.x, a.y + b.y};
    }
    friend inline  __device__ __host__ void operator+=(cufftComplex2 &a, const cufftComplex2 &b){
      a.x += b.x; a.y += b.y;
    }
  
    friend inline  __device__ __host__ cufftComplex2 operator-(const cufftComplex2 &a, const cufftComplex2 &b){
      return {a.x - b.x, a.y - b.y};
    }
    friend inline  __device__ __host__ void operator-=(cufftComplex2 &a, const cufftComplex2 &b){
      a.x -= b.x; a.y -= b.y;
    }


  
    friend inline  __device__ __host__ cufftComplex2 operator*(const cufftComplex2 &a, real b){
      cufftComplex2 res;
      res.x = a.x * b;
      res.y = a.y * b;
      return res;    
    }
    friend inline  __device__ __host__ cufftComplex2 operator*(real b, const cufftComplex2 &a){
      return a*b;
    }
    friend inline  __device__ __host__ void operator*=(cufftComplex2 &a, real b){
      a = a*b;
    }      

  
    friend inline  __device__ __host__ cufftComplex2 operator/(const cufftComplex2 &a, real b){
      cufftComplex2 res;
      res.x = a.x / b;
      res.y = a.y / b;
      return res;    
    }
    friend inline  __device__ __host__ cufftComplex2 operator/(real b, const cufftComplex2 &a){
      cufftComplex2 res;
      res.x = b/a.x;
      res.y = b/a.y;
      return res;
    }
    friend inline  __device__ __host__ void operator/=(cufftComplex2 &a, real b){
      a = a/b;
    }      

  };

  


}

#endif