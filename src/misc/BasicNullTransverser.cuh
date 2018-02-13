/*Raul P. Pelaez 2018. Defines the most simple and general posible transverser that does nothing. 
  This allows to fall back to nothing when a transverser is required.
  
 */

#ifndef BASICNULLTRANSERSER_CUH
#define BASICNULLTRANSERSER_CUH

namespace uammd{
  //Just does nothing, every function has an unspecified number of arguments. Very general.
  struct BasicNullTransverser{
    template<class ...T> inline __device__ int zero(){ return 0;}
    template<class ...T> inline __device__ int compute(T...){ return 0;}
    template<class ...T> inline __device__ void accumulate(T...){}
    template<class ...T> inline __device__ void set(T...){}       
  };
}


#endif