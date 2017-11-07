/*Raul P. Pelaez 2017. Box logic
  
  This object is used throughout UAMMD to retrieve information about the box the particles are in (if any).
  
  It can be used to apply periodic boundary conditions and check if a position lies inside of it

*/
#ifndef BOX_CUH
#define BOX_CUH
#include"global/defines.h"
#include"utils/vector.cuh"
#include"utils/GPUUtils.cuh"

namespace uammd{
  struct Box{
    real3 boxSize, minusInvBoxSize;
  
    Box():Box(0){}
    Box(real L):Box(make_real3(L)){}
    Box(real2 L):Box(make_real3(L, 0)){}
    Box(real3 L): boxSize(L), minusInvBoxSize(make_real3(real(-1.0)/L.x, real(-1.0)/L.y, real(-1.0)/L.z)){
      if(boxSize.x==real(0.0)) 	minusInvBoxSize.x = real(0.0);
      if(boxSize.y==real(0.0))	minusInvBoxSize.y = real(0.0);
      if(boxSize.z==real(0.0))	minusInvBoxSize.z = real(0.0);
    }
    inline __host__ __device__ real3 apply_pbc(const real3 &r) const{
      //return  r - floorf(r/L+real(0.5))*L; //MIC Algorithm
      const real3 offset = floorf(r*minusInvBoxSize + real(0.5)); //MIC Algorithm
      return  r + offset*boxSize;
    }
    template< class vecType>
    inline __device__ __host__ bool isInside(const vecType &pos){
      real3 boxSizehalf = real(0.5)*boxSize;
      if(pos.x <= -boxSizehalf.x || pos.x > boxSizehalf.x) return false;
      if(pos.y <= -boxSizehalf.y || pos.y > boxSizehalf.y) return false;
      if(pos.z <= -boxSizehalf.z || pos.z > boxSizehalf.z) return false;
      return true;
    }
  };

}
#endif