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
      if(boxSize.x==real(0.0) or isinf(boxSize.x)) minusInvBoxSize.x = real(0.0);
      if(boxSize.y==real(0.0) or isinf(boxSize.y)) minusInvBoxSize.y = real(0.0);
      if(boxSize.z==real(0.0) or isinf(boxSize.z)) minusInvBoxSize.z = real(0.0);
    }
    //Sets the periodicity of each dimension of the box.
    inline void setPeriodicity(bool x, bool y, bool z){
      if(!x) minusInvBoxSize.x = 0;
      if(!y) minusInvBoxSize.y = 0;
      if(!z) minusInvBoxSize.z = 0;
    }
    inline __host__ __device__ bool isPeriodicX() const{return minusInvBoxSize.x != 0;}
    inline __host__ __device__ bool isPeriodicY() const{return minusInvBoxSize.y != 0;}
    inline __host__ __device__ bool isPeriodicZ() const{return minusInvBoxSize.z != 0;}

    inline __host__ __device__ real3 apply_pbc(real3 r) const{
      //return  r - floorf(r/L+real(0.5))*L; //MIC Algorithm
      real3 offset = floorf(r*minusInvBoxSize + real(0.5)); //MIC Algorithm
      r.x += isPeriodicX()?(offset.x*boxSize.x):0;
      r.y += isPeriodicY()?(offset.y*boxSize.y):0;
      r.z += isPeriodicZ()?(offset.z*boxSize.z):0;
      return  r;
    }
    template< class vecType>
    inline __device__ __host__ bool isInside(const vecType &pos) const{
      real3 boxSizehalf = real(0.5)*boxSize;
      if(pos.x <= -boxSizehalf.x || pos.x > boxSizehalf.x) return false;
      if(pos.y <= -boxSizehalf.y || pos.y > boxSizehalf.y) return false;
      if(pos.z <= -boxSizehalf.z || pos.z > boxSizehalf.z) return false;
      return true;
    }
    inline __device__ __host__ real getVolume() const{
      if(boxSize.z != real(0.0))
	return boxSize.x*boxSize.y*boxSize.z;
      else
	return boxSize.x*boxSize.y;
    }

    bool operator == (const Box &other) const {
      return boxSize.x == other.boxSize.x and
	boxSize.y == other.boxSize.y and
	boxSize.z == other.boxSize.z and
	isPeriodicX() == other.isPeriodicX() and
	isPeriodicY() == other.isPeriodicY() and
	isPeriodicZ() == other.isPeriodicZ();
    }
    bool operator != (const Box &other) const{
      return !(this->operator==(other));
    }
  };

}
#endif
