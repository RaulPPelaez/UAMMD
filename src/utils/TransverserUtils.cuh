/*Raul P. Pelaez 2018. Some utilities for working with Transversers.

 */

#ifndef TRANSVERSERUTILS_CUH
#define TRANSVERSERUTILS_CUH

#include"utils/cxx_utils.h"
namespace uammd{
  //Defines the most simple and general posible transverser that does nothing. 
  //This allows to fall back to nothing when a transverser is required.  
  //Just does nothing, every function has an unspecified number of arguments. Very general.
  struct BasicNullTransverser{
    template<class ...T> static constexpr inline __host__ __device__ int zero(T...){ return 0;}
    template<class ...T> static constexpr inline __host__ __device__ int compute(T...){ return 0;}
    template<class ...T> static inline __host__ __device__ void accumulate(T...){}
    template<class ...T> static inline __host__ __device__ void set(T...){}       
  };


  namespace SFINAE{
    // For Transversers, detects if a Transverser has getSharedMemorySize, therefore needing to allocate extra shared memory when launching a kernel that involves it.
    SFINAE_DEFINE_HAS_MEMBER(getSharedMemorySize)

    //This delegator allows to ask a Transverser for the shared memory it needs even when getSharedMemorySize() is not present (it will return 0 by default)
    template<class T, bool general = has_getSharedMemorySize<T>::value>
    class SharedMemorySizeDelegator;

    template<class T>
    class SharedMemorySizeDelegator<T, true>{
    public:
      static inline size_t getSharedMemorySize(T& t){ return t.getSharedMemorySize();}
    };


    template<class T>
    class SharedMemorySizeDelegator<T, false>{
    public:
      static constexpr inline size_t getSharedMemorySize(const T& t){return 0;}
    };



  
    //For Transversers, detects if a Transverser has getInfo, therefore being a general Transverser
    //This macro defines a tempalted struct has_getInfo<T> that contains a static value=0 if T has not a getInfo method, and 1 otherwise
    SFINAE_DEFINE_HAS_MEMBER(getInfo);


    //Delegator allows to use a transverser as if it had a getInfo function even when it does not provide one
    //A Transverser that provides a getInfo function and whose compute function takes 4 arguments
    //  (posi, posj, infoi, infoj, being info the return type of getInfo) is called a general transverser

    template<class T, bool general = has_getInfo<T>::value>
    class Delegator;

    //Specialization for general transversers,
    //    In this case, delegator holds the value of infoi, 
    //    so it can be stored only when we have a general Transverser
    //    Additionally, it supports shared memory reads, so you can use a delegator 
    //    even if the kernel writes/reads infoj to/from shared memory, see an example in NBodyForces.cuh
  
    template<class T>
    class Delegator<T, true>{
    public:
      //Return type of getInfo
      using myType = decltype(((T*)nullptr)->getInfo(0));

      //In this case getInfo just calls the transversers getInfo
      inline __device__ void getInfo(T &t, const int &id){infoi = t.getInfo(id);}

      //Size in bytes of the type of myType
      static constexpr inline __host__ __device__ size_t sizeofInfo(){return sizeof(myType);}

      //Write the info of particle i_load to shared memory
      inline __device__ void fillSharedMemory(T &t, void * shInfo, int i_load){
	((myType*)shInfo)[threadIdx.x] = t.getInfo(i_load);
      }
      //Calls getInfo(j) and compute with 4 arguments, returns the type of zero(),  i.e the quantity
      inline __device__ auto compute(T &t, const int &j,
				     const real4& posi, const real4 &posj)-> decltype(t.zero()){
	return t.compute(posi, posj, infoi, t.getInfo(j));
      }
      //Same as above, but take infoj from the element counter of a shared memory array of type myType
      inline __device__ auto computeSharedMem(T &t,
					      const real4& posi, const real4 &posj,
					      void* shMem, int counter)-> decltype(t.zero()){
      
	return t.compute(posi, posj, infoi, ((myType*)shMem)[counter]);
      }

      myType infoi;
    };
    //When we have a simple transverser, getInfo does nothing,
    //  and compute has just two arguments
    template<class T>
    class Delegator<T, false>{
    public:
    
      static inline __device__  void getInfo(T &t, const int &id){}
      //I dont have any additional info
      static constexpr inline __host__ __device__ size_t sizeofInfo(){return 0;}
      //I do not write anything to shared memory
      static inline __device__ void fillSharedMemory(T &t, void * shInfo, int i_load){}
      static constexpr inline __device__  auto compute(T &t, const int &j,
				      const real4 &posi, const real4 &posj)-> decltype(t.zero()){
	return t.compute(posi, posj);
      }
      //I just ignore the shared memory part
      static constexpr inline __device__ auto computeSharedMem(T &t,
					      const real4& posi, const real4 &posj,
					      void* shMem, int counter)-> decltype(t.zero()){
	return t.compute(posi, posj);

      }

    };
  }
}
#endif
