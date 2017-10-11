/*Raul P. Pelaez 2017.

  C++ SFINAE utilities.

  This template black magic allows to provide a default functionality for an expected function when this function does not exists. 
  For example, a template that takes a struct and expects to find a function called "getSize()" inside it.
    With SFINAE one can call this function even when it is not present and provide a default behavior, maybe getSize() returning 0 when it is not present in the struct.

 */
#ifndef SFINAE_CUH
#define SFINAE_CUH

namespace SFINAE{

/*For Transversers, detects if a Transverser has getInfo, therefore being a general Transverser*/
template <typename T>
class has_getInfo
{
  typedef char one;
  typedef long two;

  template <typename C> static one test( decltype(&C::getInfo) ) ;
  template <typename C> static two test(...);

public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};



  /*Delegator can generalize a Transverser to use it as a general transverser
    even when no getInfo is present*/
  template<class T, bool general = has_getInfo<T>::value>
  class Delegator;

  /*Specialization for general transversers,
    In this case, delegator holds the value of infoi, 
    so it can be stored only in when we have a general Transverser
    Additionally, it supports shared memory reads, so you can use a delegator 
    even if the kernel writes/reads infoj to/from shared memory, see an example in NBodyForces.cuh
  */
  template<class T>
  class Delegator<T, true>{
  public:
    /*Return type of getInfo*/
    typedef  decltype(((T*)nullptr)->getInfo(0)) myType;

    /*In this case getInfo just calls the transversers getInfo*/
    inline __device__ void getInfo(T &t,int id){infoi = t.getInfo(id);}

    /*Size in bytes of the type of myType*/
    static inline __host__ __device__ size_t sizeofInfo(){return sizeof(myType);}

    /*Write the info of particle i_load to shared memory*/
    inline __device__ void fillSharedMemory(T &t, void * shInfo, int i_load){
      ((myType*)shInfo)[threadIdx.x] = t.getInfo(i_load);
    }
    /*Calls getInfo(j) and compute with 4 arguments, returns the type of zero(), 
      i.e the quantity*/
    inline __device__ auto compute(T &t, const int &j,
				   const real4& posi, const real4 &posj)-> decltype(t.zero()){
      return t.compute(posi, posj, infoi, t.getInfo(j));
    }
    /*Same as above, but take infoj from the element counter of a shared memory array of type myType*/
    inline __device__ auto computeSharedMem(T &t,
					    const real4& posi, const real4 &posj,
					    void* shMem, int counter)-> decltype(t.zero()){
      
      return t.compute(posi, posj, infoi, ((myType*)shMem)[counter]);
    }

    myType infoi;
  };
  /*When we have a simple transverser, getInfo does nothing,
    and compute has just two arguments*/
  template<class T>
  class Delegator<T, false>{
  public:
    
    inline __device__  void getInfo(T &t,int id){}
    /*I dont have any info*/
    static inline __host__ __device__ size_t sizeofInfo(){return 0;}
    /*I do not write anything to shared memory*/
    inline __device__ void fillSharedMemory(T &t, void * shInfo, int i_load){}
    inline __device__  auto compute(T &t, const int &j,
				    const real4 &posi, const real4 &posj)-> decltype(t.zero()){
      return t.compute(posi,posj);
    }
    /*I just ignore the shared memory part*/
    inline __device__ auto computeSharedMem(T &t,
					    const real4& posi, const real4 &posj,
					    void* shMem, int counter)-> decltype(t.zero()){
      return t.compute(posi,posj);

    }

  };

}
#endif
