/*Raul P. Pelaez 2018-2021. Some utilities for working with Transversers.

 */

#ifndef TRANSVERSERUTILS_CUH
#define TRANSVERSERUTILS_CUH
#include"global/defines.h"
#include"utils/cxx_utils.h"
namespace uammd{
  //Defines the most simple and general posible transverser that does nothing.
  //This allows to fall back to nothing when a transverser is required.
  //Just does nothing, every function has an unspecified number of arguments. Very general.
  struct BasicNullTransverser{
    template<class ...T> static constexpr inline __host__ void prepare(T...){}
    template<class ...T> static constexpr inline __host__ __device__ int zero(T...){ return 0;}
    template<class ...T> static constexpr inline __host__ __device__ int compute(T...){ return 0;}
    template<class ...T> static inline __host__ __device__ void accumulate(T...){}
    template<class ...T> static inline __host__ __device__ void set(T...){}
  };


  namespace SFINAE{
    // For Transversers, detects if a Transverser has getSharedMemorySize, therefore needing to allocate extra shared memory when launching a kernel that involves it.
    SFINAE_DEFINE_HAS_MEMBER(getSharedMemorySize)

    //This delegator allows to ask a Transverser for the shared memory it needs even when getSharedMemorySize() is not present (it will return 0 by default)
    template<class T, bool general = has_getSharedMemorySize<T>::value> struct SharedMemorySizeDelegator;
    template<class T> struct SharedMemorySizeDelegator<T, true>{
      static inline size_t getSharedMemorySize(T& t){ return t.getSharedMemorySize();}
    };
    template<class T> struct SharedMemorySizeDelegator<T, false>{
      static constexpr inline size_t getSharedMemorySize(const T& t){return 0;}
    };

    //Detects if a type has a member called energy.
    //Use with: SFINAE::has_energy<T>::value
    SFINAE_DEFINE_HAS_MEMBER(energy)

    //If an object has an energy member function it calls that function, otherwise returns real(0.0)
    template<class T, bool general = has_energy<T>::value> struct EnergyDelegator;

    template<class T> struct EnergyDelegator<T, true>{
      template<class ...Types> static inline __device__ __host__ real energy(T &t, Types... args){return t.energy(args...);}
    };

    template<class T> struct EnergyDelegator<T, false>{
      template<class ...Types> static constexpr inline __device__ __host__ real energy(T &t, Types... args){return real(0.0); }
    };

    //Detects if a type has a member called force.
    //Use with: SFINAE::has_force<T>::value
    SFINAE_DEFINE_HAS_MEMBER(force)

    //If an object has an force member function it calls that function, otherwise returns {0,0,0}
    template<class T, bool general = has_force<T>::value> struct ForceDelegator;

    template<class T> struct ForceDelegator<T, true>{
      template<class ...Types> static inline __device__ __host__ auto force(T &t, Types... args){return t.force(args...);}
    };

    template<class T> struct ForceDelegator<T, false>{
      template<class ...Types> static constexpr inline __device__ __host__ real3 force(T &t, Types... args){return real3(); }
    };

    //Detects if a type has a member called compute.
    //Use with: SFINAE::has_energy<T>::value
    SFINAE_DEFINE_HAS_MEMBER(compute)

    //If an object has an energy member function it calls that function, otherwise returns real(0.0)
    template<class T, bool general = has_compute<T>::value> struct ComputeDelegator;

    template<class T> struct ComputeDelegator<T, true>{
      template<class ...Types> static inline __device__ __host__ void compute(T &t, Types... args){t.compute(args...);}
    };

    template<class T> struct ComputeDelegator<T, false>{
      template<class ...Types> static constexpr inline __device__ __host__ void compute(T &t, Types... args){}
    };

    //Detects if a type has a member called sum.
    //Use with: SFINAE::has_energy<T>::value
    SFINAE_DEFINE_HAS_MEMBER(sum)

    //If an object has an energy member function it calls that function, otherwise returns real(0.0)
    template<class T, bool general = has_sum<T>::value> struct SumDelegator;

    template<class T> struct SumDelegator<T, true>{
      template<class ...Types> static inline __device__ __host__ void sum(T &t, Types... args){t.sum(args...);}
    };

    template<class T> struct SumDelegator<T, false>{
      template<class ...Types> static constexpr inline __device__ __host__ void sum(T &t, Types... args){}
    };


    
    //For Transversers, detects if a Transverser has getInfo, therefore being a general Transverser
    //This macro defines a tempalted struct has_getInfo<T> that contains a static value=0 if T has not a getInfo method, and 1 otherwise
    SFINAE_DEFINE_HAS_MEMBER(getInfo);


    //Delegator allows to use a transverser as if it had a getInfo function even when it does not provide one
    //A Transverser that provides a getInfo function and whose compute function takes 4 arguments
    //  (posi, posj, infoi, infoj, being info the return type of getInfo) is called a general transverser
    template<class T, bool general = has_getInfo<T>::value> class Delegator;

    //Specialization for general transversers,
    //    In this case, delegator holds the value of infoi,
    //    so it can be stored only when we have a general Transverser
    //    Additionally, it supports shared memory reads, so you can use a delegator
    //    even if the kernel writes/reads infoj to/from shared memory, see an example in NBody.cuh

    template<class T>
    class Delegator<T, true>{
    public:
      //Return type of getInfo
      using InfoType = decltype(((T *)nullptr)->getInfo(0));

      //In this case getInfo just calls the transversers getInfo
      inline __device__ void getInfo(T &t, int id){
	infoi = t.getInfo(id);
      }

      //Size in bytes of the type of myType
      static constexpr inline __host__ __device__ size_t sizeofInfo(){
        return sizeof(InfoType);
      }

      //Write the info of particle i_load to shared memory
      static inline __device__ void fillSharedMemory(T &t, void * shInfo, int i_load){
        static_cast<InfoType *>(shInfo)[threadIdx.x] = t.getInfo(i_load);
      }
      //Calls getInfo(j) and compute with 4 arguments, returns the type of zero(),  i.e the quantity
      inline __device__ auto compute(T &t, int j, real4 posi, real4 posj){
    	return t.compute(posi, posj, infoi, t.getInfo(j));
	//return t.compute(posi, posj, infoi, infoi);
      }
      //Same as above, but take infoj from the element counter of a shared memory array of type myType
      inline __device__ auto computeSharedMem(T &t, real4 posi, real4 posj, void* shMem, int counter){
    	return t.compute(posi, posj, infoi, static_cast<InfoType *>(shMem)[counter]);
      }

      InfoType infoi;
    };
    //When we have a simple transverser, getInfo does nothing, and compute has just two arguments.
    template<class T>
    class Delegator<T, false>{
    public:
      using InfoType = char;
      static inline __device__  void getInfo(T &t, int id){}
      //I dont have any additional info
      static constexpr inline __host__ __device__ size_t sizeofInfo(){return 0;}
      //I do not write anything to shared memory
      static inline __device__ void fillSharedMemory(T &t, void * shInfo, int i_load){}

      static constexpr inline __device__  auto compute(T &t, int j, real4 posi, real4 posj){
    	return t.compute(posi, posj);
      }
      //I fall back to regular compute
      static constexpr inline __device__ auto computeSharedMem(T &t, real4 posi, real4 posj, void* shMem, int counter){
    	return t.compute(posi, posj);
      }

    };


    SFINAE_DEFINE_HAS_MEMBER(accumulate)
    SFINAE_DEFINE_HAS_MEMBER(zero)
    SFINAE_DEFINE_HAS_MEMBER(prepare)

    //This class allows to use any Transverser as one that implements every possible function of the interface even when some are missing. For example if used via this class, you can assume every Transverser has a "zero()" function which, if not present will return a zero initialized value.
    template<class Transverser>
    class TransverserAdaptor: public Delegator<Transverser>{
      template<class T, enable_if_t<has_zero<T>::value> * = nullptr>
      __device__ static inline auto callzero(T &t){
	return t.zero();
      }

      template<class T, enable_if_t<not has_zero<T>::value> * = nullptr>
      __device__ static inline auto callzero(T &t){
	using ComputeType = decltype(((Delegator<T>*)nullptr)->compute(t, int(), real4(),real4()));
	return ComputeType();
      }

      template<class T, class T2>
      __device__ static inline enable_if_t<has_accumulate<T>::value> callaccumulate(T &t, T2 &total, const T2& current){
       	return t.accumulate(total, current);
      }

      template<class T, class T2>
      __device__ static inline enable_if_t<not has_accumulate<T>::value> callaccumulate(T &t, T2 &total, const T2& current){
	total = total + current;
      }

      template<class T, class ...T2>
      __host__ static inline enable_if_t<has_prepare<T>::value> callprepare(T &t, T2...args){
       	return t.prepare(args...);
      }

      template<class T, class ...T2>
      __host__ static inline enable_if_t<not has_prepare<T>::value> callprepare(T &t, T2...args){
      }

    public:
      template<class ...T> static __host__ void prepare(Transverser &tr, T...args){
	callprepare(tr, args...);
      }

      static __device__ inline auto zero(Transverser &tr){
	return callzero(tr);
      }

      template<class T> static __device__ inline void accumulate(Transverser &tr, T &total, const T &current){
	callaccumulate(tr, total, current);
      }
    };
  }
}
#endif
