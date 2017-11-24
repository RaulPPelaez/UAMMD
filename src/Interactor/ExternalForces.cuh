/*Raul P.Pelaez. 2017. External Forces Module.
  Computes a force acting on each particle due to an external potential.
  i.e harmonic confinement, gravity...
  
  Needs a functor transverser with at least one of function out of these three; force, energy, virial that takes any needed parameters and return the force, energy of virial of a given particle, i.e:

struct HarmonicWall{
  real zwall;
  real k = 0.1;
  HarmonicWall(real zwall):zwall(zwall){
  }
  
  __device__ __forceinline__ real3 force(const real4 &pos){

    return make_real3(0.0f, 0.0f, -k*(pos.z-zwall));

  }

  //If this function is not present, energy is assumed to be zero
  // __device__ __forceinline__ real energy(const real4 &pos){
    
  //   return real(0.5)*k*pow(pos.z-zwall, 2);
  // }

  std::tuple<const real4 *> getArrays(shared_ptr<ParticleData> pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }


};


ExternalForces will call getArrays expecting a list of arrays (with size pd->totalParticles() and the order of the particle properties).
It will then read for each particle index from the array list and pass the values to operator().
So the input parameters of operator() must be the same as the outputs from getArrays (but without the pointer).

Here is an example in which more properties are needed to compute the force:


struct ReallyComplexExternalForce{
  real drag;
  ReallyComplexExternalForce(real drag):drag(drag){}
  
  __device__ __forceinline__ real4 force(const real4 &pos, real3 vel, int id, real mass){
  if(id>1000)
    return make_real4(0.0f, 0.0f, -0.1f*pos.z*mass-drag*vel, 0.0f);
  else
    return make_real4(0);
  }

  std::tuple<real4*, real3*, int*, real*> getArrays(shared_ptr<ParticleData> pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto vel = pd->getVel(access::location::gpu, access::mode::read);
    auto id = pd->getId(access::location::gpu, access::mode::read);
    auto mass = pd->getMass(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw(), vel.raw(), id.raw(), mass.raw());
  }


References:
[1] https://www.murrayc.com/permalink/2015/12/05/modern-c-variadic-template-parameters-and-tuples/


TODO:
100- SFINAE force and virial

};




*/
#ifndef EXTERNALFORCES_CUH
#define EXTERNALFORCES_CUH
#include"Interactor.cuh"
#include"ParticleData/ParticleGroup.cuh"

#include"global/defines.h"
#include"third_party/type_names.h"

#include"utils/cxx_utils.h"
namespace uammd{
 
  template<class Functor>
  class ExternalForces: public Interactor, public ParameterUpdatableDelegate<Functor>{
    cudaStream_t stream;
  public:

    ExternalForces(shared_ptr<ParticleData> pd,
		   shared_ptr<ParticleGroup> pg,
		   shared_ptr<System> sys,
		   Functor tr):Interactor(pd, pg, sys,"ExternalForces/"+type_name<Functor>()),
			       tr(tr){
      this->setDelegate(&(this->tr));
    }
    //If no group is provided, a group with all particles is assumed
    ExternalForces(shared_ptr<ParticleData> pd,
		   shared_ptr<System> sys,
		   Functor tr):ExternalForces(pd, std::make_shared<ParticleGroup>(pd, sys), sys, tr){
    }

    ~ExternalForces(){
    }
  
    void sumForce(cudaStream_t st) override; /*implemented below*/   
    real sumEnergy() override;

    
    void print_info(){
      sys->log<System::MESSAGE>("[ExternalForces] Transversing with: %s", type_name<Functor>());
    }
  
  private:
    Functor tr;
  };


  namespace ExternalForces_ns{
    //Variadic template magic, these two functions allow to transform a tuple into a list of comma separated
    // arguments to a function.
    //It also allows to modify each element.
    //In this case, the pointers to the ith element in each array are dereferenced and passed fo f's () operator
    //Learn about this in [1]
    template<class Functor, class ...T, size_t ...Is>
    __device__ inline real3 unpackTupleAndCallForce_impl(Functor &f,
							 std::tuple<T...> &arrays,
							 int i, //Element of the array to dereference 
							 //Nasty variadic template trick
							 index_sequence<Is...>){
      return f.force(*(std::get<Is>(arrays)+i)...);
      
    }

    //This function allows to call unpackTuple hiding the make_index_sequence trick
    template<class Functor, class ...T>
    __device__ inline real3 unpackTupleAndCallForce(Functor &f, int i, std::tuple<T...> &arrays){
      constexpr int n= sizeof...(T); //Number of elements in tuple (AKA arguments in () operator in Functor)
      return unpackTupleAndCallForce_impl(f, arrays, i, make_index_sequence<n>());
    }


    //For each particle, calls the () operator of Functor f with the corresponding elements of the provided arrays
    template<class Functor, class ...T>
    __global__ void computeGPU(Functor f,
			       int numberParticlesInGroup,
			       ParticleGroup::IndexIterator groupIterator,
			       real4 * __restrict__ force,
			       std::tuple<T...>  arrays){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=numberParticlesInGroup) return;
      //Take my particle index
      const int myParticleIndex = groupIterator[id];

      //Sum the result of calling f(<all the requested input>) to my particle's force
      const real3 F = make_real3(force[myParticleIndex]) + unpackTupleAndCallForce(f, myParticleIndex, arrays);
      force[myParticleIndex] = make_real4(F); 
    }


  }
  template<class Functor>
  void ExternalForces<Functor>::sumForce(cudaStream_t st){
    sys->log<System::DEBUG2>("[ExternalForces] Computing forces...");
    
    int numberParticles = pg->getNumberParticles();
    
    int blocksize = 128;
    int Nthreads = blocksize<numberParticles?blocksize:numberParticles;
    int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);    

    auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
    auto groupIterator = pg->getIndexIterator(access::location::gpu);

    ExternalForces_ns::computeGPU<<<Nblocks, Nthreads, 0, st>>>(tr, numberParticles,
								groupIterator, force.raw(), tr.getArrays(pd));
  }







  namespace ExternalForces_ns{


    namespace SFINAE{
      template <typename T>
      class has_energy
      {
	typedef char one;
	typedef long two;

	template <typename C> static one test( decltype(&C::energy) ) ;
	template <typename C> static two test(...);

      public:
	enum { value = sizeof(test<T>(0)) == sizeof(char) };
      };

      template<class T, bool general = has_energy<T>::value>  struct energyDelegator;

      
      template<class T> struct energyDelegator<T, true>{
	template<class ...args>
	static inline __device__ __host__ real energy(T &f, args... t){return f.energy(t...);}
      };


      template<class T> struct energyDelegator<T, false>{
	template<class ...args>
	static inline __device__ __host__ real energy(T &f, args... t){return real(0);}
      };



    }
    //Variadic template magic, these two functions allow to transform a tuple into a list of comma separated
    // arguments to a function.
    //It also allows to modify each element.
    //In this case, the pointers to the ith element in each array are dereferenced and passed fo f's () operator
    //Learn about this in [1]
    template<class Functor, class ...T, size_t ...Is>
    __device__ inline real unpackTupleAndCallEnergy_impl(Functor &f,
						    std::tuple<T...> &arrays,
						    int i, //Element of the array to dereference 
						    //Nasty variadic template trick
						    index_sequence<Is...>){
      return SFINAE::energyDelegator<Functor>().energy(f, *(std::get<Is>(arrays)+i)...);
      
    }

    //This function allows to call unpackTuple hiding the make_index_sequence trick
    template<class Functor, class ...T>
    __device__ inline real unpackTupleAndCallEnergy(Functor &f, int i, std::tuple<T...> &arrays){
      constexpr int n= sizeof...(T); //Number of elements in tuple (AKA arguments in () operator in Functor)
      return unpackTupleAndCallEnergy_impl(f, arrays, i, make_index_sequence<n>());
    }


    //For each particle, calls the () operator of Functor f with the corresponding elements of the provided arrays
    template<class Functor, class ...T>
    __global__ void computeEnergyGPU(Functor f,
			       int numberParticlesInGroup,
			       ParticleGroup::IndexIterator groupIterator,
			       real * __restrict__ energy,
			       std::tuple<T...>  arrays){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=numberParticlesInGroup) return;
      //Take my particle index
      const int myParticleIndex = groupIterator[id];

      //Sum the result of calling f(<all the requested input>) to my particle's force
      energy[myParticleIndex] += unpackTupleAndCallEnergy(f, myParticleIndex, arrays);
    }


  }

  
  template<class Functor>
  real ExternalForces<Functor>::sumEnergy(){
    sys->log<System::DEBUG2>("[ExternalForces] Computing forces...");
    
    int numberParticles = pg->getNumberParticles();
    
    int blocksize = 128;
    int Nthreads = blocksize<numberParticles?blocksize:numberParticles;
    int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);    

    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
    auto groupIterator = pg->getIndexIterator(access::location::gpu);

    ExternalForces_ns::computeEnergyGPU<<<Nblocks, Nthreads>>>(tr, numberParticles,
							       groupIterator, energy.raw(), tr.getArrays(pd));

    
    return 0;
  }
    


}

#endif