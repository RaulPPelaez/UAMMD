/*Raul P. Pelaez 2017. Radial Potential.

  A potential must provide:
   -transversers to compute force, energy and virial through get*Transverser(Box box, shared_ptr<ParticleData> pd)
   -A maximum interaction distance (can be infty) via a method real getCutOff();
   -Handle particle types (with setPotParameters(i, j, InputParameters p)
   

  RadialPotential is a base class for potentials that depend only on the distance of two particles.

  RadialPotential needs:
      -A radial potential functor (with force and energy functions that take only distance) (See LJFunctor in  Potential.cuh)
      -A type parameter handler (See BasicParameterHandler.cuh)
      

  You can copy paste this file to create more complex potentials (ones that depend on any arbitrary particle property)
  



 */
#ifndef RADIALPOTENTIAL_CUH
#define RADIALPOTENTIAL_CUH

#include"ParticleData/ParticleData.cuh"
#include"utils/Box.cuh"
#include"PotentialBase.cuh"
#include"ParameterHandler.cuh"
#include"third_party/type_names.h"
namespace uammd{

  namespace Potential{


    //For the three computations, there are only two differences,
    // the type of the return type, and the compute function
    template<Mode mode>   struct returnType_impl;
    template<> struct returnType_impl<Mode::FORCE >{using type = real4;};
    template<> struct returnType_impl<Mode::ENERGY>{using type = real;};
    template<> struct returnType_impl<Mode::VIRIAL>{using type = real4;};
        
    template<Mode mode> using returnType = typename returnType_impl<mode>::type;

    //General Potential for radial potentials
    template<class PotentialFunctor, class ParameterHandle = BasicParameterHandler<PotentialFunctor>>
      class Radial{
      public:
	using InputPairParameters = typename PotentialFunctor::InputPairParameters;
      protected:
	std::shared_ptr<System> sys;
	
	PotentialFunctor pot;
      shared_ptr<ParameterHandle> pairParameters;
	std::string name;
      public:
	Radial(std::shared_ptr<System> sys):Radial(sys, PotentialFunctor()){}
	Radial(std::shared_ptr<System> sys, PotentialFunctor pot): pot(pot), sys(sys){
	  name = type_name_without_namespace<PotentialFunctor>();
	  pairParameters = std::make_shared<ParameterHandle>();
	  sys->log<System::MESSAGE>("[RadialPotential/%s] Initialized", name.c_str());
	}
	~Radial(){
	  sys->log<System::DEBUG>("[RadialPotential/%s] Destroyed", name.c_str());
	}

	void setPotParameters(int ti, int tj, InputPairParameters p){
	  sys->log<System::MESSAGE>("[RadialPotential/%s] Type pair %d %d parameters added", name.c_str(), ti, tj);
	  pairParameters->add(ti, tj, p);
	}
	real getCutOff(){
	  return pairParameters->getCutOff();
	}

	//Most of the code is identical for energy, force and virial transversers.
	//the force, energy and virial transverser inherit from this one. See below
	template<class resultType>
	class BaseTransverser{
	public:
	  using pairParameterIterator = typename ParameterHandle::Iterator;
	protected:
	  pairParameterIterator typeParameters;
	  resultType *result;
	  Box box;
	  PotentialFunctor pot;      
      
	public:
	  BaseTransverser(pairParameterIterator tp,
			  resultType *result,
			  Box box,
			  PotentialFunctor pot):
	    typeParameters(tp), result(result), box(box), pot(pot){}
	  
	  size_t getSharedMemorySize(){
	    return typeParameters.getSharedMemorySize();
	  }
	  
	  /*For each particle i, this function will be called for all its neighbours j with the result of compute_ij and total, with the value of the last time accumulate was called, starting with zero()*/
	  inline __device__ void accumulate(resultType& total, const resultType& current){total += current;}
	  /*This function will be called for each particle i, once when all neighbours have been transversed, with the particle index and the value total had the last time accumulate was called*/
	  /*Update the force acting on particle pi, pi is in the normal order*/
	  inline __device__ void set(int pi, const resultType &total){	   	    	    
	    result[pi] += total;
	  }
	  /*Starting value, can also be used to initialize in-kernel parameters, as it is called at the start*/
	  inline __device__ resultType zero(){
	    this->typeParameters.zero();
	    return resultType();}
	
	};

	//Only the compute function changes
	struct forceTransverser: public BaseTransverser<real4>{

	  //Inherit constructor
	  using BaseTransverser<real4>::BaseTransverser;
      
	  inline __device__  real4 compute(const real4 &ri, const real4 &rj){
	    
	    real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	    //Squared distance
	    const real r2 = dot(r12, r12);
	    if(r2 == real(0.0)) return make_real4(0);
	    
	    auto params = this->typeParameters((int) ri.w, (int) rj.w);
	    
	    return  make_real4(this->pot.force(r2, params)*r12, real(0.0));
	  }      

	};
	
	struct energyTransverser: public BaseTransverser<real>{
	  using BaseTransverser<real>::BaseTransverser;
            
	  inline __device__  real compute(const real4 &ri, const real4 &rj){
	    real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));

	    auto params = this->typeParameters((int) ri.w, (int) rj.w);
	    //Squared distance
	    const real r2 = dot(r12, r12);
	    if(r2 == real(0.0)) return real(0.0);
	    return this->pot.energy(r2, params);
	  }

      

	};

	//Create and return a transverser
	forceTransverser getForceTransverser(Box box, shared_ptr<ParticleData> pd){
	  sys->log<System::DEBUG2>("[RadialPotential/%s] ForceTransverser requested", name.c_str());
	  auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
      
	  return forceTransverser(pairParameters->getIterator(), force.raw(), box, pot);

	}

	energyTransverser getEnergyTransverser(Box box, shared_ptr<ParticleData> pd){
	  sys->log<System::DEBUG2>("[RadialPotential/%s] EnergyTransverser requested", name.c_str());
	  auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
      
	  return energyTransverser(pairParameters->getIterator(), energy.raw(), box, pot);
	}

      };

  }
}
#endif
