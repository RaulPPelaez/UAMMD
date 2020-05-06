/*Raul P. Pelaez 2017. Radial Potential.

  A potential must provide:
   -transversers to compute force, energy and virial through get*Transverser(Box box, shared_ptr<ParticleData> pd)
   -A maximum interaction distance (can be infty) via a method real getCutOff();
   -Handle particle types (with setPotParameters(i, j, InputParameters p)


   RadialPotential is a base class for potentials that depend only on the distance of two particles.

   RadialPotential needs:
   -A radial potential functor (with force* and energy functions that take only distance) (See LJFunctor in  Potential.cuh)
   -A type parameter handler (See BasicParameterHandler.cuh)

   *Notice that the force function in a RadialPotential must, in fact, return the modulus of the force divided by the distance, |f|/r.
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

      template<class resultType>
      class BaseTransverser{
      public:
	using pairParameterIterator = typename ParameterHandle::Iterator;
      protected:
	pairParameterIterator typeParameters;
	Box box;
	PotentialFunctor pot;

      public:
	BaseTransverser(pairParameterIterator tp,
			Box box,
			PotentialFunctor pot):
	  typeParameters(tp), box(box), pot(pot){}

	size_t getSharedMemorySize(){
	  return typeParameters.getSharedMemorySize();
	}

	inline __device__ void accumulate(resultType& total, const resultType& current){total += current;}


	inline __device__ resultType zero(){
	  this->typeParameters.zero();
	  return resultType();
	}

	inline __device__ real getCutOff2BetweenTypes(int ti, int tj){
	  return this->typeParameters(ti, tj).cutOff2;
	}

      };

      struct forceTransverser: public BaseTransverser<real4>{

	forceTransverser(typename ParameterHandle::Iterator tp,
			 real4 *force,
			 Box box,
			 PotentialFunctor pot):
	  BaseTransverser<real4>(tp, box, pot),
	  force(force){}

	inline __device__  real4 compute(const real4 &ri, const real4 &rj){
	  real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	  const real r2 = dot(r12, r12);
	  if(r2 == real(0.0)) return make_real4(0);
	  auto params = this->typeParameters((int) ri.w, (int) rj.w);
	  return  make_real4(this->pot.force(r2, params)*r12, real(0.0));
	}

	inline __device__ void set(int pi, const real4 &total){
	  force[pi] += total;
	}

      private:
	real4 *force;
      };

      struct energyTransverser: public BaseTransverser<real>{

	energyTransverser(typename ParameterHandle::Iterator tp,
			  real *energy,
			  Box box,
			  PotentialFunctor pot):
	  BaseTransverser<real>(tp, box, pot),
	  energy(energy){}

	inline __device__  real compute(const real4 &ri, const real4 &rj){
	  real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	  auto params = this->typeParameters((int) ri.w, (int) rj.w);
	  const real r2 = dot(r12, r12);
	  if(r2 == real(0.0)) return real(0.0);
	  return this->pot.energy(r2, params);
	}

	inline __device__ void set(int pi, const real &total){
	  energy[pi] += total;
	}

      private:
	real* energy;

      };

      struct ForceEnergyTransverser: public BaseTransverser<real4>{

	ForceEnergyTransverser(typename ParameterHandle::Iterator tp,
			       real4 *force,
			       real *energy,
			       Box box,
			       PotentialFunctor pot):
	  BaseTransverser<real4>(tp, box, pot),
	  force(force), energy(energy){}

	inline __device__  real4 compute(const real4 &ri, const real4 &rj){
	  real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	  auto params = this->typeParameters((int) ri.w, (int) rj.w);
	  const real r2 = dot(r12, r12);
	  if(r2 == real(0.0)) return real4();
	  real E = this->pot.energy(r2, params);
	  real3 F = this->pot.force(r2, params)*r12;
	  return make_real4(F, E);
	}

	inline __device__ void set(int pi, const real4 &total){
	  force[pi] += make_real4(make_real3(total), 0);
	  energy[pi] += total.w;
	}

      private:
	real* energy;
	real4* force;
      };


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

      ForceEnergyTransverser getForceEnergyTransverser(Box box, shared_ptr<ParticleData> pd){
	sys->log<System::DEBUG2>("[RadialPotential/%s] EnergyTransverser requested", name.c_str());
	auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
	auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
	return ForceEnergyTransverser(pairParameters->getIterator(), force.raw(), energy.raw(), box, pot);
      }

    };

  }
}
#endif
