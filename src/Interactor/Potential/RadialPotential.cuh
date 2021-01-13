/*Raul P. Pelaez 2017-2021. Radial Potential.

   RadialPotential is a base class for potentials that depend only on the distance of two particles.

   RadialPotential needs:
   -A radial potential functor (with force* and energy functions that take only distance) (See LJFunctor in  Potential.cuh)
   -Optionally, a type parameter handler (See BasicParameterHandler.cuh)

   *Notice that the force function in a RadialPotential must, in fact, return the modulus of the force divided by the distance, |f|/r.
   Potentials provide Transversers to PairForces in order to compute forces, energies and/or both at the same time. Many aspects of the Potential and Transverser interfaces are optional and provide default behavior, when a function is optional it will be denoted as such in this header.

   The Potential interface requires a given class/struct to provide the following public member functions:
   Required members:
   real getCutOff(); //The maximum cut-off the potential requires.
   ForceTransverser getForceTransverser(Box box, shared_ptr<ParticleData> pd); //Provides a Transverser that computes the force
   Optional members:
   EnergyTransverser getEnergyTransverser(Box box, shared_ptr<ParticleData> pd); //Provides a Transverser that computes the energy
   //If not present defaults to every interaction having zero energy contribution
   ForceEnergyTransverser getForceEnergyTransverser(Box box, shared_ptr<ParticleData> pd); //Provides a Transverser that computes the force and energy at the same time
   //If not present defaults to sequentially computing force and energy one after the other.
   A struct/class adhering to the Potential interface can also be ParameterUpdatable[1].

   For more information on how to code a new Potential see examples/customPotentials.cu
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

      class BaseTransverser{
      public:
	using pairParameterIterator = typename ParameterHandle::Iterator;
      protected:
	pairParameterIterator typeParameters;
	Box box;
	PotentialFunctor pot;

      public:
	BaseTransverser(pairParameterIterator tp, Box box, PotentialFunctor pot):
	  typeParameters(tp), box(box), pot(pot){}

	inline __device__ real getCutOff2BetweenTypes(int ti, int tj){
	  return this->typeParameters(ti, tj).cutOff2;
	}

      };

      struct forceTransverser: public BaseTransverser{

	forceTransverser(typename ParameterHandle::Iterator tp, real4 *force, Box box, PotentialFunctor pot):
	  BaseTransverser(tp, box, pot),
	  force(force){}

	inline __device__  real3 compute(real4 ri, real4 rj){
	  real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	  const real r2 = dot(r12, r12);
	  if(r2 == real(0.0)){
	    return real3();
	  }
	  auto params = this->typeParameters((int) ri.w, (int) rj.w);
	  return  this->pot.force(r2, params)*r12;
	}

	inline __device__ void set(int pi, real3 total){
	  force[pi] += make_real4(total);
	}

      private:
	real4 *force;
      };

      struct energyTransverser: public BaseTransverser{

	energyTransverser(typename ParameterHandle::Iterator tp, real *energy, Box box, PotentialFunctor pot):
	  BaseTransverser(tp, box, pot),
	  energy(energy){}

	inline __device__  real compute(real4 ri, real4 rj){
	  real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	  auto params = this->typeParameters((int) ri.w, (int) rj.w);
	  const real r2 = dot(r12, r12);
	  if(r2 == real(0.0)){
	    return real(0.0);
	  }
	  return this->pot.energy(r2, params);
	}

	inline __device__ void set(int pi, real total){
	  energy[pi] += total;
	}

      private:
	real* energy;

      };

      struct ForceEnergyTransverser: public BaseTransverser{

	ForceEnergyTransverser(typename ParameterHandle::Iterator tp, real4 *force, real *energy, Box box, PotentialFunctor pot):
	  BaseTransverser(tp, box, pot),
	  force(force), energy(energy){}

	inline __device__  real4 compute(real4 ri, real4 rj){
	  real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	  auto params = this->typeParameters((int) ri.w, (int) rj.w);
	  const real r2 = dot(r12, r12);
	  if(r2 == real(0.0)){
	    return real4();
	  }
	  real E = this->pot.energy(r2, params);
	  real3 F = this->pot.force(r2, params)*r12;
	  return make_real4(F, E);
	}

	inline __device__ void set(int pi, real4 total){
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
