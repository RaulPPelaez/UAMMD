/*Raul P. Pelaez 2017-2021. Radial Potential.

   RadialPotential is a base class for potentials that depend only on the distance of two particles.

   RadialPotential needs:
   -A radial potential functor (with force and energy functions that take only distance) (See LJFunctor in  Potential.cuh)
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
#include"Interactor/Interactor.cuh"
namespace uammd{

  namespace Potential{
    //General Potential for radial potentials
    template<class PotentialFunctor, class ParameterHandle = BasicParameterHandler<PotentialFunctor>>
    class Radial{
    public:
      using InputPairParameters = typename PotentialFunctor::InputPairParameters;
    protected:
      std::shared_ptr<PotentialFunctor> pot;
      std::shared_ptr<ParameterHandle> pairParameters;
      std::string name;
    public:
      Radial():Radial(std::make_shared<PotentialFunctor>()){}

      Radial(std::shared_ptr<PotentialFunctor> pot): pot(pot){
	name = type_name_without_namespace<PotentialFunctor>();
	pairParameters = std::make_shared<ParameterHandle>();
	System::log<System::MESSAGE>("[RadialPotential/%s] Initialized", name.c_str());
      }

      ~Radial(){
	System::log<System::DEBUG>("[RadialPotential/%s] Destroyed", name.c_str());
      }

      void setPotParameters(int ti, int tj, InputPairParameters p){
	System::log<System::MESSAGE>("[RadialPotential/%s] Type pair %d %d parameters added", name.c_str(), ti, tj);
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

      struct Transverser: public BaseTransverser{
	Transverser(typename ParameterHandle::Iterator tp,
		    real4 *force, real *energy, real* virial,
		    Box box, PotentialFunctor pot):
	  BaseTransverser(tp, box, pot),
	  force(force), energy(energy), virial(virial){}

	inline __device__  ForceEnergyVirial compute(real4 ri, real4 rj){
	  real3 r12 = this->box.apply_pbc(make_real3(rj)-make_real3(ri));
	  auto params = this->typeParameters((int) ri.w, (int) rj.w);
	  const real r2 = dot(r12, r12);
	  if(r2 == real(0.0)){
	    return {};
	  }
	  real E = energy?this->pot.energy(r2, params):0;
	  real3 F = (force or virial)?this->pot.force(r2, params)*r12:real3();
	  real V = virial?dot(F, r12):0;
	  return {F, E, V};
	}

	inline __device__ void set(int pi, ForceEnergyVirial total){
	  if(force) force[pi] += make_real4(total.force, 0);
	  if(energy)energy[pi] += total.energy;
	  if(virial)virial[pi] += total.virial;
	}

      private:
	real* energy;
	real* virial;
	real4* force;	
      };

      auto getTransverser(Interactor::Computables comp, Box box, shared_ptr<ParticleData> pd){
	System::log<System::DEBUG2>("[RadialPotential/%s] ForceTransverser requested", name.c_str());
	auto force = comp.force?pd->getForce(access::location::gpu, access::mode::readwrite).begin():nullptr;
	auto energy = comp.energy?pd->getEnergy(access::location::gpu, access::mode::readwrite).begin():nullptr;
	auto virial = comp.virial?pd->getVirial(access::location::gpu, access::mode::readwrite).begin():nullptr;
	return Transverser(pairParameters->getIterator(), force, energy, virial, box, *pot);
      }

    };

  }
}
#endif
