/* Raul P. Pelaez 2018-2021. Dissipative Particle Dynamics potential.

   A DPD simulation can be seen as a regular Molecular Dynamics simulation with a special interaction force between particles [1].
   This file implements a Potential entity that when used with a VerletNVE integrator (through a PairForces interactor) will produce a DPD simulation.

References:
[1] On the numerical treatment of dissipative particle dynamics and related systems. Leimkuhler and Shang 2015. https://doi.org/10.1016/j.jcp.2014.09.008
 */
#pragma once
#include "System/System.h"
#include "ParticleData/ParticleData.cuh"
#include "misc/ParameterUpdatable.h"
#include "third_party/saruprng.cuh"
#include "utils/Box.cuh"
#include"utils/TransverserUtils.cuh"
#include"third_party/type_names.h"

namespace uammd{
  namespace Potential{
    struct DefaultDissipation{
      real gamma;
      __device__ __host__ DefaultDissipation():DefaultDissipation(1.0){}

      __device__ __host__ DefaultDissipation(real gamma):gamma(gamma){}

      template<class ...T> __device__ __host__ real dissipativeStrength(T...) const{
	return gamma;
      }

      template<class T>
      inline __device__ __host__ DefaultDissipation operator=(T gamma){return DefaultDissipation(gamma);}

    };

    template<class DissipativeStrength = DefaultDissipation>
    class DPD_impl: public ParameterUpdatable{
    protected:
      int step;
      real rcut;
      DissipativeStrength gamma; //Dissipative force strength
      real temperature;
      real sigma; //Random force strength, must be such as sigma = sqrt(2*kT*gamma)/sqrt(dt)
      real dt;
      real A; //Maximum repulsion between a pair for the conservative force
    public:
      struct Parameters{
	real cutOff = 1;
	real dt = 0;
	DissipativeStrength gamma; //1.0 by default
	real temperature;
	real A = 1;
      };

      //The system parameter is deprecated and actually unused, the
      // other constructor is left here for retrocompatibility
      DPD_impl(Parameters par): DPD_impl(nullptr, par){}

      DPD_impl(shared_ptr<System> sys, Parameters par):
        rcut(par.cutOff), dt(par.dt), gamma(par.gamma), temperature(par.temperature), A(par.A){
	System::log<System::MESSAGE>("[Potential::DPD] Created");
	step = 0;
	sigma = sqrt(2.0*temperature)/sqrt(dt);
	System::log<System::MESSAGE>("[Potential::DPD] Temperature: %f", temperature);
	System::log<System::MESSAGE>("[Potential::DPD] Cut off: %f", rcut);
	System::log<System::MESSAGE>("[Potential::DPD] aij: %f", A);
	printGamma();
      }
      void printGamma();

      ~DPD_impl(){
	System::log<System::MESSAGE>("[Potential::DPD] Destroyed");
      }

      real getCutOff(){ return rcut;}

      virtual void updateTemperature(real newTemp) override{
	temperature = newTemp;
	sigma = sqrt(2.0*temperature)/sqrt(dt);
      }

      virtual void updateTimeStep(real newdt) override{
	dt = newdt;
	sigma = sqrt(2.0*temperature)/sqrt(dt);
      }


      struct ForceTransverser{
      private:
	real4* pos;
	real3 * vel;
	real4 * force;
	Box box;
	ullint seed; //A random seed
	ullint step; //Current time step
	int N;
	real invrcut;
	//DPD force parameters
	DissipativeStrength gamma;
	real sigma;
	real A;

      public:
	ForceTransverser(real4* pos, real3 *vel, real4* force,
			 ullint seed, ullint step,
			 Box box,
			 int N,
			 real rcut,
			 DissipativeStrength gamma, real sigma, real A):
	  pos(pos), vel(vel), force(force),
	  step(step), seed(seed),
	  box(box),
	  N(N),
	  invrcut(1.0/rcut), gamma(gamma), sigma(sigma), A(A){}

	using returnInfo = real3;

	struct Info{
	  real3 vel;
	  int id;
	};

	inline __device__ returnInfo compute(const real4 &pi, const real4 &pj, const Info &infoi, const Info &infoj){
	  real3 rij = box.apply_pbc(make_real3(pi) - make_real3(pj));
	  real3 vij = make_real3(infoi.vel) - make_real3(infoj.vel);
	  //The random force must be such as Frij = Frji, we achieve this by seeding the RNG the same for pairs ij and ji
	  int i = infoi.id;
	  int j = infoj.id;
	  if(i>j) thrust::swap(i,j);
	  const int ij = i + N*j;
	  Saru rng(ij, seed, step);
	  const real rmod = sqrt(dot(rij,rij));
	  //There is an indetermination at r=0
	  if(rmod == real(0)) return make_real3(0);
	  const real invrmod = real(1.0)/rmod;
	  //The force is 0 beyond rcut
	  if(invrmod<=invrcut) return make_real3(0);
	  const real wr = real(1.0) - rmod*invrcut; //This weight function is arbitrary as long as wd = wr*wr
	  const real Fc = A*wr*invrmod;
	  const real wd = wr*wr; //Wd must be such as wd = wr^2 to ensure fluctuation dissipation balance
	  const real g = gamma.dissipativeStrength(i, j, pi, pj, infoi.vel, infoj.vel);
	  const real Fd = -g*wd*invrmod*invrmod*dot(rij, vij);
	  const real Fr = rng.gf(real(0.0), sigma*sqrt(g)*wr*invrmod).x;
	  return (Fc+Fd+Fr)*rij;
	}

	inline __device__ Info getInfo(int pi){return  {vel[pi], pi};}

	inline __device__ void set(uint pi, const returnInfo &total){ force[pi] += make_real4(total);}

      };

      ForceTransverser getForceTransverser(Box box, shared_ptr<ParticleData> pd){
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto vel = pd->getVel(access::location::gpu, access::mode::read);
	auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
	static auto seed = pd->getSystem()->rng().next();
	step++;
	int N = pd->getNumParticles();
	return ForceTransverser(pos.raw(), vel.raw(), force.raw(), seed, step, box, N, rcut, gamma, sigma, A);
      }
      //Notice that no getEnergyTransverser is present, this is not a problem as modules using this potential will fall back to a BasicNullTransverser when the method getEnergyTransverser is not found and the energy will not be computed altogether.

      BasicNullTransverser getForceEnergyTransverser(Box box, shared_ptr<ParticleData> pd){
	System::log<System::CRITICAL>("[DPD] No way of measuring energy in DPD");
	return BasicNullTransverser();
      }

    };

    template<>
    void DPD_impl<DefaultDissipation>::printGamma(){
      System::log<System::MESSAGE>("[Potential::DPD] gamma: %f", gamma.gamma);
    }

    template<class T>
    void DPD_impl<T>::printGamma(){
      System::log<System::MESSAGE>("[Potential::DPD] Using %s for dissipation", type_name<T>().c_str());
    }

    using DPD = DPD_impl<>;
  }

}
