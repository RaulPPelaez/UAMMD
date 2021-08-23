/*Raul P.Pelaez 2017. Smoothed Particle Hydrodynamics

  SPH is an Interactor module, any simplectic integrator can be used with it (i.e VerletNVE).

  Computes a force on each particle as:
  Fi = sum_j[  mj·(Pj/rhoj^2 + Pi/rhoi^2 + visij)·grad_i(Wij) ]

  Where:
    j: The neighbours of i (within a distance given by the support of Wij)

    m: mass
    P: Pressure
    rho: Density
    vis: Artificial viscosity

    W: Interpolation kernel, a smooth decaying function with a close support. See SPH_ns::Kernel

  The density on a given particle i is interpolated from its neighbours as:
   rho_i = sum_j[ mj·Wij ]

  The Pressure is given by an equation-of-state depending on interpolated properties of the particles. Currently:
    Pi = K·(rho_i-rho0)

  An artificial viscosity is introduced to allow shock phenomena ans stabilize the algorithm.
   visij = -nu( vij·rij)/(rij^2+epsilon)
    epsilon ~ 0.001
    v: velocity
    r: particle position

References:
[1] Smoothed particle hydrodynamics. JJ Monaghan. Rep. Prog. Phys. 68 (2005) 1703–1759 doi:10.1088/0034-4885/68/8/R01

 */

#ifndef SPH_CUH
#define SPH_CUH

#include"Interactor/Interactor.cuh"
#include"Interactor/NeighbourList/VerletList.cuh"
#include"third_party/type_names.h"

namespace uammd{
  class SPH: public Interactor{
  public:
    using NeighbourList = VerletList;
    struct Parameters{
      Box box;
      real support = 1.0;
      real viscosity = 50.0;
      real gasStiffness = 100.0;
      real restDensity = 0.4;
      std::shared_ptr<NeighbourList> nl = nullptr;
    };
    SPH(shared_ptr<ParticleData> pd,
	shared_ptr<ParticleGroup> pg,
	shared_ptr<System> sys,
	Parameters par);

    SPH(shared_ptr<ParticleData> pd,
	shared_ptr<System> sys,
	Parameters par):
      SPH(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, par){
    }
    ~SPH();

    virtual void sum(Interactor::Computables comp, cudaStream_t st) override;

  private:
    shared_ptr<NeighbourList> nl;
    Box box;
    real support;
    real gasStiffness;
    real restDensity;
    real viscosity;

    thrust::device_vector<real> density, pressure;

  };

}

#include"SPH.cu"

#endif

