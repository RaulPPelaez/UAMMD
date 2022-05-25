/*Raul P.Pelaez 2017-2021. Smoothed Particle Hydrodynamics

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


TODO:
100- Optimize

References:
[1] Smoothed particle hydrodynamics. JJ Monaghan. Rep. Prog. Phys. 68 (2005) 1703–1759 doi:10.1088/0034-4885/68/8/R01

 */
#include"SPH.cuh"
#include<third_party/uammd_cub.cuh>

#include"utils/GPUUtils.cuh"
#include"SPH/Kernel.cuh"

namespace uammd{
  SPH::SPH(shared_ptr<ParticleGroup> pg, Parameters par):
    Interactor(pg, "SPH/"),
    box(par.box),
    support(par.support), viscosity(par.viscosity),
    gasStiffness(par.gasStiffness), restDensity(par.restDensity),
    nl(par.nl)
  {
    sys->log<System::MESSAGE>("[SPH] Initialized.");
    if(pg->getNumberParticles() != pd->getNumParticles()){
      sys->log<System::CRITICAL>("[SPH] Not compatible with groups yet!.");
    }
  }


  SPH::~SPH(){
    sys->log<System::MESSAGE>("[SPH] Destroyed.");
  }
  namespace SPH_ns{

    //This is a Transverser, see NeighbourList.cuh, PairForces.cuh or NBodyForces.cuh for a guide on this concept
    //Computes the density on each particle
    template<class Kernel>
    struct DensityTransverser{
      Kernel kernel;
      Box box;
      real *density;
      real support;
      real *mass;
      DensityTransverser(Box box, real *density, real *mass, real support):
	box(box),density(density),mass(mass),support(support){ }

      inline __device__ real  getInfo(int i){
	if(mass) return mass[i];
	else return real(1.0);
      }
      //Starting density of each particle
      inline __device__ real zero(){ return real(0);}

      //rho_i = sum_j( mj · Wij)
      inline __device__ real compute(real4 ri, real4 rj, real massi, real massj){
	real3 rij = box.apply_pbc(make_real3(rj) - make_real3(ri));
	return massj*kernel(rij, support);
      }
      //Sum for each particle
      inline __device__ void accumulate(real &total, real current){ total += current; }
      //Write the result
      inline __device__ void set(int i, const real &total){ density[i] = total; }
    };

    //Compute Pressure, this simple functor codes the ideal gas state eq. Pi = K·(rho_i - rho0)
    struct Density2Pressure{
      real gasStiffness, restDensity;
      Density2Pressure(real gasStiffness, real restDensity):
	gasStiffness(gasStiffness),
	restDensity(restDensity){}

      inline __device__ real operator()(real density_i) const{
	return gasStiffness*(density_i - restDensity);
      }
    };

    //Computes artificial viscosity from vij and rij: vis = nu·(vj·rij)/(rij^2+epsilon·h^2)
    struct ArtificialViscosity{
      real viscosityPrefactor, epsilon;
      ArtificialViscosity(real pre, real e=0.001): viscosityPrefactor(-pre), epsilon(e){}

      inline __device__ real operator()(real3 vij, real3 rij, real support) const{
	return viscosityPrefactor*(dot(vij, rij)/(dot(rij, rij)+epsilon*support*support));
      }
    };

    //Compute and sum the force on each particle:
    // Fi = sum_j( mi·mj·(Pi/rho_i^2 + Pj/rho_j^2)· rij · W(rij)
    template<class Kernel>
    struct ForceTransverser{
      Kernel kernel; //Interpolation kernel
      real support; //Support distance of the kernel
      Box box;
      real4 *force;
      real3 * vel;
      real *density, *pressure, *mass;
      ArtificialViscosity viscosity; //Computes the viscosity term

      //The per particle information requested
      struct InfoType{
	real Pdivrhosq;
	real mass;
	real3 vel;
      };

      ForceTransverser(Box box,
		       real4 *force,
		       real3* vel,
		       real *density,
		       real *pressure,
		       real *mass,
		       ArtificialViscosity viscosity,
		       real support):
	box(box), force(force),	vel(vel), mass(mass), density(density),
	pressure(pressure), viscosity(viscosity), support(support){   }


      //Retrieve information from particle i
      inline __device__ InfoType getInfo(int i){
	InfoType info;
	if(mass) info.mass = mass[i];
	else info.mass = real(1.0);
	real rho = density[i];
	info.Pdivrhosq = pressure[i]/(rho*rho);
	info.vel = vel[i];
	return info;
      }
      //Starting force
      inline __device__ real3 zero(){ return make_real3(0);}

      //Compute Fij
      inline __device__ real3 compute(real4 ri, real4 rj, InfoType infoi, InfoType infoj){
	real3 rij = box.apply_pbc(make_real3(rj) - make_real3(ri));
        real3 vij = infoj.vel - infoi.vel;
	const real mi = infoi.mass;
	const real mj = infoj.mass;
	const real vis = viscosity(vij, rij, support);
	return mi*mj*(infoi.Pdivrhosq + infoj.Pdivrhosq + vis)*kernel.gradient(rij, support);
      }

      inline __device__ void accumulate(real3 &total, real3 current){ total += current; }
      //Write result
      inline __device__ void set(int i, const real3 &total){ force[i] += make_real4(total); }

    };



  }

  void SPH::sum(Interactor::Computables comp, cudaStream_t st){
    sys->log<System::DEBUG1>("[SPH] Summing forces");
    int numberParticles = pg->getNumberParticles();
    using Kernel = SPH_ns::Kernel::M4CubicSpline;
    if(!nl){
      nl = std::make_shared<NeighbourList>(pg);
    }
    real rcut = Kernel::getCutOff(support);
    sys->log<System::DEBUG3>("[SPH] Using cutOff: %f", rcut);
    //Update neighbour list.
    nl->update(box, rcut, st);
    sys->log<System::DEBUG3>("[SPH] Computing density");
    density.resize(numberParticles);
    auto d_density = thrust::raw_pointer_cast(density.data());
    //If mass is not allocated assume all masses are 1
    real *d_mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read).raw();
    auto densityTrans = SPH_ns::DensityTransverser<Kernel>(box, d_density, d_mass, support);
    //Compute Density
    nl->transverseList(densityTrans, st);
    sys->log<System::DEBUG3>("[SPH] Computing Pressure");
    pressure.resize(numberParticles);
    //Compute Pressure
    thrust::transform(thrust::cuda::par.on(st),
		      d_density, d_density + numberParticles,
		      pressure.begin(),
		      SPH_ns::Density2Pressure(gasStiffness, restDensity));
    sys->log<System::DEBUG3>("[SPH] Computing Force");
    auto d_pressure = thrust::raw_pointer_cast(pressure.data());
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
    auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
    SPH_ns::ArtificialViscosity vis(viscosity, 0.001);
    auto forceTrans = SPH_ns::ForceTransverser<Kernel>(box, force.raw(), vel.raw(),
						       d_density, d_pressure,
						       d_mass,
						       vis,
						       support);
    //Compute Force
    nl->transverseList(forceTrans, st);
  }

}
