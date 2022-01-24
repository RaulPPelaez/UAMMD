/*
  Raul P. Pelaez 2017-2021. Bonded Forces Interactor implementation. AKA two body bonds

  This module implements an algorithm to compute the force between two particles, or a particle and a point in space) joined a bond.

  Sends the bondList to the GPU ordered by the first particle, and two additional arrays
  storing where the information for each particle begins and ends. Identical to the sorting
  trick in CellList

  There are three types of bonds:
  -particle-particle Bonds (BondedForces)
  -particle-point Bonds (Fixed Point) (BondedForces)
  -particle-particle-particle bonds (AngularBondedForces)


  The format of the input file is the following,
  just give a list of bonds, the order doesnt matter as long as no bond is repeated:
  nbonds
  i j BONDINFO
  .
  .
  .
  nbondsFixedPoint <- Can be zero or not be in at all in the file
  i px py pz BONDINFO

  Where i,j are the indices of the particles. BONDINFO can be any number of rows, as described
  by the BondedType BondedForces is used with, see BondedType::Harmonic for an example.

  In the case of AngularBondedforces i j k are needed instead of i j. The order doesnt matter, but j must always be the central particle. See AngularBondedForces.cuh

  A bond type can be ParameterUpdatable.
*/

#ifndef BONDEDFORCES_CUH
#define BONDEDFORCES_CUH

#include"Interactor.cuh"
#include"global/defines.h"
#include<fstream>
#include<thrust/device_vector.h>
#include"third_party/type_names.h"
namespace uammd{
  struct ComputeType{
    real3 force;
    real virial;
    real energy;      
  };
    
  //Functors with different bond potentials
  namespace BondedType{
    
    namespace detail{
      inline __device__ real harmonicForceModulusDivR(real invr, real k, real r0){
	const real f = -k*(real(1.0)-r0*invr); //F = -k·(r-r0)·rvec/r
	return f;
      }

      inline __device__ real sq (real a){ return a*a;}
      inline __device__ real harmonicEnergy(real invr, real k, real r0){
	const real e = real(0.25)*k*sq(real(1.0)/invr-r0);
	return e;
      }

      inline __device__ ComputeType harmonicBond(real3 r12, Interactor::Computables comp,
						 real k, real r0){
	real r2 = dot(r12, r12);
	const real invr = rsqrt(r2);
	const real f = detail::harmonicForceModulusDivR(invr, k, r0);
	ComputeType ct;
	ct.force = f*r12;
	ct.energy = comp.energy?(detail::harmonicEnergy(invr, k, r0)):real(0.0);
	ct.virial = comp.virial?dot(ct.force,r12):real(0.0);
	return (r2==real(0.0))?(ComputeType{}):ct;
      }
    }
    //Harmonic bond for pairs of particles
    struct Harmonic{
      //Place in this struct whatever static information is needed for a given bond
      //In this case spring constant and equilibrium distance
      //the function readBond below takes care of reading each BondInfo from the file
      struct BondInfo{
	real k, r0;
      };
      //This function will be called for every bond read in the bond file and is expected to compute force/energy and or virial
      //bond_index: The index of the particle to compute force/energy/virial on
      //ids: list of indexes of the particles involved in the current bond
      //pos: list of positions of the particles involved in the current bond
      //comp: computable targets (wether force, energy and or virial are needed).
      //bi: bond information for the current bond (as returned by readBond)
      inline __device__ ComputeType compute(int bond_index,
					    int ids[2], real3 pos[2],
					    Interactor::Computables comp, BondInfo bi){
	real3 r12 = pos[1]-pos[0];
	return detail::harmonicBond(r12, comp, bi.k, bi.r0);
      }

      //This function will be called for each bond in the bond file and read the information of a bond
      //It must use the stream that is handed to it to construct a BondInfo.  
      static __host__ BondInfo readBond(std::istream &in){
	/*BondedForces will read i j, readBond has to read the rest of the line*/
	BondInfo bi;
	in>>bi.k>>bi.r0;
	return bi;
      }

    };

    //Same as Harmonic, but applies Periodic boundary conditions to the distance of a pair
    struct HarmonicPBC: public Harmonic{
      Box box;
      HarmonicPBC(Box box): box(box){}
      inline __device__ ComputeType compute(int bond_index,
					    int ids[2], real3 pos[2],
					    Interactor::Computables comp, BondInfo bi){
	real3 r12 = box.apply_pbc(pos[1]-pos[0]);
	return detail::harmonicBond(r12, comp, bi.k, bi.r0);
      }
    };

    namespace detail{
      inline __device__ real feneForceModulusDivR(real r2, real k, real r02){
	const real f = -r02*k/(r02-r2);
	return f;
      }

      inline __device__ real feneEnergy(real r2, real k, real r02){
	const real e = -real(0.25)*k*r02*log(real(1.0)-r2/r02);
	return e;
      }

      inline __device__ ComputeType feneBond(real3 r12, Interactor::Computables comp,
					     real k, real r0){
	real r2 = dot(r12, r12);
	const real f = detail::feneForceModulusDivR(r2, k, r0*r0);
	ComputeType ct;
	ct.force = f*r12;
	ct.energy = comp.energy?(detail::feneEnergy(r2, k, r0*r0)):real(0.0);
	ct.virial = comp.virial?dot(ct.force,r12):real(0.0);
	return (r2==real(0.0))?(ComputeType{}):ct;
      }
    }

    struct FENE{
      struct BondInfo{
	real r0, k;
      };

      inline __device__ ComputeType compute(int bond_index,
					    int ids[2], real3 pos[2],
					    Interactor::Computables comp, BondInfo bi){
	real3 r12 = pos[1]-pos[0];
	return detail::feneBond(r12, comp, bi.k, bi.r0);
      }      

      static BondInfo readBond(std::istream &in){
	BondInfo bi;
	in>>bi.k>>bi.r0;
	return bi;
      }

    };
    
    struct FENEPBC: public FENE{
      Box box;
      FENEPBC(Box box): box(box){}
      inline __device__ ComputeType compute(int bond_index,
					    int ids[2], real3 pos[2],
					    Interactor::Computables comp, BondInfo bi){
	real3 r12 = box.apply_pbc(pos[1]-pos[0]);
	return detail::feneBond(r12, comp, bi.k, bi.r0);
      }      

    };

  }

  template<class BondType, int particlesPerBond>
  class BondedForces: public Interactor, public ParameterUpdatableDelegate<BondType>{
  public:

    struct Parameters{
      std::string file; //File containing the bonds
    };
    //Aligning these really improves performance
    struct __align__(16)  Bond{
      int ids[particlesPerBond];
      typename BondType::BondInfo bond_info;
    };

    explicit BondedForces(shared_ptr<ParticleData> pd,
			  Parameters par,
			  std::shared_ptr<BondType> bondForce = std::make_shared<BondType>());

    ~BondedForces(){
      cudaDeviceSynchronize();
      sys->log<System::MESSAGE>("[BondedForces] Destroyed");
    }

    void sum(Computables comp, cudaStream_t st) override;

  private:
    void readBonds(std::string);
    int nbonds;
    thrust::device_vector<Bond> bondList;   //[All bonds involving the first particle with bonds, involving the second...] each bonds stores the id of the three particles in the bond. The id of the first/second... particle  with bonds is particlesWithBonds[i]
    thrust::device_vector<int> bondStart, bondEnd; //bondStart[i], Where the list of bonds of particle with bond number i start (the id of particle i is particlesWithBonds[i].
    thrust::device_vector<int> particlesWithBonds; //List of particle ids with at least one bond
    //Positions for fixed point bonds in the case of particlesPerBond==2
    thrust::device_vector<real4> fixedPointPositions;
    std::shared_ptr<BondType> bondCompute;
    int TPP; // Threads per particle
    void callComputeBonded(real4* f, real*e, real*v, cudaStream_t st);
  };
}


#include"BondedForces.cu"

#endif






