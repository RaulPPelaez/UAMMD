/*
  Raul P. Pelaez 2016. Bonded Forces Interactor implementation. AKA two body bonds

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
  //Functors with different bond potentials
  namespace BondedType{
    //BondedForces needs a functor that computes the force of a pair, 
    // you can implement a new one in the input file and pass it as template argument to BondedForces
    
    //Harmonic bond, a good example on how to implement a bonded force
    struct Harmonic{
      /*Needs a struct called BondInfo with 
	the parameters that characterize a bond*/
      struct BondInfo{
	real k, r0;
      };
      /*A device function called force with these arguments that returns the force for a given bond.
	Note that this function will be called for ij and ji*/
      /*In the case of a Fixed Point bond, j will be -1*/
      //i,j: id of particles in bond
      //r12: ri-rj
      //bi: bond information.
      inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
	real r2 = dot(r12, r12);
	if(r2==real(0.0)) return make_real3(0.0);
      
	real invr = rsqrt(r2);
	real f = -bi.k*(real(1.0)-bi.r0*invr); //F = -k·(r-r0)·rvec/r
	return f*r12;
      }
      /*A function called readbond that reads a bond from in (the bond file).
	This function will be called for every line in the file except for the first*/
      static __host__ BondInfo readBond(std::istream &in){
	/*BondedForces will read i j, readBond has to read the rest of the line*/
	BondInfo bi;
	in>>bi.k>>bi.r0;
	return bi;
      }

      inline __device__ real energy(int i, int j, const real3 &r12, const BondInfo &bi){
	real r2 = dot(r12, r12);
	if(r2==real(0.0)) return real(0.0);

	real invr = rsqrt(r2);
	const real dr = real(1.0)-bi.r0*invr;
      
	return real(0.5)*bi.k*dr*dr;
      }
    
    };
    //Same as Harmonic, but applies Periodic boundary conditions to the distance of a pair
    struct HarmonicPBC: public Harmonic{
      Box box;
      HarmonicPBC(Box box): box(box){}
      inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
	return Harmonic::force(i, j, box.apply_pbc(r12), bi);
      }
      
      inline __device__ real energy(int i, int j, const real3 &r12, const BondInfo &bi){
	return Harmonic::energy(i, j, box.apply_pbc(r12), bi);
      }
    };

    struct FENE{
      struct BondInfo{
	real r0, k;
      };
      inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
	real r2 = dot(r12, r12);
	real r02 = bi.r0*bi.r0;
    
	return -r02*bi.k/(r02-r2)*r12; 
      }    
      inline __device__ real energy(int i, int j, const real3 &r12, const BondInfo &bi){
	real r2 = dot(r12, r12);
	real r02 = bi.r0*bi.r0;
    
	return -r02*bi.k/(r02-r2); 
      }

    
      static BondInfo readBond(std::istream &in){
	BondInfo bi;
	in>>bi.k>>bi.r0;
	return bi;
      }

    };  
  }


  //Two body bonded forces. Handles particle-particle and particle.point bonds
  template<class BondType>
  class BondedForces: public Interactor, public ParameterUpdatableDelegate<BondType>{
  public:
  
    struct Parameters{
      std::string file; //File containing the bonds
    };
    //Aligning these really improves performance
    //A particle-particle bond
    struct __align__(16)  Bond{
      int i,j;
      typename BondType::BondInfo bond_info;
    };
    //A fixed point bond
    struct __align__(16) BondFP{
      int i;
      real3 pos;
      typename BondType::BondInfo bond_info;
    };


    explicit BondedForces(shared_ptr<ParticleData> pd,
			  shared_ptr<System> sys,
			  Parameters par,
			  BondType bondForce);
    explicit BondedForces(shared_ptr<ParticleData> pd,
			  shared_ptr<System> sys,
			  Parameters par):
      BondedForces(pd, sys, par, BondType()){}
  
    

  
    ~BondedForces();
    void callComputeBondedForces(cudaStream_t st);
    void sumForce(cudaStream_t st = 0) override;  
    real sumEnergy() override;
    //real sumVirial() override;
  
  private:
    void init();
    void initParticleParticle();
    void initFixedPoint();
  
    int nbonds;
    thrust::device_vector<Bond> bondList;
    thrust::device_vector<Bond*> bondStart;
    thrust::device_vector<int> nbondsPerParticle;  
  
    int nbondsFP; //Fixed Point
    thrust::device_vector<BondFP> bondListFP;
    thrust::device_vector<BondFP*> bondStartFP;
    thrust::device_vector<int> nbondsPerParticleFP;

    BondType bondForce;
  
    int TPP; // Threads per particle

  };
}


#include"BondedForces.cu"

#endif






