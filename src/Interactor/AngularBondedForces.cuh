/*Raul P. Pelaez 2019. Three bonded forces, AKA three body springs.

  Joins three particles with an angle bond i---j---k

  Needs an input file containing the bond information as:
  nbonds
  i j k BONDINFO
  .
  .
  .

  K is the harmonic spring constant, r0 its eq. distance and ang0 the eq angle between ijk.
  The order doesnt matter as long as j is always the central particle in the bond.

  Where i,j,k are the indices of the particles. BONDINFO can be any number of rows, as described
  by the BondedType AngularBondedForces is used with, see AngularBondedForces_ns::AngularBond for an example.

  The order doesnt matter, but j must always be the central particle.
  A bond type can be ParameterUpdatable.


  USAGE:

  //Choose a bond type
  using AngularBondType = AngularBondedForces_ns::AngularBond;
  using Angular = AngularBondedForces<AngularBondType>;


  //AngularBond needs a simulation box
  Box box(128);
  Angular::Parameters ang_params;
  ang_params.readFile = "angular.bonds";
  auto pot = std::make_shared<AngularBondType>(box);
  auto abf = make_shared<Angular>(pd, sys, ang_params, pot);
  ...
  myIntegrator->addInteractor(abf);
  ...
 */
#ifndef ANGULARBONDEDFORCES_CUH
#define ANGULARBONDEDFORCES_CUH

#include"Interactor.cuh"
#include"global/defines.h"
#include<thrust/device_vector.h>
#include<vector>
#include"utils/exception.h"
#include<limits>

namespace uammd{

  namespace BondedType{
    struct Angular{
      Box box;
      Angular(Box box): box(box){}
      struct BondInfo{
	real ang0, k;
      };
      inline __device__ real3 force(int i, int j, int k,
				    int bond_index,
				    const real3 &posi,
				    const real3 &posj,
				    const real3 &posk,
				    const BondInfo &bond_info){
	const real ang0 = bond_info.ang0;
	const real kspring = bond_info.k;

	//         i -------- j -------- k
	//             rij->     rjk ->
	//Compute distances and vectors
	//---rij---
	const real3 rij =  box.apply_pbc(posj - posi);
	const real rij2 = dot(rij, rij);
	const real invsqrij = rsqrt(rij2);
	//---rkj---
	const real3 rjk =  box.apply_pbc(posk - posj);
	const real rjk2 = dot(rjk, rjk);
	const real invsqrjk = rsqrt(rjk2);


	const real a2 = invsqrij * invsqrjk;

        real cijk = dot(rij, rjk)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)
	//Cos must stay in range
	if(cijk>real(1.0)) cijk = real(1.0);
	else if (cijk<real(-1.0)) cijk = -real(1.0);

	real ampli;

	// //Approximation for small angle displacements
	// real sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
	// //sijk cant be zero to avoid division by zero
	// if(sijk<std::numeric_limits<real>::min()) sijk = std::numeric_limits<real>::min();
	// ampli = -kspring * (acos(cijk) - ang0)/sijk; //The force amplitude -k·(theta-theta_0)
	//ampli = -kspring*(-sijk*cos(ang0)+cijk*sin(ang0))+ang0; //k(1-cos(ang-ang0))

	 if(ang0 == real(0.0)){
	   ampli = -real(2.0)*kspring;
	 }
	 else{
	   const real theta = acos(cijk);
	   if(theta==real(0.0))  return make_real3(0);
	   const real sinthetao2 = sin(real(0.5)*theta);
	   ampli = -real(2.0)*kspring*(sinthetao2 - sin(ang0*real(0.5)))/sinthetao2;
	 }


	//Magical trigonometric relations to infere the direction of the force

	const real a11 = ampli*cijk/rij2;
	const real a12 = ampli*a2;
	const real a22 = ampli*cijk/rjk2;

	//Sum according to my position in the bond
	// i ----- j ------ k
	if(bond_index==i){
	  return make_real3(a12*rjk -a11*rij); //Angular spring
	}
	else if(bond_index==j){
	  //Angular spring
	  return real(-1.0)*make_real3((-a11 - a12)*rij + (a12 + a22)*rjk);
	}
	else if(bond_index==k){
	  //Angular spring
	  return real(-1.0)*make_real3(a12*rij -a22*rjk);
	}
	return make_real3(0);
      }

      inline __device__ real energy(int i, int j, int k,
				    int bond_index,
				    const real3 &posi,
				    const real3 &posj,
				    const real3 &posk,
				    const BondInfo &bond_info){
	const real ang0 = bond_info.ang0;
	const real kspring = bond_info.k;
	//         i -------- j -------- k
	//             rij->     rjk ->
	//Compute distances and vectors
	//---rij---
	const real3 rij =  box.apply_pbc(posj - posi);
	const real rij2 = dot(rij, rij);
	const real invsqrij = rsqrt(rij2);
	//---rkj---
	const real3 rjk =  box.apply_pbc(posk - posj);
	const real rjk2 = dot(rjk, rjk);
	const real invsqrjk = rsqrt(rjk2);
	const real a2 = invsqrij * invsqrjk;
        real cijk = dot(rij, rjk)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)
	//Cos must stay in range
	if(cijk>real(1.0)) cijk = real(1.0);
	else if (cijk<real(-1.0)) cijk = -real(1.0);
	real ampli;
	// //Approximation for small angle displacements
	// real sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
	// //sijk cant be zero to avoid division by zero
	// if(sijk<std::numeric_limits<real>::min()) sijk = std::numeric_limits<real>::min();
	// ampli = -kspring * (acos(cijk) - ang0)/sijk; //The force amplitude -k·(theta-theta_0)
	//ampli = -kspring*(-sijk*cos(ang0)+cijk*sin(ang0))+ang0; //k(1-cos(ang-ang0))
	const real theta = acos(cijk);
	if(theta==real(0.0))  return real(0);
	const real sinthetao2 = sin(real(0.5)*theta);
	const real stmst= (sinthetao2 - sin(ang0*real(0.5)));
	ampli = real(4.0)*kspring*stmst*stmst;
	//Split the bond energy between the three particles
	return ampli/real(3.0);
      }

      static BondInfo readBond(std::istream &in){
	BondInfo bi;
	in>>bi.k>>bi.ang0;
	return bi;
      }

    };
  }

  namespace AngularBondedForces_ns{
    using AngularBond = BondedType::Angular;
    
    template<class Bond>
    class BondProcessor{
      int numberParticles;
      std::vector<std::vector<int>> isInBonds;
      std::vector<Bond> bondList;
      std::set<int> particlesWithBonds;

      void registerParticleInBond(int particleIndex, int b){
	isInBonds[particleIndex].push_back(b);
	particlesWithBonds.insert(particleIndex);

      }
    public:

      BondProcessor(int numberParticles):
      numberParticles(numberParticles),
	isInBonds(numberParticles){
      }

      void hintNumberBonds(int nbonds){
	bondList.reserve(nbonds);
      }

      void registerBond(Bond b){
	int bondIndex = bondList.size();
	bondList.push_back(b);
	registerParticleInBond(b.i, bondIndex);
	registerParticleInBond(b.j, bondIndex);
	registerParticleInBond(b.k, bondIndex);
      }

      std::vector<int> getParticlesWithBonds() const{
	std::vector<int> pwb;
	pwb.assign(particlesWithBonds.begin(), particlesWithBonds.end());
	return std::move(pwb);
      }

      std::vector<Bond> getBondListOfParticle(int index) const{
	std::vector<Bond> blst;
	blst.resize(isInBonds[index].size());
	fori(0, blst.size()){
	  blst[i] = bondList[isInBonds[index][i]];
	}
	return std::move(blst);
      }

      void checkDuplicatedBonds(){
	struct AngularBondCompareLessThan{
	  bool operator()(const Bond& lhs, const Bond &rhs) const{
	    auto lhs_t = std::make_tuple(lhs.i, lhs.j, lhs.k);
	    auto rhs_t = std::make_tuple(rhs.i, rhs.j, rhs.k);
	    if(std::get<2>(lhs_t) < std::get<0>(lhs_t)){
	      std::swap(std::get<2>(lhs_t), std::get<0>(lhs_t));
	    }
	    if(std::get<2>(rhs_t) < std::get<0>(rhs_t)){
	      std::swap(std::get<2>(rhs_t), std::get<0>(rhs_t));
	    }
	    const bool lessThan = lhs_t < rhs_t;
	    if(lessThan){
	      return true;
	    }
	    else{
	      return false;
	    }
	  }
	};
	std::set<Bond, AngularBondCompareLessThan> checkDuplicates;
	fori(0, bondList.size()){
	  if(!checkDuplicates.insert(bondList[i]).second)
	    System::log<System::WARNING>("[AngularBondedForces] Bond %d %d %d with index %d is duplicated!",
					 bondList[i].i, bondList[i].j, bondList[i].k, i);
	}
      }
    };

    class BondReader{
      std::ifstream in;
      int nbonds = 0;
    public:
      BondReader(std::string bondFile): in(bondFile){
	if(!in){
	  throw std::runtime_error("[BondReader] File " + bondFile + " cannot be opened.");
	}
	in>>nbonds;
      }

      int getNumberBonds(){
	return nbonds;
      }

      template<class Bond, class BondType>
      Bond readNextBond(){
	int i, j, k;
	if(!(in>>i>>j>>k)){
	  throw std::ios_base::failure("File unreadable");
	}
	Bond bond;
	bond.i = i;
	bond.j = j;
	bond.k = k;
	bond.bond_info = BondType::readBond(in);
	return bond;
      }

    };

  }

  template<class BondType>
  class AngularBondedForces: public Interactor, public ParameterUpdatableDelegate<BondType>{
  public:

    struct __align__(16) Bond{
      int i,j,k;
      typename BondType::BondInfo bond_info;
    };

    struct Parameters{
      std::string file;
    };

    explicit AngularBondedForces(shared_ptr<ParticleData> pd,
				 shared_ptr<System> sys,
				 Parameters par,
				 std::shared_ptr<BondType> bondType = std::make_shared<BondType>());
    
    explicit AngularBondedForces(shared_ptr<ParticleData> pd,
				 shared_ptr<System> sys,
				 Parameters par,
				 BondType bondType):
      AngularBondedForces(pd, sys, par, std::make_shared<BondType>(bondType)){}

    ~AngularBondedForces() = default;

    void sumForce(cudaStream_t st) override;
    real sumEnergy() override;

  private:
    static constexpr int numberParticlesPerBond = 3;
    using BondProcessor = AngularBondedForces_ns::BondProcessor<Bond>;
    using BondReader = AngularBondedForces_ns::BondReader;

    BondProcessor readBondFile(std::string bondFile);
    void generateBondList(const BondProcessor &bondProcessor);

    int nbonds;
    thrust::device_vector<Bond> bondList;   //[All bonds involving the first particle with bonds, involving the second...] each bonds stores the id of the three particles in the bond. The id of the first/second... particle  with bonds is particlesWithBonds[i]
    thrust::device_vector<int> bondStart, bondEnd; //bondStart[i], Where the list of bonds of particle with bond number i start (the id of particle i is particlesWithBonds[i].
    thrust::device_vector<int> particlesWithBonds; //List of particle ids with at least one bond
    int TPP; //Threads per particle

    std::shared_ptr<BondType> bondType;
  };

}
#include"AngularBondedForces.cu"
#endif
