/*Raul P. Pelaez 2019. Four bonded forces, AKA torsional springs.

  Joins four particles with a torsional bond i---j---k---l

  Needs an input file containing the bond information as:
  nbonds
  i j k l BONDINFO
  .
  .
  .

  Where i,j,k,l are the indices of the particles. BONDINFO can be any number of rows, as described
  by the BondedType TorsionalBondedForces is used with, see TorsionalBondedForces_ns::TorsionalBond for an example.

  The order doesnt matter.
  A bond type can be ParameterUpdatable.

  USAGE:

  //Choose a bond type
  using TorsionalBondType = TorsionalBondedForces_ns::TorsionalBond;
  using Torsional = TorsionalBondedForces<TorsionalBondType>;


  //TorsionalBond needs a simulation box
  Box box(128);
  Torsional::Parameters ang_params;
  ang_params.readFile = "torsional.bonds";
  auto abf = make_shared<Torsional>(pd, sys, ang_params, TorsionalBondType(box));
  ...
  myIntegrator->addInteractor(abf);
  ...
 */
#ifndef TORSIONALBONDEDFORCES_CUH
#define TORSIONALBONDEDFORCES_CUH

#include"Interactor.cuh"
#include"global/defines.h"
#include<thrust/device_vector.h>
#include<vector>
#include"utils/exception.h"
#include<limits>

namespace uammd{

  namespace TorsionalBondedForces_ns{
    struct TorsionalBond{
    private:
      
      __device__ real3 cross(real3 a, real3 b){
	return make_real3(a.y*b.z - a.z*b.y, (-a.x*b.z + a.z*b.x), a.x*b.y - a.y*b.x);
      }
      
    public:
      Box box;
      TorsionalBond(Box box): box(box){}
      
      struct BondInfo{
	real phi0, k;
      };
      
      inline __device__ real3 force(int j, int k, int m, int n,
				    int bond_index,
				    const real3 &posj,
				    const real3 &posk,
				    const real3 &posm,
				    const real3 &posn,
				    const BondInfo &bond_info){
	const real3 rjk = box.apply_pbc(posk - posj);
	const real3 rkm = box.apply_pbc(posm - posk);
	const real3 rmn = box.apply_pbc(posn - posm);
	real3 njkm = cross(rjk, rkm);
	real3 nkmn = cross(rkm, rmn);
	const real n2 = dot(njkm, njkm);
	const real nn2 = dot(nkmn, nkmn);
	if(n2 > 0 and nn2 > 0) {
	  const real invn = rsqrt(n2);
	  const real invnn = rsqrt(nn2);
	  const real cosphi = dot(njkm, nkmn)*invn*invnn;
	  real Fmod = 0;
	  // #define SMALL_ANGLE_BENDING
	  // #ifdef SMALL_ANGLE_BENDING
	  const real phi = acos(cosphi);
	  if(cosphi*cosphi <= 1 and phi*phi > 0){
	    Fmod = -bond_info.k*(phi - bond_info.phi0)/sin(phi);
	  }
	  //#endif
	  njkm *= invn;
	  nkmn *= invnn;
	  const real3 v1 = (nkmn - cosphi*njkm)*invn;
	  const real3 fj = Fmod*cross(v1, rkm);
	  if(bond_index == j){
	    return real(-1.0)*fj;
	  }
	  const real3 v2 = (njkm - cosphi*nkmn)*invnn;	   
	  const real3 fk = Fmod*cross(v2, rmn);
	  const real3 fm = Fmod*cross(v1, rjk);	   
	  if(bond_index == k){
	    return fm + fj - fk;
	  }
	  const real3 fn = Fmod*cross(v2, rkm);
	  if(bond_index == m){
	    return fn + fk - fm;
	  }
	  if(bond_index == n){
	    return real(-1.0)*fn;
	  }
	}
        return real3();
      }

      static BondInfo readBond(std::istream &in){
	BondInfo bi;
	in>>bi.k>>bi.phi0;
	return bi;
      }

    };    
  }

  namespace TorsionalBondedForces_ns{
      
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
	registerParticleInBond(b.l, bondIndex);
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

      void  checkDuplicatedBonds(){
	//TODO
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
	int i, j, k, l;
	if(!(in>>i>>j>>k>>l)){
	  throw std::ios_base::failure("File unreadable");
	}
	Bond bond;
	bond.i = i;
	bond.j = j;
	bond.k = k;
	bond.l = l;
	bond.bond_info = BondType::readBond(in);
	return bond;
      }
      
    };
      
  }


  template<class BondType>
  class TorsionalBondedForces: public Interactor, public ParameterUpdatableDelegate<BondType>{
  public:
    
    struct __align__(16) Bond{
      int i,j,k,l;
      typename BondType::BondInfo bond_info;
    };
    
    struct Parameters{
      std::string readFile;
    };

    explicit TorsionalBondedForces(shared_ptr<ParticleData> pd,
				   shared_ptr<System> sys,
				   Parameters par,
				   BondType bondType);
    
    ~TorsionalBondedForces() = default;

    void sumForce(cudaStream_t st) override;
    real sumEnergy() override;

  private:
    static constexpr int numberParticlesPerBond = 4;
    using BondProcessor = TorsionalBondedForces_ns::BondProcessor<Bond>;
    using BondReader = TorsionalBondedForces_ns::BondReader;
    
    BondProcessor readBondFile(std::string bondFile);
    void generateBondList(const BondProcessor &bondProcessor);
    
    int nbonds;
    thrust::device_vector<Bond> bondList;   //[All bonds involving the first particle with bonds, involving the second...] each bonds stores the id of the three particles in the bond. The id of the first/second... particle  with bonds is particlesWithBonds[i]
    thrust::device_vector<int> bondStart, bondEnd; //bondStart[i], Where the list of bonds of particle with bond number i start (the id of particle i is particlesWithBonds[i].
    thrust::device_vector<int> particlesWithBonds; //List of particle ids with at least one bond
    int TPP; //Threads per particle
    
    BondType bondType;
  };

}
#include"TorsionalBondedForces.cu"
#endif
