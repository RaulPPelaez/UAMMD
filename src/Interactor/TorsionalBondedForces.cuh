/*Raul P. Pelaez 2019-2021. Four bonded forces, AKA torsional springs.

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

  namespace BondedType{
    struct Torsional{
    private:

      __device__ real3 cross(real3 a, real3 b){
	return make_real3(a.y*b.z - a.z*b.y, (-a.x*b.z + a.z*b.x), a.x*b.y - a.y*b.x);
      }

    public:
      Box box;
      Torsional(Box box): box(box){}

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

      inline __device__ real energy(int j, int k, int m, int n,
				    int bond_index,
				    const real3 &posj,
				    const real3 &posk,
				    const real3 &posm,
				    const real3 &posn,
				    const BondInfo &bond_info){
	return 0;
      }

      static BondInfo readBond(std::istream &in){
	BondInfo bi;
	in>>bi.k>>bi.phi0;
	return bi;
      }

    };

    //Salvatore Assenza 2020.
    //Fourier like LAMMPS: U=kdih(1+cos(phi-phi0)) with phi in [-pi,pi]
    //kdih has to be given in my units
    struct FourierLAMMPS{
    public:
      Box box;
      FourierLAMMPS(Box box): box(box){}

      struct BondInfo{
	real phi0, kdih;
      };

      static BondInfo readBond(std::istream &in){
	BondInfo bi;
	in>>bi.kdih>>bi.phi0;
	return bi;
      }

      inline __device__ real3 force(int i1, int i2, int i3, int i4,
                                    int bond_index,
                                    real3 pos1,
                                    real3 pos2,
                                    real3 pos3,
                                    real3 pos4,
                                    BondInfo bond_info){
	//define useful quantities
	  const real3 r12 = box.apply_pbc(pos2 - pos1);
	  const real3 r23 = box.apply_pbc(pos3 - pos2);
	  const real3 r34 = box.apply_pbc(pos4 - pos3);
	  const real3 v123 = cross(r12, r23);
	  const real3 v234 = cross(r23, r34);
	  const real v123q = dot(v123, v123);
	  const real v234q = dot(v234, v234);
	  if(v123q < real(1e-15) or v234q < real(1e-15)){
	    return make_real3(0);
	  }
	  const real invsqv123 = rsqrt(v123q);
	  const real invsqv234 = rsqrt(v234q);
	  const real cosPhi = thrust::max(real(-1.0), thrust::min(real(1.0), dot(v123, v234)*invsqv123 * invsqv234));
	  const real phi = signOfPhi(r12, r23, r34)*acos(cosPhi);
	  if(fabs(phi)<real(1e-10) or real(M_PI) - fabs(phi) < real(1e-10)){
	    return make_real3(0);
	  }
	  /// in order to change the potential, you just need to modify this with (dU/dphi)/sin(phi)
	  const real pref = -bond_info.kdih*sin(phi-bond_info.phi0)/sin(phi);
	  //compute force
	  const real3 vu234 = v234*invsqv234;
	  const real3 vu123 = v123*invsqv123;
	  const real3 w1 = (vu234 - cosPhi*vu123)*invsqv123;
	  const real3 w2 = (vu123 - cosPhi*vu234)*invsqv234;
	  if (bond_index == i1){
	    return pref*make_real3(cross(w1, r23));
	  }
	  else if (bond_index == i2){
	    const real3 r13 = box.apply_pbc(pos3 - pos1);
	    return pref*make_real3(cross(w2, r34) - cross(w1, r13));
	  }
	  else if (bond_index == i3){
	    const real3 r24 = box.apply_pbc(pos4 - pos2);
	    return pref*make_real3(cross(w1, r12) - cross(w2, r24));
	  }
	  else if (bond_index == i4){
	    return pref*make_real3(cross(w2, r23));
	  }
	  return make_real3(0);
      }

      inline __device__ real energy(int i1, int i2, int i3, int i4,
                                    int bond_index,
                                    real3 pos1,
                                    real3 pos2,
                                    real3 pos3,
                                    real3 pos4,
                                    BondInfo &bond_info){
	//define useful quantities
	const real3 r12 = box.apply_pbc(pos2 - pos1);
	const real3 r23 = box.apply_pbc(pos3 - pos2);
	const real3 r34 = box.apply_pbc(pos4 - pos3);
	const real3 v123 = cross(r12, r23);
	const real3 v234 = cross(r23, r34);
	const real v123q = dot(v123, v123);
	const real v234q = dot(v234, v234);
	if (v123q < real(1e-15) || v234q < real(1e-15))
	  return real(0.0);
	const real cosPhi = thrust::max(real(-1.0), thrust::min(real(1.0), dot(v123, v234)*rsqrt(v123q)*rsqrt(v234q)));
	const real dphi = signOfPhi(r12, r23, r34)*acos(cosPhi) - bond_info.phi0;
	return real(0.25)*bond_info.kdih*(1+cos(dphi));  //U=kdih(1+cos(phi-phi0))
      }

    private:
      inline __device__ real signOfPhi(real3 r12, real3 r23, real3 r34){
	const real3 ru23 = r23*rsqrt(dot(r23, r23));
	const real3 uloc1 = r12*rsqrt(dot(r12, r12));
	const real3 uloc2 = ru23 - dot(uloc1, ru23)*uloc1;
	const real3 uloc3 = cross(uloc1, uloc2);
	const real segnophi = (dot(r34, uloc3) < 0)?real(-1.0):real(1.0);
	return segnophi;
      }

    };
  }

  namespace TorsionalBondedForces_ns{
    using TorsionalBond = BondedType::Torsional;
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
      std::string file;
    };

    explicit TorsionalBondedForces(shared_ptr<ParticleData> pd,
				   shared_ptr<System> sys,
				   Parameters par,
				   std::shared_ptr<BondType> bondType = std::make_shared<BondType>());
    
    explicit TorsionalBondedForces(shared_ptr<ParticleData> pd,
				   shared_ptr<System> sys,
				   Parameters par,
				   BondType bondType):
      TorsionalBondedForces(pd, sys, par, std::make_shared<BondType>(bondType)){}

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

    std::shared_ptr<BondType> bondType;
  };

}
#include"TorsionalBondedForces.cu"
#endif
