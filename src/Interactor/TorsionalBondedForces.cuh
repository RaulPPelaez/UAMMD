/*Raul P. Pelaez 2019-2021. Four bonded forces, AKA torsional springs.

  Joins four particles with a torsional bond i---j---k---l

  This Interactor is just an specialization of BondedForces for the case of 4 particles per bond.
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

#include"Interactor/BondedForces.cuh"

namespace uammd{

  namespace BondedType{
    //An harmonic torsional bond
    struct Torsional{
    private:
      
      __device__ real3 cross(real3 a, real3 b){
	return make_real3(a.y*b.z - a.z*b.y, (-a.x*b.z + a.z*b.x), a.x*b.y - a.y*b.x);
      }

    public:
      Box box;
      Torsional(real3 lbox /*Parameters par*/): box(Box(lbox)){}
      //Place in this struct whatever static information is needed for a given bond
      //In this case spring constant and equilibrium distance
      //the function readBond below takes care of reading each BondInfo from the file
      struct BondInfo{
	real phi0, k;
      };
      //This function will be called for every bond read in the bond file and is expected to compute force/energy and or virial
      //bond_index: The index of the particle to compute force/energy/virial on
      //ids: list of indexes of the particles involved in the current bond
      //pos: list of positions of the particles involved in the current bond
      //comp: computable targets (wether force, energy and or virial are needed).
      //bi: bond information for the current bond (as returned by readBond)
      inline __device__ ComputeType compute(int bond_index, int ids[4], real3 pos[4], Interactor::Computables comp, BondInfo bi){
	const real3 rjk = box.apply_pbc(pos[1] - pos[0]);
	const real3 rkm = box.apply_pbc(pos[2] - pos[1]);
	const real3 rmn = box.apply_pbc(pos[3] - pos[2]);
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
	    Fmod = -bi.k*(phi - bi.phi0)/sin(phi);
	  }
	  //#endif
	  njkm *= invn;
	  nkmn *= invnn;
	  ComputeType ct{};
	  const real3 v1 = (nkmn - cosphi*njkm)*invn;
	  const real3 fj = Fmod*cross(v1, rkm);
	  if(bond_index == ids[1]){
	    ct.force = real(-1.0)*fj;
	    return ct;
	  }
	  const real3 v2 = (njkm - cosphi*nkmn)*invnn;
	  const real3 fk = Fmod*cross(v2, rmn);
	  const real3 fm = Fmod*cross(v1, rjk);
	  if(bond_index == ids[2]){
	    ct.force = fm + fj - fk;
	    return ct;
	  }
	  const real3 fn = Fmod*cross(v2, rkm);
	  if(bond_index == ids[3]){
	    ct.force = fn + fk - fm;
	    return ct;
	  }
	  else if(bond_index == ids[4]){
	    ct.force = real(-1.0)*fn;
	    return ct;
	  }
	}
	return ComputeType{};
      }
      //This function will be called for each bond in the bond file and read the information of a bond
      //It must use the stream that is handed to it to construct a BondInfo.
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

      //Place in this struct whatever static information is needed for a given bond
      //In this case spring constant and equilibrium distance
      //the function readBond below takes care of reading each BondInfo from the file
      struct BondInfo{
	real phi0, kdih;
      };

      //This function will be called for each bond in the bond file and read the information of a bond
      //It must use the stream that is handed to it to construct a BondInfo.
      static BondInfo readBond(std::istream &in){
	BondInfo bi;
	in>>bi.kdih>>bi.phi0;
	return bi;
      }

      //This function will be called for every bond read in the bond file and is expected to compute force/energy and or virial
      //bond_index: The index of the particle to compute force/energy/virial on
      //ids: list of indexes of the particles involved in the current bond
      //pos: list of positions of the particles involved in the current bond
      //comp: computable targets (wether force, energy and or virial are needed).
      //bi: bond information for the current bond (as returned by readBond)
      inline __device__ ComputeType compute(int bond_index, int ids[4], real3 pos[4], Interactor::Computables comp, BondInfo bi){
	//define useful quantities
	const real3 r12 = box.apply_pbc(pos[1] - pos[0]);
	const real3 r23 = box.apply_pbc(pos[2] - pos[1]);
	const real3 r34 = box.apply_pbc(pos[3] - pos[2]);
	const real3 v123 = cross(r12, r23);
	const real3 v234 = cross(r23, r34);
	const real v123q = dot(v123, v123);
	const real v234q = dot(v234, v234);
	ComputeType ct{};
	if(v123q < real(1e-15) or v234q < real(1e-15)){
	  return ct;
	}
	if(comp.energy){
	  const real cosPhi = thrust::max(real(-1.0), thrust::min(real(1.0), dot(v123, v234)*rsqrt(v123q)*rsqrt(v234q)));
	  const real dphi = signOfPhi(r12, r23, r34)*acos(cosPhi) - bi.phi0;
	  ct.energy = real(0.25)*bi.kdih*(1+cos(dphi));  //U=kdih(1+cos(phi-phi0))
	}
	if(not comp.force and not comp.virial) return ct;
	const real invsqv123 = rsqrt(v123q);
	const real invsqv234 = rsqrt(v234q);
	const real cosPhi = thrust::max(real(-1.0), thrust::min(real(1.0), dot(v123, v234)*invsqv123 * invsqv234));
	const real phi = signOfPhi(r12, r23, r34)*acos(cosPhi);
	if(fabs(phi)<real(1e-10) or real(M_PI) - fabs(phi) < real(1e-10)){
	  return ct;
	}
	/// in order to change the potential, you just need to modify this with (dU/dphi)/sin(phi)
	const real pref = -bi.kdih*sin(phi-bi.phi0)/sin(phi);
	//compute force
	const real3 vu234 = v234*invsqv234;
	const real3 vu123 = v123*invsqv123;
	const real3 w1 = (vu234 - cosPhi*vu123)*invsqv123;
	const real3 w2 = (vu123 - cosPhi*vu234)*invsqv234;
	if (bond_index == ids[0]){
	  ct.force = pref*make_real3(cross(w1, r23));
	  ct.virial = comp.virial?dot(ct.force, r23):0;
	}
	else if (bond_index == ids[1]){
	  const real3 r13 = box.apply_pbc(pos[2] - pos[0]);
	  const real3 c34 = cross(w2, r34);
	  const real3 c13 = cross(w1, r13);
	  ct.force = pref*(c34 - c13);
	  ct.virial = comp.virial?(dot(pref*c34, r34) - dot(pref*c13, r13)):0;
	}
	else if (bond_index == ids[2]){
	  const real3 r24 = box.apply_pbc(pos[3] - pos[1]);
	  const real3 c12 = cross(w1, r12);
	  const real3 c24 = cross(w2, r24);
	  ct.force = pref*(c12 - c24);
	  ct.virial = comp.virial?(dot(pref*c12, r12) - dot(pref*c24, r24)):0;
	}
	else if (bond_index == ids[3]){
	  ct.force = pref*(cross(w2, r23));
	  ct.virial = comp.virial?dot(ct.force, r23):0;
	}
	return ct;
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
  }
  template<class BondType>
  using TorsionalBondedForces = BondedForces<BondType, 4>;
}
#endif
