/* Raul P. Pelaez 2021
   Copy pastable examples of bonded interactions.
   This file contains ways to create two, three and four particle bonded interactions.

   UAMMD interactor modules always need some kind of specialization. 
   For example, UAMMD offers a BondedForces module and provides some specialization (such as FENE or Harmonic bonds).
   You can, however, specialize it with any structure that follows the necessary rules.
   In this code you have some examples with specializations for the different bonded interactions.   

   Every bond interaction needs a file with a list of bonded particles and any needed data for each of them.
   In the case of the harmonic bond below the file must have the following format:
   
   [number of bonds]
   i j k r0
   .
   .
   .
   
   The data needed for each bond (in the cas eof the harmonic bond belo k and r0) can be customized, see HarmonicBond::readBond
   With two particle bonds (BondedForces) a special kind of bond, called fixed point bond, can also be included in the file.
   Instead of joining two particles, a fixed bond joins a particle and a location in space. If fixed point bonds are required they must be placed after the two particle bonds (note the number of particle-particle bonds can be zero if only fixed point bonds exist):
   [number of bonds]
   i j Kspring r0
   .
   .
   .
   [number of fixed point bonds]
   i x y z Kspring r0
   .
   .
   .
   
   The file format for three and four particle bonds is similar, but instead of listing two particle ids each line must contain 3 r 4 particle names:
   For angular bonds:
   [number of bonds]
   i j k Kspring ang0
   .
   .
   For torsional bonds:
   [number of bonds]
   i j k l Kspring ang0
   .
   .

 */

#include"uammd.cuh"
#include"Interactor/BondedForces.cuh"
#include"Interactor/AngularBondedForces.cuh"
#include"Interactor/TorsionalBondedForces.cuh"
using namespace uammd;

//This struct contains the basic uammd modules for convenience.
struct UAMMD{
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
};


//Harmonic bond for pairs of particles
struct HarmonicBond{
  HarmonicBond(/*Parameters par*/){
    //In this case no parameter is needed beyond whats in the bond file.
  }
  //Place in this struct whatever static information is needed for a given bond
  //In this case spring constant and equilibrium distance
  //the function readBond below takes care of reading each BondInfo from the file
  struct BondInfo{
    real k, r0;
  };
  //This function will be called for every bond read in the bond file
  //In the case of a Fixed Point bond, j will be -1
  //i,j: id of particles in bond
  //r12: ri-rj
  //bi: bond information.
  inline __device__ real3 force(int i, int j, real3 r12, BondInfo bi){
    real r2 = dot(r12, r12);
    if(r2==real(0.0)) return make_real3(0.0);
    real invr = rsqrt(r2);
    real f = -bi.k*(real(1.0)-bi.r0*invr); //F = -k·(r-r0)·rvec/r
    return f*r12;
  }
  
  inline __device__ real energy(int i, int j, real3 r12, BondInfo bi){
    real r2 = dot(r12, r12);
    if(r2==real(0.0)) return real(0.0);
    real r = sqrt(r2);
    const real dr = r-bi.r0;
    return real(0.5)*bi.k*dr*dr;
  }

  //This function will be called for each bond in the bond file
  //It must use the stream that is handed to it to construct a BondInfo.  
  static __host__ BondInfo readBond(std::istream &in){
    /*BondedForces will read i j, readBond has to read the rest of the line*/
    BondInfo bi;
    in>>bi.k>>bi.r0;
    return bi;
  }

};

//This angular potential is similar to the HarmonicBond above, the difference is that
//Now three particles are involved in each bond instead of two
struct Angular{
  Box box;
  Angular(real3 lbox/*Parameters par*/): box(Box(lbox)){}
  
  struct BondInfo{
    real ang0, k;
  };
  
  inline __device__ real3 force(int i, int j, int k,
				int bond_index,
				real3 posi,
				real3 posj,
				real3 posk,
				BondInfo bond_info){
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
				real3 posi,
				real3 posj,
				real3 posk,
				BondInfo bond_info){
    return 0;
  }

  static BondInfo readBond(std::istream &in){
    BondInfo bi;
    in>>bi.k>>bi.ang0;
    return bi;
  }  
};

//This torsional potential is similar to the HarmonicBond above, the difference is that
//Now four particles are involved in each bond instead of two
struct Torsional{
private:

  __device__ real3 cross(real3 a, real3 b){
    return make_real3(a.y*b.z - a.z*b.y, (-a.x*b.z + a.z*b.x), a.x*b.y - a.y*b.x);
  }

public:
  Box box;
  Torsional(real3 lbox /*Parameters par*/): box(Box(lbox)){}

  struct BondInfo{
    real phi0, k;
  };

  inline __device__ real3 force(int j, int k, int m, int n,
				int bond_index,
				real3 posj,
				real3 posk,
				real3 posm,
				real3 posn,
				BondInfo bond_info){
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
				real3 posj,
				real3 posk,
				real3 posm,
				real3 posn,
				BondInfo bond_info){
    return 0;
  }

  static BondInfo readBond(std::istream &in){
    BondInfo bi;
    in>>bi.k>>bi.phi0;
    return bi;
  }

};


std::shared_ptr<Interactor> createBondInteractor(UAMMD sim){
  using Bond = HarmonicBond;
  using BF = BondedForces<Bond>;
  typename BF::Parameters params;
  params.file = "bondfile.dat";
  //You can pass an instance of the bond as a shared_ptr, which will allow you to modify the bond properties at any time
  //from outside BondedForces
  auto bond = std::make_shared<Bond>();
  auto bf = std::make_shared<BF>(sim.pd, sim.sys, params, bond);
  return bf;
}

std::shared_ptr<Interactor> createAngularBondInteractor(UAMMD sim){
  using Bond = Angular;
  using BF = AngularBondedForces<Bond>;
  typename BF::Parameters params;
  params.file = "angular.bonds";
  real3 lbox = make_real3(32,32,32);
  auto bond = std::make_shared<Bond>(lbox);
  auto bf = std::make_shared<BF>(sim.pd, sim.sys, params, bond);
  return bf;
}

std::shared_ptr<Interactor> createTorsionalBondInteractor(UAMMD sim){
  using Bond = Torsional;
  using BF = TorsionalBondedForces<Bond>;
  typename BF::Parameters params;
  params.file = "torsional.bonds"; 
  real3 lbox = make_real3(32,32,32);
  auto bond = std::make_shared<Bond>(lbox);
  auto bf = std::make_shared<BF>(sim.pd, sim.sys, params, bond);
  return bf;
}


int main(){

  //Just an empty main so this file can be compiled on its own

  return 0;
}
