/*Raul P. Pelaez 2021. Three bonded forces, AKA three body springs.

  Joins three particles with an angle bond i---j---k

  This Interactor is just an alias of BondedForces specialized for three
  particles per bond.

  Needs an input file containing the bond information as:
  nbonds
  i j k BONDINFO
  .
  .
  .

  K is the harmonic spring constant, r0 its eq. distance and ang0 the eq angle
  between ijk. The order doesnt matter as long as j is always the central
  particle in the bond.

  Where i,j,k are the indices of the particles. BONDINFO can be any number of
  rows, as described by the BondedType AngularBondedForces is used with, see
  AngularBondedForces_ns::AngularBond for an example.

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
#include "Interactor/BondedForces.cuh"
namespace uammd {

namespace BondedType {
// An angular harmonic bond
struct Angular {
  Box box;
  Angular(real3 lbox /*Parameters par*/) : box(Box(lbox)) {}

  // Place in this struct whatever static information is needed for a given bond
  // In this case spring constant and equilibrium distance
  // the function readBond below takes care of reading each BondInfo from the
  // file
  struct BondInfo {
    real ang0, k;
  };

  // This function will be called for every bond read in the bond file and is
  // expected to compute force/energy and or virial bond_index: The index of the
  // particle to compute force/energy/virial on ids: list of indexes of the
  // particles involved in the current bond pos: list of positions of the
  // particles involved in the current bond comp: computable targets (wether
  // force, energy and or virial are needed). bi: bond information for the
  // current bond (as returned by readBond)
  inline __device__ ComputeType compute(int bond_index, int ids[3],
                                        real3 pos[3],
                                        Interactor::Computables comp,
                                        BondInfo bi) {
    const real ang0 = bi.ang0;
    const real kspring = bi.k;
    //         i -------- j -------- k
    //             rij->     rjk ->
    // Compute distances and vectors
    //---rij---
    const real3 rij = box.apply_pbc(pos[1] - pos[0]);
    const real rij2 = dot(rij, rij);
    const real invsqrij = rsqrt(rij2);
    //---rkj---
    const real3 rjk = box.apply_pbc(pos[2] - pos[1]);
    const real rjk2 = dot(rjk, rjk);
    const real invsqrjk = rsqrt(rjk2);
    const real a2 = invsqrij * invsqrjk;
    real cijk =
        dot(rij, rjk) * a2; // cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)
    // Cos must stay in range
    if (cijk > real(1.0))
      cijk = real(1.0);
    else if (cijk < real(-1.0))
      cijk = -real(1.0);
    real ampli;
    ComputeType ct{};
    // //Approximation for small angle displacements
    // real sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) =
    // sqrt(1-cos(theta)^2)
    // //sijk cant be zero to avoid division by zero
    // if(sijk<std::numeric_limits<real>::min()) sijk =
    // std::numeric_limits<real>::min(); ampli = -kspring * (acos(cijk) -
    // ang0)/sijk; //The force amplitude -k·(theta-theta_0)
    // ampli = -kspring*(-sijk*cos(ang0)+cijk*sin(ang0))+ang0;
    // //k(1-cos(ang-ang0))
    if (ang0 == real(0.0)) {
      ampli = -real(2.0) * kspring;
    } else {
      const real theta = acos(cijk);
      if (theta == real(0.0))
        return ComputeType{};
      const real sinthetao2 = sin(real(0.5) * theta);
      ampli = -real(2.0) * kspring * (sinthetao2 - sin(ang0 * real(0.5))) /
              sinthetao2;
    }
    // Magical trigonometric relations to infere the direction of the force
    const real a11 = ampli * cijk / rij2;
    const real a12 = ampli * a2;
    const real a22 = ampli * cijk / rjk2;
    // Sum according to my position in the bond
    //  i ----- j ------ k
    if (bond_index == ids[0]) {
      ct.force = make_real3(a12 * rjk - a11 * rij);
    } else if (bond_index == ids[1]) {
      ct.force =
          real(-1.0) * make_real3((-a11 - a12) * rij + (a12 + a22) * rjk);
    } else if (bond_index == ids[2]) {
      ct.force = real(-1.0) * make_real3(a12 * rij - a22 * rjk);
    }
    return ct;
  }

  // This function will be called for each bond in the bond file and read the
  // information of a bond It must use the stream that is handed to it to
  // construct a BondInfo.
  static BondInfo readBond(std::istream &in) {
    BondInfo bi;
    in >> bi.k >> bi.ang0;
    return bi;
  }
};
} // namespace BondedType

namespace AngularBondedForces_ns {
using AngularBond = BondedType::Angular;
}

template <class BondType> using AngularBondedForces = BondedForces<BondType, 3>;
} // namespace uammd
#endif
