/* Raul P. Pelaez 2021
   Copy pastable examples of bonded interactions.
   This file contains ways to create two, three and four particle bonded
   interactions.

   UAMMD interactor modules always need some kind of specialization.
   For example, UAMMD offers a BondedForces module and provides some
   specialization (such as FENE or Harmonic bonds). You can, however, specialize
   it with any structure that follows the necessary rules. In this code you have
   some examples with specializations for the different bonded interactions.

   Every bond interaction needs a file with a list of bonded particles and any
   needed data for each of them. In the case of the harmonic bond below the file
   must have the following format:

   [number of bonds]
   i j k r0
   .
   .
   .

   The data needed for each bond (in the cas eof the harmonic bond belo k and
   r0) can be customized, see HarmonicBond::readBond With two particle bonds
   (BondedForces) a special kind of bond, called fixed point bond, can also be
   included in the file. Instead of joining two particles, a fixed bond joins a
   particle and a location in space. If fixed point bonds are required they must
   be placed after the two particle bonds (note the number of particle-particle
   bonds can be zero if only fixed point bonds exist): [number of bonds] i j
   Kspring r0
   .
   .
   .
   [number of fixed point bonds]
   i x y z Kspring r0
   .
   .
   .

   The file format for three and four particle bonds is similar, but instead of
   listing two particle ids each line must contain 3 r 4 particle names: For
   angular bonds: [number of bonds] i j k Kspring ang0
   .
   .
   For torsional bonds:
   [number of bonds]
   i j k l Kspring ang0
   .
   .

 */

#include "Interactor/BondedForces.cuh"
#include "uammd.cuh"
#include <fstream>
#include <random>
// #include"Interactor/AngularBondedForces.cuh"
// #include"Interactor/TorsionalBondedForces.cuh"
using namespace uammd;

// This struct contains the basic uammd modules for convenience.
struct UAMMD {
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
};

// Harmonic bond for pairs of particles
struct HarmonicBond {
  HarmonicBond(/*Parameters par*/) {
    // In this case no parameter is needed beyond whats in the bond file.
  }
  // Place in this struct whatever static information is needed for a given bond
  // In this case spring constant and equilibrium distance
  // the function readBond below takes care of reading each BondInfo from the
  // file
  struct BondInfo {
    real k, r0;
  };
  // This function will be called for every bond read in the bond file and is
  // expected to compute force/energy and or virial bond_index: The index of the
  // particle to compute force/energy/virial on ids: list of indexes of the
  // particles involved in the current bond pos: list of positions of the
  // particles involved in the current bond comp: computable targets (wether
  // force, energy and or virial are needed). bi: bond information for the
  // current bond (as returned by readBond)
  inline __device__ real sq(real a) { return a * a; }
  inline __device__ ComputeType compute(int bond_index, int ids[2],
                                        real3 pos[2],
                                        Interactor::Computables comp,
                                        BondInfo bi) {
    real3 r12 = pos[1] - pos[0];
    real r2 = dot(r12, r12);
    const real invr = rsqrt(r2);
    const real f = -bi.k * (real(1.0) - bi.r0 * invr); // F = -k·(r-r0)·rvec/r
    ComputeType ct;
    ct.force = f * r12;
    ct.energy = comp.energy ? (real(0.5) * bi.k * sq(real(1.0) / invr - bi.r0))
                            : real(0.0);
    ct.virial = comp.virial ? dot(ct.force, r12) : real(0.0);
    return (r2 == real(0.0)) ? (ComputeType{}) : ct;
  }

  // This function will be called for each bond in the bond file and read the
  // information of a bond It must use the stream that is handed to it to
  // construct a BondInfo.
  static __host__ BondInfo readBond(std::istream &in) {
    /*BondedForces will read i j, readBond has to read the rest of the line*/
    BondInfo bi;
    in >> bi.k >> bi.r0;
    return bi;
  }
};

// This angular potential is similar to the HarmonicBond above, the difference
// is that Now three particles are involved in each bond instead of two
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

// //This torsional potential is similar to the HarmonicBond above, the
// difference is that
// //Now four particles are involved in each bond instead of two
struct Torsional {
private:
  __device__ real3 cross(real3 a, real3 b) {
    return make_real3(a.y * b.z - a.z * b.y, (-a.x * b.z + a.z * b.x),
                      a.x * b.y - a.y * b.x);
  }

public:
  Box box;
  Torsional(real3 lbox /*Parameters par*/) : box(Box(lbox)) {}
  // Place in this struct whatever static information is needed for a given bond
  // In this case spring constant and equilibrium distance
  // the function readBond below takes care of reading each BondInfo from the
  // file
  struct BondInfo {
    real phi0, k;
  };
  // This function will be called for every bond read in the bond file and is
  // expected to compute force/energy and or virial bond_index: The index of the
  // particle to compute force/energy/virial on ids: list of indexes of the
  // particles involved in the current bond pos: list of positions of the
  // particles involved in the current bond comp: computable targets (wether
  // force, energy and or virial are needed). bi: bond information for the
  // current bond (as returned by readBond)
  inline __device__ ComputeType compute(int bond_index, int ids[4],
                                        real3 pos[4],
                                        Interactor::Computables comp,
                                        BondInfo bi) {
    const real3 rjk = box.apply_pbc(pos[1] - pos[0]);
    const real3 rkm = box.apply_pbc(pos[2] - pos[1]);
    const real3 rmn = box.apply_pbc(pos[3] - pos[2]);
    real3 njkm = cross(rjk, rkm);
    real3 nkmn = cross(rkm, rmn);
    const real n2 = dot(njkm, njkm);
    const real nn2 = dot(nkmn, nkmn);
    if (n2 > 0 and nn2 > 0) {
      const real invn = rsqrt(n2);
      const real invnn = rsqrt(nn2);
      const real cosphi = dot(njkm, nkmn) * invn * invnn;
      real Fmod = 0;
      // #define SMALL_ANGLE_BENDING
      // #ifdef SMALL_ANGLE_BENDING
      const real phi = acos(cosphi);
      if (cosphi * cosphi <= 1 and phi * phi > 0) {
        Fmod = -bi.k * (phi - bi.phi0) / sin(phi);
      }
      // #endif
      njkm *= invn;
      nkmn *= invnn;
      ComputeType ct{};
      const real3 v1 = (nkmn - cosphi * njkm) * invn;
      const real3 fj = Fmod * cross(v1, rkm);
      if (bond_index == ids[1]) {
        ct.force = real(-1.0) * fj;
        return ct;
      }
      const real3 v2 = (njkm - cosphi * nkmn) * invnn;
      const real3 fk = Fmod * cross(v2, rmn);
      const real3 fm = Fmod * cross(v1, rjk);
      if (bond_index == ids[2]) {
        ct.force = fm + fj - fk;
        return ct;
      }
      const real3 fn = Fmod * cross(v2, rkm);
      if (bond_index == ids[3]) {
        ct.force = fn + fk - fm;
        return ct;
      } else if (bond_index == ids[4]) {
        ct.force = real(-1.0) * fn;
        return ct;
      }
    }
    return ComputeType{};
  }
  // This function will be called for each bond in the bond file and read the
  // information of a bond It must use the stream that is handed to it to
  // construct a BondInfo.
  static BondInfo readBond(std::istream &in) {
    BondInfo bi;
    in >> bi.k >> bi.phi0;
    return bi;
  }
};

// You can use these functions to create an interactor that can be directly
// added to an integrator
std::shared_ptr<Interactor> createBondInteractor(UAMMD sim) {
  using Bond = HarmonicBond;
  using BF = BondedForces<Bond, 2>;
  typename BF::Parameters params;
  params.file = "bonds.dat";
  // You can pass an instance of the bond as a shared_ptr, which will allow you
  // to modify the bond properties at any time from outside BondedForces
  auto bond = std::make_shared<Bond>();
  auto bf = std::make_shared<BF>(sim.pd, params, bond);
  return bf;
}

std::shared_ptr<Interactor> createAngularBondInteractor(UAMMD sim) {
  using Bond = Angular;
  using BF = BondedForces<Bond, 3>;
  typename BF::Parameters params;
  params.file = "angular.bonds";
  real3 lbox = make_real3(32, 32, 32);
  auto bond = std::make_shared<Bond>(lbox);
  auto bf = std::make_shared<BF>(sim.pd, params, bond);
  return bf;
}

std::shared_ptr<Interactor> createTorsionalBondInteractor(UAMMD sim) {
  using Bond = Torsional;
  using BF = BondedForces<Bond, 4>;
  typename BF::Parameters params;
  params.file = "torsional.bonds";
  real3 lbox = make_real3(32, 32, 32);
  auto bond = std::make_shared<Bond>(lbox);
  auto bf = std::make_shared<BF>(sim.pd, params, bond);
  return bf;
}

// Initialize UAMMD with some arbitrary particles
UAMMD initializeUAMMD() {
  UAMMD sim;
  sim.sys = std::make_shared<System>();
  constexpr int numberParticles = 50000;
  sim.pd = std::make_shared<ParticleData>(numberParticles);
  auto pos = sim.pd->getPos(access::cpu, access::write);
  std::generate(pos.begin(), pos.end(), [&]() {
    return make_real4(sim.sys->rng().uniform3(-0.5, 0.5) * 32, 0);
  });
  return sim;
}

int main() {

  // Just an empty main so this file can be compiled on its own
  auto sim = initializeUAMMD();
  { // Create some random bonds between the particles, averaging 500
    // bonds/particle
    std::vector<int2> bs;
    std::ofstream out("bonds.dat");
    int N = sim.pd->getNumParticles();
    std::default_random_engine generator;
    std::uniform_int_distribution<int> n_dist(0, N - 1);
    std::uniform_int_distribution<int> n_bonds(0, 1000);
    fori(0, N) {
      forj(0, n_bonds(generator)) {
        int jj = n_dist(generator);
        bs.push_back({i, jj});
      }
    }

    out << bs.size() << std::endl;
    fori(0, bs.size()) { out << bs[i].x << " " << bs[i].y << " 10 0\n"; }
  }
  auto bonds = createBondInteractor(sim);

  fori(0, 20) bonds->sum({.force = true}, 0);
  // Measure the time of the different sum modes
  cudaStream_t st;
  cudaStreamCreate(&st);
  Timer tim;
  int ntest = 1000;
  cudaDeviceSynchronize();
  tim.tic();
  fori(0, ntest)
      bonds->sum({.force = true, .energy = false, .virial = false}, st);
  cudaDeviceSynchronize();
  sim.sys->log<System::MESSAGE>("Time per update (force): %g ms",
                                tim.toc() / ntest * 1e3);
  cudaDeviceSynchronize();
  tim.tic();
  fori(0, ntest) bonds->sum({.force = true, .energy = true}, st);
  cudaDeviceSynchronize();
  sim.sys->log<System::MESSAGE>("Time per update (force+ener): %g ms",
                                tim.toc() / ntest * 1e3);
  cudaDeviceSynchronize();
  tim.tic();
  fori(0, ntest)
      bonds->sum({.force = true, .energy = false, .virial = true}, st);
  cudaDeviceSynchronize();
  sim.sys->log<System::MESSAGE>("Time per update (force+virial): %g ms",
                                tim.toc() / ntest * 1e3);
  cudaDeviceSynchronize();
  tim.tic();
  fori(0, ntest)
      bonds->sum({.force = true, .energy = true, .virial = true}, st);
  cudaDeviceSynchronize();
  sim.sys->log<System::MESSAGE>("Time per update (force+ener+vir): %g ms",
                                tim.toc() / ntest * 1e3);
  sim.sys->finish();
  return 0;
}
