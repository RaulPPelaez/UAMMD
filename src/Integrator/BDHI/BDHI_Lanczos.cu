/*Raul P. Pelaez 2017.
  BDHI Lanczos submodule.

  Computes the mobility matrix on the fly when needed, so it is a mtrix free
  method.

  M·F is computed as an NBody interaction (a dense Matrix vector product).

  BdW is computed using the Lanczos algorithm [1].


  References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in
  Brownian dynamics simulations. -http://dx.doi.org/10.1063/1.4742347 [2] J.
  Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347

*/
#include "BDHI_Lanczos.cuh"
#include "Interactor/NBody.cuh"
#include "misc/LanczosAlgorithm/MatrixDot.h"
#include "utils/container.h"

namespace uammd {
namespace BDHI {

Lanczos::Lanczos(shared_ptr<ParticleGroup> pg, Parameters par)
    : pg(pg), hydrodynamicRadius(par.hydrodynamicRadius),
      temperature(par.temperature), tolerance(par.tolerance),
      rpy(par.viscosity), par(par) {
  System::log<System::MESSAGE>("[BDHI::Lanczos] Initialized");
  auto pd = pg->getParticleData();
  // Lanczos algorithm computes,
  // given an object that computes the product of a Matrix(M) and a vector(v),
  // sqrt(M)·v
  lanczosAlgorithm = std::make_shared<lanczos::Solver>();
  if (par.hydrodynamicRadius > 0)
    System::log<System::MESSAGE>(
        "[BDHI::Lanczos] Self mobility: %g",
        rpy(0, par.hydrodynamicRadius, par.hydrodynamicRadius).x);
  else {
    System::log<System::MESSAGE>("[BDHI::Lanczos] Self mobility dependent on "
                                 "particle radius as 1/(6πηa)");
  }
  if (par.hydrodynamicRadius < 0 and !pd->isRadiusAllocated())
    System::log<System::CRITICAL>(
        "[BDHI::Lanczos] You need to provide Lanczos with either an "
        "hydrodynamic radius or via the individual particle radius.");
  if (par.hydrodynamicRadius > 0 and pd->isRadiusAllocated())
    System::log<System::MESSAGE>("[BDHI::Lanczos] Taking particle radius from "
                                 "parameter's hydrodynamicRadius");
  // Init rng
  curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(curng, pd->getSystem()->rng().next());
  thrust::device_vector<real> noise(30000);
  auto noise_ptr = thrust::raw_pointer_cast(noise.data());
  // Warm cuRNG
  curandgeneratenormal(curng, noise_ptr, noise.size(), 0.0, 1.0);
  curandgeneratenormal(curng, noise_ptr, noise.size(), 0.0, 1.0);
}

namespace Lanczos_ns {
/*Compute the product Mv = M·v, computing M on the fly when needed, without
 * storing it*/
/*This critital compute is the 99% of the execution time in a BDHI simulation*/
/*Each thread handles one particle with the other N, including itself*/
/*That is 3 full lines of M, or 3 elements of M·v per thread, being the x y z of
 * ij with j=0:N-1*/
/*In other words. M is made of NxN boxes of size 3x3,
  defining the x,y,z mobility between particle pairs,
  each thread handles a row of boxes and multiplies it by three elements of v*/
/*vtype can be real3 or real4*/
template <class vtype> struct NbodyMatrixFreeMobilityDot {
  typedef real3 computeType;
  typedef real4 infoType; // v[i], radius[i]
  NbodyMatrixFreeMobilityDot(vtype *v, real3 *Mv,
                             real rh, // Used only if radius is null
                             real *radius, BDHI::RotnePragerYamakawa rpy)
      : v(v), Mv(Mv), rpy(rpy), radius(radius), rh(rh) {}
  /*Start with 0*/
  inline __device__ computeType zero() { return make_real3(0); }

  inline __device__ infoType getInfo(int pi) {
    return make_real4(make_real3(v[pi]), radius ? radius[pi] : rh);
  }
  /*Just count the interaction*/
  inline __device__ computeType compute(const real4 &pi, const real4 &pj,
                                        const infoType &info_i,
                                        const infoType &info_j) {
    /*Distance between the pair*/
    const real3 rij = make_real3(pi) - make_real3(pj);
    const real r = sqrt(dot(rij, rij));
    const real3 vj = make_real3(info_j);
    /*Compute RPY coefficients, see more info in BDHI::RPYutils::RPY*/
    const real2 c12 = rpy(r, info_i.w, info_j.w);

    const real f = c12.x;
    const real gdivr2 = c12.y;

    /*Self mobility*/
    if (r == real(0.0))
      return f * vj;
    /*This expression is a little obfuscated, Mij*vj = f(rij)·I + g(rij)/rij^2 ·
      \vec{rij}\diadic \vec{rij} ) · \vec{vij} Where f and g are the
      hydrodinamic kernel coefficients
    */
    const real gv = gdivr2 * dot(rij, vj);
    /*gv = g(r)·( vx·rx + vy·ry + vz·rz )*/
    /*(g(r)·v·(r(diadic)r) )_ß = gv·r_ß*/
    const real3 Mv_t = f * vj + gv * rij;
    return Mv_t;
  }
  /*Sum the result of each interaction*/
  inline __device__ void accumulate(computeType &total,
                                    const computeType &cur) {
    total += cur;
  }

  /*Write the final result to global memory*/
  inline __device__ void set(int id, const computeType &total) {
    Mv[id] = total;
  }
  vtype *v;
  real3 *Mv;
  real rh;
  real *radius;
  BDHI::RotnePragerYamakawa rpy;
};

/*A functor to pass to LanczosAlgorithm the operation Mv = M·v*/
template <typename vtype> struct Dotctor : public lanczos::MatrixDot {
  using myTransverser = Lanczos_ns::NbodyMatrixFreeMobilityDot<vtype>;
  myTransverser Mv_tr;
  shared_ptr<NBody> nbody;
  cudaStream_t st;
  Dotctor(BDHI::RotnePragerYamakawa rpy, real rh, real *radius,
          shared_ptr<NBody> nbody, cudaStream_t st)
      : Mv_tr(nullptr, nullptr, rh, radius, rpy), nbody(nbody), st(st) {}

  inline void operator()(real *v, real *Mv) {
    Mv_tr.v = (vtype *)v;   /*src*/
    Mv_tr.Mv = (real3 *)Mv; /*Result*/
    nbody->transverse(Mv_tr, st);
  }
};
} // namespace Lanczos_ns

void Lanczos::computeMF(real3 *MF, cudaStream_t st) {
  /*For M·v product. Being M the Mobility and v an arbitrary array.
    The M·v product can be seen as an Nbody interaction Mv_j = sum_i(Mij*vi)
    Where Mij = RPY( |rij|^2 ).

    Although M is 3Nx3N, it is treated as a Matrix of NxN boxes of size 3x3,
    and v is a vector3.
  */
  System::log<System::DEBUG1>("[BDHI::Lanczos] MF");
  using myTransverser = Lanczos_ns::NbodyMatrixFreeMobilityDot<real4>;
  auto pd = pg->getParticleData();
  auto force = pd->getForce(access::location::gpu, access::mode::read);
  auto radius =
      pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);
  real *radius_ptr = this->hydrodynamicRadius > 0 ? nullptr : radius.raw();
  myTransverser Mv_tr(force.raw(), MF, this->hydrodynamicRadius, radius_ptr,
                      rpy);
  NBody nbody(pg);
  nbody.transverse(Mv_tr, st);
}

void Lanczos::computeBdW(real3 *BdW, cudaStream_t st) {
  System::log<System::DEBUG1>("[BDHI::Lanczos] BdW");
  if (temperature > real(0.0)) {
    st = 0;
    int numberParticles = pg->getNumberParticles();
    auto nbody = std::make_shared<NBody>(pg);
    auto pd = pg->getParticleData();
    /*Lanczos Algorithm needs a functor that provides the dot product of M and a
     * vector*/
    auto radius =
        pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);
    real *radius_ptr = this->hydrodynamicRadius > 0 ? nullptr : radius.raw();
    Lanczos_ns::Dotctor<real3> Mdot(rpy, this->hydrodynamicRadius, radius_ptr,
                                    nbody, st);
    // Filling V instead of an external array (for v in sqrt(M)·v) is faster
    uninitialized_cached_vector<real3> noise(numberParticles);
    curandgeneratenormal(curng, (real *)noise.data().get(),
                         3 * numberParticles + (3 * numberParticles) % 2,
                         real(0.0), real(1.0));
    // lanczosAlgorithm->solve(Mdot, (real*) BdW, noise, numberParticles, st);
    lanczosAlgorithm->run(Mdot, (real *)BdW, (real *)noise.data().get(),
                          tolerance, 3 * numberParticles, st);
  }
}
} // namespace BDHI
} // namespace uammd
