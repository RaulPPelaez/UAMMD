#include "DPD/Potential.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"
#include "Interactor/PairForces.cuh"
#include "VerletDPD.cuh"
#include "third_party/type_names.h"
#include <random>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
namespace uammd {
namespace dpd {
template <DissipationKernel Kernel>
Verlet<Kernel>::Verlet(shared_ptr<ParticleGroup> pg, Verlet::Parameters par)
    : Integrator(pg, "Verlet"), dt(par.dt), lambda(par.lambda), is2D(par.is2D),
      mass(par.mass), dissipation(par.dissipation) {
  if (lambda < 0 || lambda > 1) {
    throw std::runtime_error("Verlet requires a lambda value between 0 and 1.");
  }
  if (dt < 0) {
    throw std::runtime_error("Verlet requires a positive time step (dt).");
  }
  if (mass < 0 && !pd->isMassAllocated()) {
    throw std::runtime_error(
        "Verlet requires a mass to be set, either through the parameters or "
        "by allocating mass in the particle data.");
  }
  if (!dissipation) {
    throw std::runtime_error("Verlet requires a dissipation kernel to be set.");
  }
  sys->log<System::MESSAGE>("[Verlet] Created with parameters: "
                            "lambda=%f, is2D=%d, mass=%f",
                            lambda, is2D, mass);
  sys->log<System::MESSAGE>("[Verlet] Dissipation kernel: %s",
                            type_name<Kernel>().c_str());
  CudaSafeCall(cudaStreamCreate(&stream));

  if (!pd->isVelAllocated()) {
    auto vel = pd->getVel(access::cpu, access::write);
    auto g_vel = pg->getPropertyIterator(vel);
    std::mt19937 gen(pd->getSystem()->rng().next());
    real mean = 0;
    real stdev = sqrt(par.temperature);
    std::normal_distribution<real> dis(mean, stdev);
    std::generate_n(g_vel, pg->getNumberParticles(),
                    [&]() { return make_real3(dis(gen), dis(gen), dis(gen)); });
  }

  using NeighbourList = VerletList;
  using DPDPotential = DPDPotential<Kernel>;
  using PFDPD = PairForces<DPDPotential, NeighbourList>;
  typename DPDPotential::Parameters potparams;
  potparams.dissipation = par.dissipation;
  potparams.cutOff = par.rcut;
  auto pot = std::make_shared<DPDPotential>(potparams);
  typename PFDPD::Parameters params;
  params.box = par.box;
  auto pairforces = std::make_shared<PFDPD>(pg, params, pot);
  this->addInteractor(pairforces);
}
template <DissipationKernel Kernel> Verlet<Kernel>::~Verlet() {
  sys->log<System::MESSAGE>("[Verlet] Destroyed.");
  cudaStreamDestroy(stream);
}

template <DissipationKernel Kernel>
void Verlet<Kernel>::computeCurrentForces() {

  {
    auto forces = pd->getForce(access::gpu, access::write);
    auto g_forces = pg->getPropertyIterator(forces);
    thrust::fill(thrust::cuda::par.on(stream), g_forces,
                 g_forces + pg->getNumberParticles(), real4{});
  }
  for (auto &interactor : interactors) {
    interactor->sum({.force = true}, stream);
  }
}
template <DissipationKernel Kernel> void Verlet<Kernel>::forwardTime() {
  if (step == 0) {
    computeCurrentForces();
  }
  thrust::device_vector<real3, System::allocator_thrust<real3>>
      velocityHalfStep(pg->getNumberParticles(), real3{});
  real3 *velocityHalfStep_ptr =
      thrust::raw_pointer_cast(velocityHalfStep.data());
  auto i_mass = pd->getMassIfAllocated(access::gpu, access::read);
  auto g_mass = pg->getPropertyIterator(i_mass);
  bool has_mass = pd->isMassAllocated();
  {
    auto vel = pd->getVel(access::gpu, access::readwrite);
    auto g_vel = pg->getPropertyIterator(vel);
    auto force = pd->getForce(access::gpu, access::read);
    auto g_force = pg->getPropertyIterator(force);
    auto pos = pd->getPos(access::gpu, access::readwrite);
    auto g_pos = pg->getPropertyIterator(pos);
    // First step: compute v_ n+1/2 and p_n+1/2, store in velocityHalfStep too
    auto cit = thrust::make_counting_iterator(0);
    const real mass = this->mass;
    const real dt = this->dt;
    const real lambda = this->lambda;
    thrust::for_each_n(thrust::cuda::par.on(stream), cit,
                       pg->getNumberParticles(), [=] __device__(const auto &i) {
                         real3 p = make_real3(g_pos[i]);
                         real3 v = make_real3(g_vel[i]);
                         real3 f = make_real3(g_force[i]);
                         real m = has_mass ? g_mass[i] : mass;
                         real3 half_step_vel = v + real(0.5) * f * dt / m;
                         real3 vel_prediction = v + lambda * dt * f / m;
                         g_pos[i] =
                             make_real4(p + half_step_vel * dt, g_pos[i].w);
                         g_vel[i] = vel_prediction;
                         velocityHalfStep_ptr[i] = half_step_vel;
                       });
  }
  computeCurrentForces();
  {
    auto vel = pd->getVel(access::gpu, access::readwrite);
    auto g_vel = pg->getPropertyIterator(vel);
    auto force = pd->getForce(access::gpu, access::read);
    auto g_force = pg->getPropertyIterator(force);
    auto cit = thrust::make_counting_iterator(0);
    const real mass = this->mass;
    const real dt = this->dt;
    thrust::for_each_n(thrust::cuda::par.on(stream), cit,
                       pg->getNumberParticles(), [=] __device__(const auto &i) {
                         real3 f = make_real3(g_force[i]);
                         real3 half_step_vel = velocityHalfStep_ptr[i];
                         real m = has_mass ? g_mass[i] : mass;
                         g_vel[i] = half_step_vel + real(0.5) * dt * f / m;
                       });
  }
  step++;
}
template <DissipationKernel Kernel> real Verlet<Kernel>::sumEnergy() {
  real kineticEnergy = 0.0;
  {
    auto vel = pd->getVel(access::gpu, access::read);
    auto g_vel = pg->getPropertyIterator(vel);
    auto i_mass = pd->getMassIfAllocated(access::gpu, access::read);
    auto g_mass = pg->getPropertyIterator(i_mass);
    bool has_mass = pd->isMassAllocated();
    auto cit = thrust::make_counting_iterator(0);
    const auto mass = this->mass;
    kineticEnergy = thrust::transform_reduce(
        thrust::cuda::par.on(stream), cit, cit + pg->getNumberParticles(),
        [=] __device__(const auto &i) -> real {
          real3 v = make_real3(g_vel[i]);
          real m = has_mass ? g_mass[i] : mass;
          return 0.5 * m * dot(v, v);
        },
        real(0), thrust::plus<real>());
  }
  return kineticEnergy;
}
} // namespace dpd
} // namespace uammd
