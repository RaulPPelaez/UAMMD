/*Raul P. Pelaez 2017-2021. Verlet NVT Integrator Gronbech Jensen module.
  See VerletNVT.cuh
  The algorithm implemented is GronbechJensen[1]

-----
References:

[1] N. Gronbech-Jensen, and O. Farago: "A simple and effective Verlet-type
algorithm for simulating Langevin dynamics", Molecular Physics (2013).
http://dx.doi.org/10.1080/00268976.2012.760055

 */

#include "../VerletNVT.cuh"
#include "third_party/saruprng.cuh"
// See Basic.cu for initialization etc, this class is inherited from Basic
namespace uammd {
namespace VerletNVT {
namespace GronbechJensen_ns {

// Integrate the movement 0.5 dt and reset the forces in the first step
// (step==1) Uses the Gronbech Jensen scheme[1]
//   r[t+dt] = r[t] + b·dt·v[t] + b·dt^2/(2·m) + b·dt/(2·m) · noise[t+dt]
//   v[t+dt] = a·v[t] + dt/(2·m)·(a·f[t] + f[t+dt]) + b/m·noise[t+dt]
//  b = 1/( 1 + \gamma·dt/(2·m))
//  a = (1 - \gamma·dt/(2·m) ) ·b
//  \gamma = friction*mass
template <int step>
__global__ void integrateGPU(real4 *__restrict__ pos, real3 *__restrict__ vel,
                             real4 *__restrict__ force,
                             const real *__restrict__ mass, real defaultMass,
                             ParticleGroup::IndexIterator indexIterator, int N,
                             real dt, real friction, bool is2D,
                             real noiseAmplitude, uint stepNum, uint seed) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N)
    return;
  // Index of current particle in group
  const int i = indexIterator[id];

  const real invMass = real(1.0) / (defaultMass > 0 ? defaultMass : mass[i]);
  if (step == 1) {
    Saru rng(id, stepNum, seed);
    noiseAmplitude *= rsqrt(invMass); // sqrt(2*kT*mass*friction*dt)
    real3 noisei = make_real3(rng.gf(0, noiseAmplitude),
                              is2D ? real() : rng.gf(0, noiseAmplitude).x);
    const real gdthalfinvMass = friction * dt * real(0.5);
    const real b = real(1.0) / (real(1.0) + gdthalfinvMass);
    const real a = (real(1.0) - gdthalfinvMass) * b;
    real3 p = make_real3(pos[i]);
    p = p + b * dt * vel[i] +
        real(0.5) * invMass * dt * b * (dt * make_real3(force[i]) + noisei);
    pos[i] = make_real4(p, pos[i].w);
    vel[i] = a * vel[i] + dt * real(0.5) * invMass * a * make_real3(force[i]) +
             b * invMass * noisei;
    force[i] = make_real4(0);
  } else {
    vel[i] += dt * real(0.5) * invMass * make_real3(force[i]);
  }
  if (is2D)
    vel[i].z = real(0.0);
}
} // namespace GronbechJensen_ns

template <int step> void GronbechJensen::callIntegrate() {
  int numberParticles = pg->getNumberParticles();
  int Nthreads = 128;
  int Nblocks =
      numberParticles / Nthreads + ((numberParticles % Nthreads) ? 1 : 0);
  // An iterator with the global indices of my groups particles
  auto groupIterator = pg->getIndexIterator(access::location::gpu);
  // Get all necessary properties
  auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
  auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
  auto force = pd->getForce(access::location::gpu, access::mode::read);
  auto mass =
      pd->getMassIfAllocated(access::location::gpu, access::mode::read).raw();
  if (this->defaultMass > 0) {
    mass = nullptr;
  }
  // First step integration and reset forces
  GronbechJensen_ns::integrateGPU<step><<<Nblocks, Nthreads, 0, stream>>>(
      pos.raw(), vel.raw(), force.raw(), mass, defaultMass, groupIterator,
      numberParticles, dt, friction, is2D, noiseAmplitude, steps, seed);
  CudaCheckError();
}
// Move the particles in my group 1 dt in time.
void GronbechJensen::forwardTime() {
  CudaCheckError();
  for (auto updatable : updatables)
    updatable->updateSimulationTime(steps * dt);
  steps++;
  sys->log<System::DEBUG1>("[%s] Performing integration step %d", name.c_str(),
                           steps);
  // First simulation step is special
  if (steps == 1) {
    resetForces();
    for (auto updatable : updatables) {
      updatable->updateTemperature(temperature);
      updatable->updateTimeStep(dt);
    }
    for (auto forceComp : interactors) {
      forceComp->sum({.force = true, .energy = false, .virial = false}, stream);
    }
    CudaSafeCall(cudaDeviceSynchronize());
  }
  // First integration step and force reset
  callIntegrate<1>();
  for (auto forceComp : interactors) {
    forceComp->sum({.force = true, .energy = false, .virial = false}, stream);
  }
  CudaCheckError();
  // Second integration step
  callIntegrate<2>();
}
} // namespace VerletNVT
} // namespace uammd
