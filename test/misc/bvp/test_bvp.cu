/*Raul P. Pelaez 2025. Tests for the BVP solver
 */
#include "misc/BoundaryValueProblem/BVPSolver.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>
using namespace uammd;
using complex = cufftComplex_t<real>;

constexpr real tolerance = 1e-13;
struct Parameters {
  real k = 2;
  real H = 1;
  real gamma = 1;
  int nz = 32;
  complex alpha = {1, 0};
  complex beta = {1, 0};
};

class TopBoundaryConditions {
  real k, H;

public:
  TopBoundaryConditions(real k, real H) : k(k), H(H) {}

  real getFirstIntegralFactor() const { return (k != 0) * H; }

  real getSecondIntegralFactor() const { return k != 0 ? (k * H * H) : (1.0); }
};

class BottomBoundaryConditions {
  real k, H;

public:
  BottomBoundaryConditions(real k, real H) : k(k), H(H) {}

  real getFirstIntegralFactor() const { return (k != 0) * H; }

  real getSecondIntegralFactor() const { return k != 0 ? (-k * H * H) : (1.0); }
};

template <class BoundaryConditions, class Klist>
class BoundaryConditionsDispatch {
  const Klist *klist;
  real H;

public:
  BoundaryConditionsDispatch(const Klist &klist, real H) : klist(&klist), H(H) {}

  BoundaryConditions operator()(int i) const {
    return BoundaryConditions((*klist)[i], H);
  }
};

template <class BoundaryConditions, class Klist>
auto make_boundary_dispatcher(Klist &klist, real H) {
  return thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0),
      BoundaryConditionsDispatch<BoundaryConditions, Klist>(klist, H));
}

template <class Container> auto cheb2real(Container &cn_gpu) {
  std::vector<complex> cn(cn_gpu.size());
  thrust::copy(cn_gpu.begin(), cn_gpu.end(), cn.begin());
  int nz = cn.size();
  std::vector<complex> res(nz, complex());
  fori(0, cn.size()) {
    real z = i * M_PI / (nz - 1);
    forj(0, cn.size()) { res[i] += cn[j] * complex{cos(j * z), cos(j * z)}; }
  }
  return res;
}

template <class Container> auto real2cheb(Container &cn_gpu) {
  std::vector<complex> cn(cn_gpu.size());
  thrust::copy(cn_gpu.begin(), cn_gpu.end(), cn.begin());
  int nz = cn.size();
  std::vector<complex> res(nz, complex());
  fori(0, cn.size()) {
    real pm = i == 0 ? 1 : 2;
    res[i] += pm / (nz - 1) * (0.5 * (cn[0] * pow(-1, i) + cn[nz - 1]));
    forj(1, cn.size() - 1) {
      real z = j / (nz - 1.0);
      res[i] += (pm / (nz - 1)) * cn[j] * pow(-1, i) * cospi(i * z);
    }
  }
  return res;
}

auto abs(complex z) { return sqrt(z.x * z.x + z.y * z.y); }

// Compute the right–hand side for f(z) = exp(-gamma*z*z)
std::vector<complex> computeRightHandSideExpGamma(const Parameters &par) {
  int nz = par.nz;
  real H = par.H;
  std::vector<complex> f(nz, {0, 0});
  for (int i = 0; i < nz; i++) {
    real z = H * cospi(static_cast<real>(i) / (nz - 1));
    real val = exp(-par.gamma * z * z) * H * H;
    f[i] = {val, val};
  }
  return real2cheb(f);
}

// Compute a random right–hand side (used in the MATLAB–comparison test)
std::vector<complex> computeRightHandSideRandom(const Parameters &par) {
  int nz = par.nz;
  std::vector<complex> f(nz, {0, 0});
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<real> uniform(-1.0, 1.0);
  for (int i = 0; i < nz; i++) {
    f[i] = {uniform(eng), uniform(eng)};
  }
  return real2cheb(f);
}

// The analytic solution (in real space) for the given BVP.
// Note: The formula below is taken from the original test code.
real evaluateSolution(real x, real H, real a, real k, real b, real c) {
  long double pi = M_PIl;
  // Solution for f(z) = exp(-gamma*z*z)
  return -(expl(-powl((-2 * a - k), 2) / (4 * a) -
                powl((k - 2 * a), 2) / (4 * a) - k * x - 2 * k) *
           (-2 * sqrtl(a) * b * k *
                expl(powl((-2 * a - k), 2) / (4 * a) +
                     powl((k - 2 * a), 2) / (4 * a) + 2 * k * x + k) +
            2 * sqrtl(a) * c * k *
                expl(powl((-2 * a - k), 2) / (4 * a) +
                     powl((k - 2 * a), 2) / (4 * a) + k) +
            sqrtl(pi) * k * erf((2 * a + k) / (2 * sqrtl(a))) *
                expl(powl(k, 2) / (4 * a) + powl((-2 * a - k), 2) / (4 * a) +
                     powl((k - 2 * a), 2) / (4 * a) + 2 * k * x + 2 * k) +
            sqrtl(pi) * k *
                expl(powl(k, 2) / (4 * a) + powl((-2 * a - k), 2) / (4 * a) +
                     powl((k - 2 * a), 2) / (4 * a) + 2 * k) *
                erf((2 * a * x - k) / (2 * sqrtl(a))) -
            sqrtl(pi) * k * erf((2 * a * x + k) / (2 * sqrtl(a))) *
                expl(powl(k, 2) / (4 * a) + powl((-2 * a - k), 2) / (4 * a) +
                     powl((k - 2 * a), 2) / (4 * a) + 2 * k * x + 2 * k) -
            sqrtl(pi) * k * erf((-2 * a - k) / (2 * sqrtl(a))) *
                expl(powl(k, 2) / (4 * a) + powl((-2 * a - k), 2) / (4 * a) +
                     powl((k - 2 * a), 2) / (4 * a) + 2 * k) -
            sqrtl(a) * expl(powl(k, 2) / (4 * a) +
                            powl((-2 * a - k), 2) / (4 * a) + 2 * k * x) +
            sqrtl(a) *
                expl(powl(k, 2) / (4 * a) + powl((k - 2 * a), 2) / (4 * a) +
                     2 * k * x + 2 * k) -
            sqrtl(a) *
                expl(powl(k, 2) / (4 * a) + powl((-2 * a - k), 2) / (4 * a)) +
            sqrtl(a) * expl(powl(k, 2) / (4 * a) +
                            powl((k - 2 * a), 2) / (4 * a) + 2 * k))) /
         (4 * sqrtl(a) * powl(k, 2));
  // Solution for f(z)=0
  // return exp(-k*(x+1))*(c-b*exp(2*k*x))/(2*k);
}

auto computeSolutionCheb(Parameters par) {
  std::vector<complex> sol(par.nz);
  for (int i = 0; i < par.nz; i++) {
    real x = -cospi((real(i)) / (par.nz - 1));
    real sr =
        evaluateSolution(x, par.H, par.gamma, par.k, par.alpha.x, par.beta.x);
    real si =
        evaluateSolution(x, par.H, par.gamma, par.k, par.alpha.y, par.beta.y);
    sol[i] = complex{sr, si};
  }
  return real2cheb(sol);
}

// Get the normalization from a vector (max absolute value of real and imag
// parts separately).
complex getErrorNormalization(const std::vector<complex> &v) {
  complex norm = {0, 0};
  for (const auto &a : v) {
    if (std::fabs(a.x) > std::fabs(norm.x))
      norm.x = a.x;
    if (std::fabs(a.y) > std::fabs(norm.y))
      norm.y = a.y;
  }
  return norm;
}

// The kernel that “dispatches” the BVP solver for each copy.
template <class Solver>
__global__ void solveKernel(Solver solver, complex *fn, complex *an,
                            complex *cn, Parameters par, int ncopy) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= ncopy)
    return;
  int offset = id * par.nz;
  // Offset the pointers for each copy.
  solver.solve(id, fn + offset, par.alpha, par.beta, an + offset, cn + offset);
}

// Call the BVP solver (this mimics the original callBVPSolver)
template <class Container, class Solver>
void callBVPSolver(Container &fn, Container &cn, Solver &solver, Parameters par,
                   int ncopy) {
  int nz = par.nz;
  // Resize the input vectors so that the first copy is replicated over ncopy
  // copies.
  fn.resize(nz * ncopy);
  cn.resize(nz * ncopy);
  for (int i = 1; i < ncopy; i++) {
    std::copy(fn.begin(), fn.begin() + nz, fn.begin() + nz * i);
  }
  // In this example we assume that fn and cn live on the device as
  // thrust::device_vector.
  complex *d_fn = thrust::raw_pointer_cast(fn.data());
  // For simplicity, we use an auxiliary copy for the solver (as in the original
  // code).
  auto an = fn;
  complex *d_an = thrust::raw_pointer_cast(an.data());
  complex *d_cn = thrust::raw_pointer_cast(cn.data());
  // Get the GPU solver from the solver shared pointer.
  auto gpu_solver = solver->getGPUSolver();
  int nblocks = ncopy / 128 + 1;
  solveKernel<<<nblocks, 128>>>(gpu_solver, d_fn, d_an, d_cn, par, ncopy);
  cudaDeviceSynchronize();
}

auto createBVP(std::vector<real> klist, Parameters par) {
  auto bvp = std::make_shared<BVP::BatchedBVPHandlerReal>(
      klist, make_boundary_dispatcher<TopBoundaryConditions>(klist, par.H),
      make_boundary_dispatcher<BottomBoundaryConditions>(klist, par.H),
      klist.size(), par.H, par.nz);
  return bvp;
}

// Test the BVP solver with identical copies
TEST(BVPSolverTest, IdenticalCopies) {
  // ncopy is the number of copies solved in batch.
  constexpr int ncopy = 1000;
  Parameters par; // default: k=2, H=1, gamma=1, nz=16, alpha=beta={1,0}
  // (Note: the analytic solution is valid only for H=1)
  std::vector<real> klist(ncopy, par.k);
  // Create the BVP solver using the provided klist and parameters.
  auto solver = createBVP(klist, par);

  // Prepare the right–hand side using the analytic expression
  // f(z)=exp(-gamma*z*z)
  thrust::device_vector<complex> fn = computeRightHandSideExpGamma(par);
  thrust::device_vector<complex> cn = fn; // copy of f for storing the solution

  // Call the solver (which fills cn with the computed Chebyshev coefficients)
  callBVPSolver(fn, cn, solver, par, ncopy);

  // Additionally, check that all copies are identical.
  for (int j = 1; j < ncopy; j++) {
    for (int i = 0; i < par.nz; i++) {
      const complex &a = cn[j * par.nz + i];
      const complex &b = cn[i]; // first copy
      EXPECT_DOUBLE_EQ(a.x, b.x) << "Copy " << j << ", element " << i;
      EXPECT_DOUBLE_EQ(a.y, b.y) << "Copy " << j << ", element " << i;
    }
  }
}

// Known solution test (all copies use the same k)
TEST(BVPSolverTest, KnownSolution) {
  // ncopy is the number of copies solved in batch.
  constexpr int ncopy = 1;

  Parameters par; // default: k=2, H=1, gamma=1, nz=16, alpha=beta={1,0}
  // (Note: the analytic solution is valid only for H=1)
  std::vector<real> klist(ncopy, par.k);
  // Create the BVP solver using the provided klist and parameters.
  auto solver = createBVP(klist, par);

  // Prepare the right–hand side using the analytic expression
  // f(z)=exp(-gamma*z*z)
  thrust::device_vector<complex> fn = computeRightHandSideExpGamma(par);
  thrust::device_vector<complex> cn = fn; // copy of f for storing the solution

  // Call the solver (which fills cn with the computed Chebyshev coefficients)
  callBVPSolver(fn, cn, solver, par, ncopy);

  // Check the error against the analytic (theory) solution for each copy
  constexpr real errorThreshold = tolerance;
  for (size_t j = 0; j < klist.size(); j++) {
    Parameters localPar = par;
    localPar.k = klist[j];
    auto theory = computeSolutionCheb(localPar);
    complex norm = getErrorNormalization(theory);
    for (int i = 0; i < par.nz; i++) {
      // The index for copy j
      const complex &computed = cn[j * par.nz + i];
      const complex &expected = theory[i];
      // Compute a (componentwise) relative error:
      real errX =
          std::fabs((computed.x - expected.x) / (std::fabs(norm.x) + 1e-25));
      real errY =
          std::fabs((computed.y - expected.y) / (std::fabs(norm.y) + 1e-25));
      EXPECT_LE(errX, errorThreshold)
          << "Error in copy " << j << ", coeff " << i
          << ". Computed: " << computed.x << ", Expected: " << expected.x;
      EXPECT_LE(errY, errorThreshold)
          << "Error in copy " << j << ", coeff " << i
          << ". Computed: " << computed.y << ", Expected: " << expected.y;
    }
  }
}

// Random k test
TEST(BVPSolverTest, RandomK) {
  // ncopy is the number of copies solved in batch.
  constexpr int ncopy = 2;
  Parameters par;
  std::vector<real> klist(ncopy, 0);
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<real> distr(0.1, 2.0);
  for (auto &k : klist)
    k = distr(eng);

  auto solver = createBVP(klist, par);
  thrust::device_vector<complex> fn = computeRightHandSideExpGamma(par);
  thrust::device_vector<complex> cn = fn;
  callBVPSolver(fn, cn, solver, par, ncopy);

  constexpr real errorThreshold = tolerance;
  for (size_t j = 0; j < klist.size(); j++) {
    Parameters localPar = par;
    localPar.k = klist[j];
    auto theory = computeSolutionCheb(localPar);
    complex norm = getErrorNormalization(theory);
    for (int i = 0; i < par.nz; i++) {
      const complex &computed = cn[j * par.nz + i];
      const complex &expected = theory[i];
      real errX =
          std::fabs((computed.x - expected.x) / (std::fabs(norm.x) + 1e-15));
      real errY =
          std::fabs((computed.y - expected.y) / (std::fabs(norm.y) + 1e-15));
      EXPECT_LE(errX, errorThreshold)
          << "Random k test error in copy " << j << ", coeff " << i;
      EXPECT_LE(errY, errorThreshold)
          << "Random k test error in copy " << j << ", coeff " << i;
    }
  }
}

// Random RHS test (check that all copies are identical)
TEST(BVPSolverTest, RandomRHS) {
  // ncopy is the number of copies solved in batch.
  constexpr int ncopy = 31;
  Parameters par;
  std::vector<real> klist(ncopy, par.k); // use constant k for this test
  auto solver = createBVP(klist, par);
  thrust::device_vector<complex> fn = computeRightHandSideRandom(par);
  thrust::device_vector<complex> cn = fn;
  callBVPSolver(fn, cn, solver, par, ncopy);

  // Check that all copies are identical.
  for (int j = 1; j < ncopy; j++) {
    for (int i = 0; i < par.nz; i++) {
      const complex &a = cn[j * par.nz + i];
      const complex &b = cn[i]; // first copy
      EXPECT_DOUBLE_EQ(a.x, b.x)
          << "Random RHS: copy " << j << ", element " << i;
      EXPECT_DOUBLE_EQ(a.y, b.y)
          << "Random RHS: copy " << j << ", element " << i;
    }
  }
}
