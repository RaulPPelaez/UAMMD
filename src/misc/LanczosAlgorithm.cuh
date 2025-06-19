/*Raul P. Pelaez 2022. Lanczos Algotihm,
  Computes the matrix-vector product sqrt(M)·v using a recursive algorithm.
  For that, it requires a functor in which the () operator takes an output real*
array and an input real* (both device memory) as: inline void operator()(real*
in_v, real * out_Mv); This function must fill "out" with the result of
performing the M·v dot product- > out = M·a_v. If M has size NxN and the cost of
the dot product is O(M). The total cost of the algorithm is O(m·M). Where m <<
N. If M·v performs a dense M-V product, the cost of the algorithm would be
O(m·N^2). References: [1] Krylov subspace methods for computing hydrodynamic
interactions in Brownian dynamics simulations J. Chem. Phys. 137, 064106 (2012);
doi: 10.1063/1.4742347 Some notes: From what I have seen, this algorithm
converges to an error of ~1e-3 in a few steps (<5) and from that point a lot of
iterations are needed to lower the error. It usually achieves machine precision
in under 50 iterations. If the matrix does not have a sqrt (not positive
definite, not symmetric...) it will usually be reflected as a nan in the current
error estimation. An exception will be thrown in this case.
*/

#ifndef LANCZOSALGORITHM2_CUH
#define LANCZOSALGORITHM2_CUH
#include "LanczosAlgorithm/MatrixDot.h"
#include "LanczosAlgorithm/device_blas.h"
#include "LanczosAlgorithm/device_container.h"
#include "global/defines.h"
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
namespace uammd {
namespace lanczos {

struct Solver {
  Solver();

  ~Solver();

  // Given a Dotctor that computes a product M·v (where M is handled by Dotctor
  // ), computes Bv = sqrt(M)·v Returns the number of iterations required to
  // achieve the requested tolerance B = sqrt(M)
  int run(MatrixDot *dot, real *Bv, const real *v, real tolerance, int N,
          cudaStream_t st = 0);
  int run(MatrixDot &dot, real *Bv, const real *v, real tolerance, int N,
          cudaStream_t st = 0) {
    return run(&dot, Bv, v, tolerance, N, st);
  }
  int run(std::function<void(real *, real *)> dot, real *Bv, const real *v,
          real tolerance, int N, cudaStream_t st = 0) {
    auto lanczos_dot = createMatrixDotAdaptor(dot);
    return run(lanczos_dot, Bv, v, tolerance, N, st);
  }
  // Given a Dotctor that computes a product M·v (where M is handled by Dotctor
  // ), computes Bv = sqrt(M)·v Returns the residual after numberIterations
  // iterations B = sqrt(M)
  real runIterations(MatrixDot *dot, real *Bz, const real *z,
                     int numberIterations, int N);
  real runIterations(MatrixDot &dot, real *Bv, const real *v,
                     int numberIterations, int N) {
    return runIterations(&dot, Bv, v, numberIterations, N);
  }
  real runIterations(std::function<void(real *, real *)> dot, real *Bv,
                     const real *v, int numberIterations, int N) {
    auto lanczos_dot = createMatrixDotAdaptor(dot);
    return runIterations(lanczos_dot, Bv, v, numberIterations, N);
  }

  void setIterationHardLimit(int newLimit) {
    this->iterationHardLimit = newLimit;
  }

  int getLastRunRequiredSteps() { return this->lastRunRequiredSteps; }

private:
  // Increases storage space
  real computeError(real *Bz, int N, int iter);
  void registerRequiredStepsForConverge(int steps_needed);

  cublasHandle_t cublas_handle;
  cudaStream_t st = 0;
  device_container<real> oldBz;
  int check_convergence_steps;
  int iterationHardLimit = 200;
  int lastRunRequiredSteps = 0;
};
} // namespace lanczos
} // namespace uammd
#include "LanczosAlgorithm/LanczosAlgorithm.cu"
#endif
