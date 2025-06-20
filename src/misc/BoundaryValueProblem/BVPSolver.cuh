/*Raul P. Pelaez 2019. Boundary Value Problem solver

  Solves the equation:
  y''(z)-k y(z)^2 = f(z)
  With the boundary conditions:
  tfi y(1)' + tsi y(1) = \alpha ;      bfi y(-1)' + bsi y(-1) = \beta ;
  Where tfi, tsi, bfi and bsi are some arbitrary factors.

  In Chebyshev space.
  It uses a Schur complement approach so that defining:

  y''(z) = \sum_n a_n T_n(z)
  y'(z) = \sum_n d_n T_n(z)
  y(z) = \sum_n c_n T_n(z)
  f(z) = \sum_n f_n T_n(z)

  And using the relations between the different Chebyshev coefficients the
  system can be written in matrix form as:

  ( A B ; C D ) ( a0; a1; ...; c0; d0 ) = ( f0; f1; ...; \alpha; \beta )

  The A matrix happens to be pentadiagonal, C and D encode the boundary
  conditions. This code solves the system above in two steps: Subsystem solve to
  obtain c0 and d0: ( C A^-1 B - D )(c0; d0) = C A^-1 (f0; f1;...) - (\alpha;
  \beta) Pentadiagonal solve to obtain a0,a1,...: A (a0; a1; ...) = (f0; f1;
  ...) - B (c0; d0)

  From the a_n coefficients the second integral coefficients c_n can be obtained
  through a simple recursive relation.
*/
#ifndef BVPSOLVER_CUH
#define BVPSOLVER_CUH
#include "System/System.h"
#include "misc/BoundaryValueProblem/BVPMemory.cuh"
#include "misc/BoundaryValueProblem/BVPSchurComplementMatrices.cuh"
#include "misc/BoundaryValueProblem/KBPENTA.cuh"
#include "misc/BoundaryValueProblem/MatrixUtils.h"
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <utils/exception.h>
#include <vector>
namespace uammd {
namespace BVP {

namespace detail {
template <typename U> class SubsystemSolver {
  int nz;
  real H;
  StorageHandle<U> CinvA_storage;
  StorageHandle<U> CinvABmD_storage;

public:
  SubsystemSolver(int nz, real H) : nz(nz), H(H) {}

  void registerRequiredStorage(StorageRegistration &memoryManager) {
    CinvA_storage = memoryManager.registerStorageRequirement<U>(2 * nz + 2);
    CinvABmD_storage = memoryManager.registerStorageRequirement<U>(4);
  }

  template <class TopBC, class BottomBC>
  void precompute(U k, const TopBC &top, const BottomBC &bottom,
                  StorageRetriever &memoryManager) {
    auto CinvA_it = memoryManager.retrieveStorage(CinvA_storage);
    auto CinvABmD_it = memoryManager.retrieveStorage(CinvABmD_storage);
    auto invA = computeInverseSecondIntegralMatrix(k, H, nz);
    SchurBoundaryCondition_impl<U> bcs(nz, H);
    auto CandD = bcs.computeBoundaryConditionMatrix(top, bottom);
    U D[4] = {CandD[2 * nz], CandD[2 * nz + 1], CandD[2 * nz + 2],
              CandD[2 * nz + 3]};
    auto CinvA = matmul(CandD, nz, 2, invA, nz, nz);
    std::copy(CinvA.begin(), CinvA.end(), CinvA_it);
    U CinvAB[4];
    U B00 = -k * k * H * H;
    U B11 = -k * k * H * H;
    CinvAB[0] = CinvA[0] * B00;
    CinvAB[1] = CinvA[1] * B11;
    CinvAB[2] = CinvA[0 + nz] * B00;
    CinvAB[3] = CinvA[1 + nz] * B11;
    fori(0, 4) CinvABmD_it[i] = CinvAB[i] - D[i];
  }

  template <class T, class FnIterator>
  __device__ thrust::pair<T, T> solve(const FnIterator &fn, T alpha, T beta,
                                      StorageRetriever &memoryManager) {
    const auto CinvA = memoryManager.retrieveStorage(CinvA_storage);
    const auto CinvAfmab = computeRightHandSide(alpha, beta, fn, CinvA);
    const auto CinvABmD_it = memoryManager.retrieveStorage(CinvABmD_storage);
    U CinvABmD[4] = {CinvABmD_it[0], CinvABmD_it[1], CinvABmD_it[2],
                     CinvABmD_it[3]};

    const auto c0d0 = solveSubsystem(CinvABmD, CinvAfmab);
    return c0d0;
  }

private:
  template <class T>
  __device__ thrust::pair<T, T>
  solveSubsystem(U CinvABmD[4], thrust::pair<T, T> CinvAfmab) const {
    auto c0d0 = solve2x2System<T>(CinvABmD, CinvAfmab);
    return c0d0;
  }

  template <class T, class FnIterator, class CinvAIterator>
  __device__ thrust::pair<T, T>
  computeRightHandSide(T alpha, T beta, const FnIterator &fn,
                       const CinvAIterator &CinvA) const {
    T CinvAfmab[2];
    CinvAfmab[0] = CinvAfmab[1] = T();
    for (int i = 0; i < nz; i++) {
      CinvAfmab[0] += CinvA[i] * fn[i];
      CinvAfmab[1] += CinvA[nz + i] * fn[i];
    }
    CinvAfmab[0] -= alpha;
    CinvAfmab[1] -= beta;
    return thrust::make_pair(CinvAfmab[0], CinvAfmab[1]);
  }
};

template <typename U> class PentadiagonalSystemSolver {
  int nz;
  real H;
  KBPENTA_mod<U> pentasolve;

public:
  PentadiagonalSystemSolver(int nz, real H) : nz(nz), H(H), pentasolve(nz) {}

  void registerRequiredStorage(StorageRegistration &memoryManager) {
    pentasolve.registerRequiredStorage(memoryManager);
  }

  void precompute(U k, StorageRetriever &memoryManager) {
    U diagonal[nz];
    U diagonal_p2[nz];
    diagonal_p2[nz - 2] = diagonal_p2[nz - 1] = 0;
    U diagonal_m2[nz];
    diagonal_m2[0] = diagonal_m2[1] = 0;
    SecondIntegralMatrix sim(nz);
    const U kH2 = k * k * H * H;
    for (int i = 0; i < nz; i++) {
      diagonal[i] = 1.0 - kH2 * sim.getElement(i, i);
      if (i < nz - 2)
        diagonal_p2[i] = -kH2 * sim.getElement(i + 2, i);
      if (i > 1)
        diagonal_m2[i] = -kH2 * sim.getElement(i - 2, i);
    }
    pentasolve.store(diagonal, diagonal_p2, diagonal_m2, memoryManager);
  };

  template <class FnIterator, class AnIterator>
  __device__ void solve(const FnIterator &fnMinusBc0d0, AnIterator &an,
                        StorageRetriever &memoryManager) {
    pentasolve.solve(an, fnMinusBc0d0, memoryManager);
  }
};
} // namespace detail

template <typename U> class BoundaryValueProblemSolver_impl {
  detail::PentadiagonalSystemSolver<U> pent;
  detail::SubsystemSolver<U> sub;
  StorageHandle<U> waveVector;
  int nz;
  real H;

public:
  BoundaryValueProblemSolver_impl(int nz, real H)
      : nz(nz), H(H), sub(nz, H), pent(nz, H) {}

  void registerRequiredStorage(StorageRegistration &mem) {
    waveVector = mem.registerStorageRequirement<U>(1);
    sub.registerRequiredStorage(mem);
    pent.registerRequiredStorage(mem);
  }

  template <class TopBC, class BottomBC>
  void precompute(StorageRetriever &mem, U k, const TopBC &top,
                  const BottomBC &bot) {
    auto k_access = mem.retrieveStorage(waveVector);
    k_access[0] = k;
    pent.precompute(k, mem);
    sub.precompute(k, top, bot, mem);
  }

  template <class T, class FnIterator, class AnIterator, class CnIterator>
  __device__ void solve(FnIterator &fn, T alpha, T beta, AnIterator &an,
                        CnIterator &cn, StorageRetriever &mem) {
    const auto k = *(mem.retrieveStorage(waveVector));
    T c0, d0;
    thrust::tie(c0, d0) = sub.solve(fn, alpha, beta, mem);
    const auto kH2 = k * k * H * H;
    fn[0] += kH2 * c0;
    fn[1] += kH2 * d0;
    pent.solve(fn, an, mem);
    fn[0] -= kH2 * c0;
    fn[1] -= kH2 * d0;
    fillSecondIntegralCoefficients(an, cn, c0, d0);
  }

private:
  template <class T, class AnIterator, class CnIterator>
  __device__ void fillSecondIntegralCoefficients(const AnIterator &an,
                                                 CnIterator &cn, T c0, T d0) {
    SecondIntegralMatrix si(nz);
    for (int i = 0; i < nz; i++) {
      cn[i] = si.computeSecondIntegralCoefficient(i, an, c0, d0) * H * H;
    }
  }
};

using BoundaryValueProblemSolverReal = BoundaryValueProblemSolver_impl<real>;
using BoundaryValueProblemSolverComplex =
    BoundaryValueProblemSolver_impl<thrust::complex<real>>;

template <typename U> class BatchedBVPHandler_impl;

template <typename U> struct BatchedBVPGPUSolver_impl {
private:
  int numberSystems;
  BoundaryValueProblemSolver_impl<U> bvpSolver;
  char *gpuMemory;
  friend class BatchedBVPHandler_impl<U>;
  BatchedBVPGPUSolver_impl(int numberSystems,
                           BoundaryValueProblemSolver_impl<U> bvpSolver,
                           char *raw)
      : numberSystems(numberSystems), bvpSolver(bvpSolver), gpuMemory(raw) {}

public:
  template <class T, class FnIterator, class AnIterator, class CnIterator>
  __device__ void solve(int instance, const FnIterator &fn, T alpha, T beta,
                        const AnIterator &an, const CnIterator &cn) {
    StorageRetriever memoryAccess(numberSystems, instance, gpuMemory);
    bvpSolver.solve(fn, alpha, beta, an, cn, memoryAccess);
  }
};

using BatchedBVPGPUSolverReal = BatchedBVPGPUSolver_impl<real>;
using BatchedBVPGPUSolverComplex =
    BatchedBVPGPUSolver_impl<thrust::complex<real>>;

template <typename U> class BatchedBVPHandler_impl {
  int numberSystems;
  BoundaryValueProblemSolver_impl<U> bvp;
  thrust::device_vector<char> gpuMemory;

public:
  template <class WaveVectorIterator, class BatchedTopBC, class BatchedBottomBC>
  BatchedBVPHandler_impl(const WaveVectorIterator &klist, BatchedTopBC top,
                         BatchedBottomBC bot, int numberSystems, real H, int nz)
      : numberSystems(numberSystems), bvp(nz, H) {
    precompute(klist, top, bot);
  }

  BatchedBVPGPUSolver_impl<U> getGPUSolver() {
    auto raw = thrust::raw_pointer_cast(gpuMemory.data());
    BatchedBVPGPUSolver_impl<U> d_solver(numberSystems, bvp, raw);
    return d_solver;
  }

private:
  template <class WaveVectorIterator, class BatchedTopBC, class BatchedBottomBC>
  void precompute(const WaveVectorIterator &klist, BatchedTopBC &top,
                  BatchedBottomBC &bot) {
    auto allocationSize = countRequiredMemory();
    std::vector<char> cpuMemory(allocationSize);
    for (int i = 0; i < numberSystems; i++) {
      StorageRetriever memoryAccess(numberSystems, i, cpuMemory.data());
      bvp.precompute(memoryAccess, klist[i], top[i], bot[i]);
    }
    gpuMemory = cpuMemory;
  }

  size_t countRequiredMemory() {
    StorageRegistration storageCounter(numberSystems);
    bvp.registerRequiredStorage(storageCounter);
    return storageCounter.getRequestedStorageBytes();
  }
};

using BatchedBVPHandlerReal = BatchedBVPHandler_impl<real>;
using BatchedBVPHandlerComplex = BatchedBVPHandler_impl<thrust::complex<real>>;

} // namespace BVP

} // namespace uammd

#endif
