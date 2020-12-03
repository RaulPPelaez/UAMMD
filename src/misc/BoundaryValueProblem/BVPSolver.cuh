/*Raul P. Pelaez 2019. Boundary Value Problem solver

  Solves the equation:
  y''(z)-k y(z)^2 = f(z)
  With the boundary conditions:
  tfi y(1)' + tsi k y(1) = \alpha ;      bfi y(-1)' - bsi k y(-1) = \beta ;
  Where tfi, tsi, bfi and bsi are some arbitrary factors.

  In Chebyshev space.
  It uses a Schur complement approach so that defining:

  y''(z) = \sum_n a_n T_n(z)
  y'(z) = \sum_n d_n T_n(z)
  y(z) = \sum_n c_n T_n(z)
  f(z) = \sum_n f_n T_n(z)

  And using the relations between the different Chebyshev coefficients the system can be written in matrix form as>

  ( A B ; C D ) ( a0; a1; ...; c0; d0 ) = ( f0; f1; ...; \alpha; \beta )

  The A matrix happens to be pentadiagonal, C and D encode the boundary conditions.
  This code solves the system above in two steps:
  Subsystem solve to obtain c0 and d0:
  ( C A^-1 B - D )(c0; d0) = C A^-1 (f0; f1;...) - (\alpha; \beta)
  Pentadiagonal solve to obtain a0,a1,...:
  A (a0; a1; ...) = (f0; f1; ...) - B (c0; d0)

  From the a_n coefficients the second integral coefficients c_n can be obtained through a simple recursive relation.
*/
#ifndef BVPSOLVER_CUH
#define BVPSOLVER_CUH
#include"System/System.h"
#include<utils/exception.h>
#include"misc/BoundaryValueProblem/MatrixUtils.h"
#include"misc/BoundaryValueProblem/BVPSchurComplementMatrices.cuh"
#include"misc/BoundaryValueProblem/KBPENTA.cuh"
#include "misc/BoundaryValueProblem/BVPMemory.cuh"
#include<thrust/pair.h>
#include<thrust/device_vector.h>
#include<vector>
namespace uammd{
  namespace BVP{

    namespace detail{
      class SubsystemSolver{
	int nz;
	real H;
	StorageHandle<real> CinvA_storage;
	StorageHandle<real4> CinvABmD_storage;

      public:

	SubsystemSolver(int nz, real H):nz(nz), H(H){}

	void registerRequiredStorage(StorageRegistration &memoryManager){
	  CinvA_storage =    memoryManager.registerStorageRequirement<real>(2*nz+2);
	  CinvABmD_storage = memoryManager.registerStorageRequirement<real4>(1);
	}

	template<class TopBC, class BottomBC>
	void precompute(real k, const TopBC &top, const BottomBC &bottom, StorageRetriever &memoryManager){
	  auto CinvA_it = memoryManager.retrieveStorage(CinvA_storage);
	  auto CinvABmD_it = memoryManager.retrieveStorage(CinvABmD_storage);
	  auto invA = computeInverseSecondIntegralMatrix(k, H, nz);
	  SchurBoundaryCondition bcs(nz, H);
	  auto CandD = bcs.computeBoundaryConditionMatrix(k, top, bottom);
	  real4 D = make_real4(CandD[2*nz], CandD[2*nz+1], CandD[2*nz+2], CandD[2*nz+3]);
	  auto CinvA = matmul(CandD, nz, 2, invA, nz, nz);
	  std::copy(CinvA.begin(), CinvA.end(), CinvA_it);
	  real4 CinvAB;
	  real B00 = -k*k*H*H;
	  real B11 = -k*k*H*H;
	  CinvAB.x = CinvA[0]*B00;
	  CinvAB.y = CinvA[1]*B11;
	  CinvAB.z = CinvA[0+nz]*B00;
	  CinvAB.w = CinvA[1+nz]*B11;
	  CinvABmD_it[0] = CinvAB - D;
	}

	template<class T, class FnIterator>
	__device__ thrust::pair<T,T> solve(const FnIterator& fn,
					   T alpha, T beta,
					   StorageRetriever &memoryManager){
	  const auto CinvA = memoryManager.retrieveStorage(CinvA_storage);
	  const auto CinvAfmab = computeRightHandSide(alpha, beta, fn, CinvA);
	  const real4 CinvABmD = *(memoryManager.retrieveStorage(CinvABmD_storage));
	  const auto c0d0 = solveSubsystem(CinvABmD, CinvAfmab);
	  return c0d0;
	}

      private:

	template<class T>
	__device__ thrust::pair<T,T> solveSubsystem(real4 CinvABmD, thrust::pair<T,T> CinvAfmab) const{
	  const real det  = CinvABmD.x*CinvABmD.w - CinvABmD.y*CinvABmD.z;
	  const T d0 = (CinvAfmab.second*CinvABmD.x - CinvAfmab.first*CinvABmD.z)/det;
	  const T c0 = (CinvAfmab.first - d0*CinvABmD.y)/CinvABmD.x;
	  return thrust::make_pair(c0, d0);
	}

	template<class T, class FnIterator, class CinvAIterator>
	__device__ thrust::pair<T,T> computeRightHandSide(T alpha, T beta, const FnIterator& fn,
							  const CinvAIterator &CinvA) const{
	  T CinvAfmab[2];
	  CinvAfmab[0] = CinvAfmab[1] = T();
	  for(int i = 0; i<nz; i++){
	    CinvAfmab[0] += CinvA[i]*fn[i];
	    CinvAfmab[1] += CinvA[nz+i]*fn[i];

	  }
	  CinvAfmab[0] -= alpha;
	  CinvAfmab[1] -= beta;
	  return thrust::make_pair(CinvAfmab[0], CinvAfmab[1]);
	}

      };
      
      class PentadiagonalSystemSolver{
	int nz;
	real H;
	KBPENTA_mod pentasolve;
      public:

	PentadiagonalSystemSolver(int nz, real H):
	  nz(nz), H(H), pentasolve(nz){}

	void registerRequiredStorage(StorageRegistration &memoryManager){
	  pentasolve.registerRequiredStorage(memoryManager);
	}

	void precompute(real k, StorageRetriever &memoryManager){
	  real diagonal[nz];
	  real diagonal_p2[nz]; diagonal_p2[nz-2] = diagonal_p2[nz-1] = 0;
	  real diagonal_m2[nz]; diagonal_m2[0] = diagonal_m2[1] = 0;
	  SecondIntegralMatrix sim(nz);
	  const real kH2 = k*k*H*H;
	  for(int i = 0; i<nz; i++){
	    diagonal[i] = 1.0 - kH2*sim.getElement(i,i);
	    if(i<nz-2) diagonal_p2[i] = -kH2*sim.getElement(i+2, i);
	    if(i>1)   diagonal_m2[i] = -kH2*sim.getElement(i-2, i);
	  }
	  pentasolve.store(diagonal, diagonal_p2, diagonal_m2, memoryManager);
	};

	template<class FnIterator, class AnIterator>
	__device__ void solve(const FnIterator& fnMinusBc0d0, AnIterator& an, StorageRetriever &memoryManager){
	  pentasolve.solve(an, fnMinusBc0d0, memoryManager);
	}

      };
    }

    class BoundaryValueProblemSolver{
      detail::PentadiagonalSystemSolver pent;
      detail::SubsystemSolver sub;
      StorageHandle<real> waveVector;
      int nz;
      real H;
    public:
      BoundaryValueProblemSolver(int nz, real H): nz(nz), H(H), sub(nz, H), pent(nz, H){}

      void registerRequiredStorage(StorageRegistration &mem){
	waveVector = mem.registerStorageRequirement<real>(1);
	sub.registerRequiredStorage(mem);
	pent.registerRequiredStorage(mem);
      }

      template<class TopBC, class BottomBC>
      void precompute(StorageRetriever &mem, real k, const TopBC &top, const BottomBC &bot){
	auto k_access = mem.retrieveStorage(waveVector);
	k_access[0] = k;
	pent.precompute(k, mem);
	sub.precompute(k, top, bot, mem);
      }

      template<class T, class FnIterator, class AnIterator, class CnIterator>
      __device__ void solve(FnIterator& fn,
			    T alpha, T beta,
			    AnIterator& an,
			    CnIterator& cn,
			    StorageRetriever &mem){
        const real k = *(mem.retrieveStorage(waveVector));
	T c0, d0;
	thrust::tie(c0, d0) = sub.solve(fn, alpha, beta, mem);
	const real kH2 = k*k*H*H;
	fn[0] += kH2*c0;
	fn[1] += kH2*d0;
	pent.solve(fn, an, mem);
	fn[0] -= kH2*c0;
	fn[1] -= kH2*d0;
	fillSecondIntegralCoefficients(an, cn, c0, d0);
      }

    private:

      template<class T, class AnIterator, class CnIterator>
      __device__ void fillSecondIntegralCoefficients(const AnIterator &an, CnIterator &cn, T c0, T d0){
	SecondIntegralMatrix si(nz);
	for(int i = 0; i<nz; i++){
	  cn[i] = si.computeSecondIntegralCoefficient(i, an, c0, d0)*H*H;
	}
      }

    };

    class BatchedBVPHandler;

    struct BatchedBVPGPUSolver{
    private:
      int numberSystems;
      BoundaryValueProblemSolver bvpSolver;
      char* gpuMemory;
      friend class BatchedBVPHandler;
      BatchedBVPGPUSolver(int numberSystems, BoundaryValueProblemSolver bvpSolver, char *raw):
	numberSystems(numberSystems), bvpSolver(bvpSolver), gpuMemory(raw){}
    public:

      template<class T, class FnIterator, class AnIterator, class CnIterator>
      __device__ void solve(int instance,
			    const FnIterator& fn,
			    T alpha, T beta,
			    AnIterator& an,
			    CnIterator& cn){
	StorageRetriever memoryAccess(numberSystems, instance, gpuMemory);
	bvpSolver.solve(fn, alpha, beta, an, cn, memoryAccess);
      }

    };

    class BatchedBVPHandler{
      int numberSystems;
      BoundaryValueProblemSolver bvp;
      thrust::device_vector<char> gpuMemory;
    public:

      template<class WaveVectorIterator, class BatchedTopBC, class BatchedBottomBC>
      BatchedBVPHandler(const WaveVectorIterator &klist,
			BatchedTopBC top, BatchedBottomBC bot,
			int numberSystems, real H, int nz):
	numberSystems(numberSystems),
	bvp(nz, H){
	precompute(klist, top, bot);
      }

      BatchedBVPGPUSolver getGPUSolver(){
	auto raw = thrust::raw_pointer_cast(gpuMemory.data());
	BatchedBVPGPUSolver d_solver(numberSystems, bvp, raw);
	return d_solver;
      }

    private:

      template<class WaveVectorIterator, class BatchedTopBC, class BatchedBottomBC>
      void precompute(const WaveVectorIterator &klist, BatchedTopBC &top, BatchedBottomBC &bot){
	auto allocationSize = countRequiredMemory();
	std::vector<char> cpuMemory(allocationSize);
	for(int i = 0; i<numberSystems; i++){
	  StorageRetriever memoryAccess(numberSystems, i, cpuMemory.data());
	  bvp.precompute(memoryAccess, klist[i], top[i], bot[i]);
	}
	gpuMemory = cpuMemory;
      }

      size_t countRequiredMemory(){
	StorageRegistration storageCounter(numberSystems);
	bvp.registerRequiredStorage(storageCounter);
	return storageCounter.getRequestedStorageBytes();
      }

    };

  }

}


#endif


























