/*Raul P. Pelaez 2020. Boundary Value Problem solver for DP Poisson.
*/
#ifndef BVPPOISSONSLAB_CUH
#define BVPPOISSONSLAB_CUH
#include "misc/BoundaryValueProblem/BVPSolver.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/BoundaryConditions.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/FastChebyshevTransform.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include "global/defines.h"
#include "utils/utils.h"
#include <thrust/iterator/counting_iterator.h>

namespace uammd{
  namespace DPPoissonSlab_ns{
    namespace bvp_ns{

      struct BVPKernelTemporalStorage{
	BVP::StorageHandle<cufftComplex> potentialChebyshevCoefficients;
	BVP::StorageHandle<cufftComplex> secondDerivativeChebyshevCoefficients;
	BVP::StorageHandle<cufftComplex> firstDerivativeChebyshevCoefficients;
	size_t allocationSize;
      };

      BVPKernelTemporalStorage setUpBVPKernelTemporalStorage(int numberConcurrentThreads, int nz){
	BVP::StorageRegistration mem(numberConcurrentThreads);
	BVPKernelTemporalStorage tmp;
	tmp.potentialChebyshevCoefficients = mem.registerStorageRequirement<cufftComplex>(nz);
	tmp.secondDerivativeChebyshevCoefficients = mem.registerStorageRequirement<cufftComplex>(nz);
	tmp.firstDerivativeChebyshevCoefficients = mem.registerStorageRequirement<cufftComplex>(nz);
	tmp.allocationSize = mem.getRequestedStorageBytes();
	return tmp;
      }

      template<class FNIterator, class OutputIterator>
      __device__ void firstDerivativeChebyshevCoefficients(FNIterator &fn, OutputIterator &out,
							   int nz, real H){
	cufftComplex Dfnp2 = cufftComplex();
	cufftComplex Dfnp1 = cufftComplex();
	for(int i = nz-1; i>=0; i--){
	  cufftComplex Dfni = cufftComplex();
	  if(i<=nz-2) Dfni = Dfnp2 + real(2.0)*(i+1)*fn[i+1]/H;
	  if(i==0) Dfni *= real(0.5);
	  out[i] = Dfni;
	  Dfnp2 = Dfnp1;
	  if(i<=nz-2){
	    Dfnp1 = Dfni;
	  }
	}
      }
      //TODO: There is no need for firstDerivativeChebyshevCoefficients, it could be directly stored in gridFieldPotential[i].z
      template<class BVPSolver>
      __global__ void solveBVPFieldPotential(BVPSolver bvp, int nkx, int nky, int nz,
					  real2 Lxy, real H,
					  cufftComplex* gridCharges,
					  cufftComplex4 *gridFieldPotential_raw,
					  BVPKernelTemporalStorage tmp,
					  char* tmp_storage,
					  real permitivity){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
	if(id >= nky*(nkx/2+1)){
	  return;
	}
	Index3D indexer(nkx/2+1, nky, 2*nz-2);
	auto fn = make_third_index_iterator(gridCharges, ik.x, ik.y, indexer);
	 fori(0, nz){
	   fn[i] /= permitivity;
	 }
	BVP::StorageRetriever stor(nky*(nkx/2+1), id, tmp_storage);
	auto potential = stor.retrieveStorage(tmp.potentialChebyshevCoefficients);
	auto an = stor.retrieveStorage(tmp.secondDerivativeChebyshevCoefficients);
	bvp.solve(id, fn,  cufftComplex(), cufftComplex(),  an, potential);
	auto dphi_dz = stor.retrieveStorage(tmp.firstDerivativeChebyshevCoefficients);
	firstDerivativeChebyshevCoefficients(potential, dphi_dz, nz, H);
	auto gridFieldPotential = make_third_index_iterator(gridFieldPotential_raw, ik.x, ik.y, indexer);
	const IndexToWaveNumber id2wn(nkx, nky);
	const WaveNumberToWaveVector wn2wv(Lxy);
	const real2 kvec = wn2wv(id2wn(id));
	const bool isPairedX = ik.x != (nkx/2);
	const bool isPairedY = ik.y != (nky/2);
	fori(0, nz){
	  const cufftComplex poti = potential[i];
	  gridFieldPotential[i].x = -cufftComplex({-kvec.x*poti.y, kvec.x*poti.x})*isPairedX;
	  gridFieldPotential[i].y = -cufftComplex({-kvec.y*poti.y, kvec.y*poti.x})*isPairedY;
	  gridFieldPotential[i].z = {-dphi_dz[i].x, -dphi_dz[i].y};
	  gridFieldPotential[i].w = poti;
	  fn[i] *= permitivity;
	}
      }
    }

    class BVPPoissonSlab{
      std::shared_ptr<BVP::BatchedBVPHandlerReal> bvpSolver;
      real2 Lxy;
      real H;
      int3 cellDim;
      real permitivity;

    public:

      struct Parameters{
	real2 Lxy;
	real H;
	int3 cellDim;
	real permitivity;
      };

      BVPPoissonSlab(Parameters par):
        Lxy(par.Lxy), H(par.H), cellDim(par.cellDim),
      	permitivity(par.permitivity){
	System::log<System::DEBUG>("[BVP] Initialized");
	initializeBoundaryValueProblemSolver();
      }

      template<class Container>
      cached_vector<cufftComplex4> solveFieldPotential(Container &gridChargesFourier, cudaStream_t st){
	System::log<System::DEBUG2>("[DPPoissonSlab] BVP solve");
	cufftComplex* d_gridChargesFourier = thrust::raw_pointer_cast(gridChargesFourier.data());
	const int3 n = cellDim;
	cached_vector<cufftComplex4> gridFieldPotentialFourier((n.x/2+1)*n.y*(2*n.z-2));
	cufftComplex4* d_gridFieldPotentialFourier = thrust::raw_pointer_cast(gridFieldPotentialFourier.data());
	const int numberSystems = n.y*(n.x/2+1);
	auto tmp = bvp_ns::setUpBVPKernelTemporalStorage(numberSystems, n.z);
	cached_vector<char> tmp_storage(tmp.allocationSize);
	auto tmp_storage_ptr = thrust::raw_pointer_cast(tmp_storage.data());
	auto gpuSolver = bvpSolver->getGPUSolver();
	const int blockSize = 64;
	const int numberBlocks = numberSystems/blockSize+1;
        bvp_ns::solveBVPFieldPotential<<<numberBlocks, blockSize, 0, st>>>(gpuSolver, n.x, n.y, n.z,
									Lxy, H,
									d_gridChargesFourier,
									d_gridFieldPotentialFourier,
									tmp,
									tmp_storage_ptr,
									permitivity);
	CudaCheckError();
	System::log<System::DEBUG2>("[DPPoissonSlab] BVP solve done");
	return gridFieldPotentialFourier;
      }

    private:

      void initializeBoundaryValueProblemSolver(){
	System::log<System::DEBUG>("[DPPoissonSlab] Initializing BVP solver");
	const int2 nk = {cellDim.x, cellDim.y};
	auto klist = DPPoissonSlab_ns::make_wave_vector_modulus_iterator(nk, Lxy);
	auto topBC = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
						     BoundaryConditionsDispatch<TopBoundaryConditions, decltype(klist)>(klist, H));
	auto bottomBC = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
						   BoundaryConditionsDispatch<BottomBoundaryConditions, decltype(klist)>(klist, H));

	int numberSystems = (nk.x/2+1)*nk.y;
	this->bvpSolver = std::make_shared<BVP::BatchedBVPHandlerReal>(klist, topBC, bottomBC, numberSystems, H, cellDim.z);
	CudaCheckError();
      }
    };
  }
}

#endif
