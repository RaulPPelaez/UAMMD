#ifndef KBPENTA_CUH
#define KBPENTA_CUH
#include"BVPMemory.cuh"
#include<vector>
namespace uammd{
  namespace BVP{

//Algorithm adapted from  http://dx.doi.org/10.1080/00207160802326507 for a special case of only three diagonals being non zero
    class KBPENTA_mod{
      StorageHandle<real> storageHandle;
      int nz;
    public:

      KBPENTA_mod(int nz): nz(nz){}

      void registerRequiredStorage(StorageRegistration &memoryManager){
	storageHandle =  memoryManager.registerStorageRequirement<real>(3*nz+2);
      }

      void store(real *diagonal, real *diagonal_p2, real *diagonal_m2, StorageRetriever &memoryManager){
	auto storage = memoryManager.retrieveStorage(storageHandle);
	std::vector<real> beta(nz+1, 0);
	beta[0] = 0;
	beta[1] = diagonal[nz-nz];
	beta[2] = diagonal[nz-(nz-1)];
	for(int i = 3; i<=nz; i++){
	  beta[i] = diagonal[nz-(nz-i+1)] - diagonal_m2[nz-(nz-i+1)]*diagonal_p2[nz-(nz-i+3)]/beta[i-2];
	}

	for(int i = 0; i<=nz; i++){
	  storage[i] = beta[i];
	  storage[i+nz+1] = diagonal_p2[i];
	  storage[i+2*nz+1] = diagonal_m2[i];
	}
      }

      template<class XIterator, class YIterator>
      __device__ void solve(const XIterator &x,
			    const YIterator &rightHandSide,
			    StorageRetriever &memoryManager){
	auto storage = memoryManager.retrieveStorage(storageHandle);
	const auto beta = storage;
	const auto diagonal_p2 = storage + nz + 1;
	const auto diagonal_m2 = storage + 2*nz + 1;
	x[0] = rightHandSide[nz - nz];
	x[1] = rightHandSide[nz - (nz-1)];
	for(int i = 2; i<nz; i++){
	  x[i] = rightHandSide[nz - (nz-i)] - diagonal_m2[nz - (nz-i)] * x[i-2] / beta[i-1];
	}
	x[nz-1] = x[nz-1]/beta[nz];
	x[nz-2] = x[nz-2]/beta[nz-1];
	for(int i = nz-3; i>=0; i--){
	  x[i] = (x[i] - diagonal_p2[i]*x[i+2])/beta[i+1];
	}
      }

    };

  }
}
#endif
