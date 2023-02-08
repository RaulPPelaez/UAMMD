/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver
 */
#include"Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"
#include"utils.cuh"
#include"BVPStokes.cuh"
#include "utils/debugTools.h"
namespace uammd{
  namespace DPStokesSlab_ns{

    FluidData<complex> DPStokes::solveBVPVelocity(DataXYZ<complex> &gridForcesFourier,
						  cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] BVP solve");
      const int3 n = grid.cellDim;
      const int3 nfou = {n.x/2+1, n.y, 2*n.z-2};
      FluidData<complex> fluid(nfou);
      const int numberSystems = n.y*(n.x/2+1);
      auto tmp = DPStokesSlab_ns::setUpBVPKernelTemporalStorage(numberSystems, n.z);
      cached_vector<char> tmp_storage(tmp.allocationSize);
      const auto d_precomputedPressure = thrust::raw_pointer_cast(zeroModePressureChebyshevIntegrals.data());
      const auto d_precomputedVelocity = thrust::raw_pointer_cast(zeroModeVelocityChebyshevIntegrals.data());
      const int blockSize = 64;
      const int numberBlocks = numberSystems/blockSize+1;
      solveBVPVelocityD<<<numberBlocks, blockSize, 0,st>>>(bvpSolver->getGPUSolver(), n.x, n.y, n.z,
							   make_real2(Lx, Ly), H*0.5,
							   DataXYZPtr<complex>(gridForcesFourier),
							   fluid.getPointers(),
							   tmp,
							   thrust::raw_pointer_cast(tmp_storage.data()),
							   d_precomputedPressure, d_precomputedVelocity,
							   viscosity, mode);
      CudaCheckError();
      System::log<System::DEBUG2>("[DPStokes] BVP solve done");
      return fluid;
    }


  }
}
