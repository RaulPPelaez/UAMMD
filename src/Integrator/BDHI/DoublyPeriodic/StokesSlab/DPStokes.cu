/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver
 */
#include"Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"
#include "misc/IBM.cuh"
#include"utils.cuh"
#include"BVPStokes.cuh"
namespace uammd{
  namespace DPStokesSlab_ns{

    cached_vector<real4> DPStokes::Mdot(real4* pos, real4* forces, int numberParticles, cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] Far field computation");
      auto gridData = spreadForces(pos, forces, numberParticles, st);
      auto gridDataFourier = fct->forwardTransform(gridData, st);
      solveBVPVelocity(gridDataFourier, st);
      gridData = fct->inverseTransform(gridDataFourier, st);
      auto particleVelocities = interpolateVelocity(gridData, pos, numberParticles, st);
      return particleVelocities;
    }

    cached_vector<real4> DPStokes::spreadForces(real4* pos, real4* forces, int numberParticles, cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] Spreading forces");
      const int3 n = grid.cellDim;
      cached_vector<real4> gridForce(2*(n.x/2+1)*n.y*(2*n.z-2));
      std::fill(gridForce.begin(), gridForce.end(), real4());
      auto d_gridForce = thrust::raw_pointer_cast(gridForce.data());
      IBM<Kernel, Grid> ibm(kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
      ibm.spread(pos, forces, d_gridForce, numberParticles, st);
      CudaCheckError();
      return gridForce;
    }

    void DPStokes::solveBVPVelocity(cached_vector<cufftComplex4> &gridForcesFourier, cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] BVP solve");
      cufftComplex4* d_gridForceFourier = thrust::raw_pointer_cast(gridForcesFourier.data());
      cufftComplex4* d_gridVelocityFourier = thrust::raw_pointer_cast(gridForcesFourier.data());
      const int3 n = grid.cellDim;
      const int numberSystems = n.y*(n.x/2+1);
      auto tmp = DPStokesSlab_ns::setUpBVPKernelTemporalStorage(numberSystems, n.z);
      cached_vector<char> tmp_storage(tmp.allocationSize);
      const auto d_precomputedPressure = thrust::raw_pointer_cast(zeroModePressureChebyshevIntegrals.data());
      const auto d_precomputedVelocity = thrust::raw_pointer_cast(zeroModeVelocityChebyshevIntegrals.data());
      const real2 Lxy = make_real2(box.boxSize);
      const real H = box.boxSize.z*0.5;
      const int blockSize = 64;
      const int numberBlocks = numberSystems/blockSize+1;
      solveBVPVelocityD<<<numberBlocks, blockSize, 0,st>>>(bvpSolver->getGPUSolver(), n.x, n.y, n.z,
							  Lxy, H,
							  d_gridForceFourier,
							  d_gridVelocityFourier,
							  tmp,
							  thrust::raw_pointer_cast(tmp_storage.data()),
							  d_precomputedPressure, d_precomputedVelocity,
							  viscosity);
      CudaCheckError();
      System::log<System::DEBUG2>("[DPStokes] BVP solve done");
    }

    cached_vector<real4> DPStokes::interpolateVelocity(cached_vector<real4> &gridData, real4* pos, int numberParticles, cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] Interpolating forces and energies");
      cached_vector<real4> particleVelsAndPressure(numberParticles);
      thrust::fill(thrust::cuda::par.on(st), particleVelsAndPressure.begin(), particleVelsAndPressure.end(), real4());
      real4* d_gridVelocity = thrust::raw_pointer_cast(gridData.data());
      const int3 n = grid.cellDim;
      IBM<Kernel, Grid> ibm(kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
      ibm.gather(pos, thrust::raw_pointer_cast(particleVelsAndPressure.data()), d_gridVelocity, *qw, numberParticles, st);
      CudaCheckError();
      return particleVelsAndPressure;
    }


  }
}
