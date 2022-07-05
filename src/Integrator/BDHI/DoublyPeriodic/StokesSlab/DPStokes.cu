/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver
 */
#include"Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"
#include "misc/IBM.cuh"
#include"utils.cuh"
#include"BVPStokes.cuh"
#include "utils/debugTools.h"
#include"utils/NVTXTools.h"
namespace uammd{
  namespace DPStokesSlab_ns{

    //Computes the hydrodynamic displacements (velocities) coming from the forces
    // acting on a group of positions.
    cached_vector<real3> DPStokes::Mdot(real4* pos, real4* forces, int numberParticles, cudaStream_t st){
      auto M = Mdot(pos, forces, nullptr, numberParticles, st);
      return M.first;
    }

    //Computes the linear and angular hydrodynamic displacements (velocities) coming from
    // the forces and torques acting on a group of positions
    //If the torques pointer is null, the function will only compute and return the translational part
    // of the mobility
    std::pair<cached_vector<real3>, cached_vector<real3>>
    DPStokes::Mdot(real4* pos, real4* forces, real4* torques, int numberParticles, cudaStream_t st){
      cudaDeviceSynchronize();
      System::log<System::DEBUG2>("[DPStokes] Computing displacements");
      auto gridData = spreadForces(pos, forces, numberParticles, st);
      auto gridDataCheb = fct->forwardTransform(gridData, st);
      if(torques){//Torques are added in Cheb space
	addSpreadTorquesFourier(pos, torques, numberParticles, gridDataCheb, st);
      }
      solveBVPVelocity(gridDataCheb, st);
      if(mode != WallMode::none){
      	correction->correctSolution(gridDataCheb, gridDataCheb, st);
      }
      cached_vector<real3> particleAngularVelocities;
      if(torques){
	//Ang. velocities are interpolated from the curl of the velocity, which is
	// computed in Cheb space.
	auto gridAngVelsCheb = computeGridAngularVelocityCheb(gridDataCheb, st);
	auto gridAngVels = fct->inverseTransform(gridAngVelsCheb, st);
	particleAngularVelocities = interpolateAngularVelocity(gridAngVels, pos, numberParticles, st);
      }
      gridData = fct->inverseTransform(gridDataCheb, st);
      auto particleVelocities = interpolateVelocity(gridData, pos, numberParticles, st);
      CudaCheckError();
      return {particleVelocities, particleAngularVelocities};
    }

    cached_vector<real4> DPStokes::spreadForces(real4* pos, real4* forces, int numberParticles, cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] Spreading forces");
      const int3 n = grid.cellDim;
      cached_vector<real4> gridForce(2*(n.x/2+1)*n.y*(2*n.z-2));
      thrust::fill(thrust::cuda::par.on(st), gridForce.begin(), gridForce.end(), real4());
      auto d_gridForce = thrust::raw_pointer_cast(gridForce.data());
      IBM<Kernel, Grid> ibm(kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
      ibm.spread(pos, forces, d_gridForce, numberParticles, st);
      CudaCheckError();
      return gridForce;
    }

    namespace detail{
      //Computes the coefficients of the derivative of "f" in Cheb space.
      //Stores the result in Df, which can be aliased to f
      template<class Iter>
      __device__ void chebyshevDerivate(const Iter &f, Iter &Df, int nz, real halfH){
	using T = typename std::iterator_traits<Iter>::value_type;
	T fip1 = T();
	T Dpnp2 = T();
	T Dpnp1 = T();
	for(int i = nz-1; i>=0; i--){
	  T Dpni = T();
	  if(i<=nz-2) Dpni = Dpnp2 + real(2.0)*(i+1)*fip1/halfH;
	  if(i==0) Dpni *= real(0.5);
	  fip1 = f[i];
	  Df[i] = Dpni;
	  Dpnp2 = Dpnp1;
	  if(i<=nz-2){
	    Dpnp1 = Dpni;
	  }
	}
      }

      //Compute the curl of the torque in chebyshev space (T is torque in cheb)
      // 0.5\nabla \times T = 0.5 (i*k_x i*k_y \partial_z)\times (T_x T_y T_z) =
      // = 0.5( i*k_y*T_z - \partial_z(T_y), \partial_z(T_x) - i*k_x*T_z, i*k_x*T_y - i*k_y*T_x)
      //Add to the output vector, which might contain the contribution from the forces
      //The input torque vector is overwritten
      __global__ void addTorqueCurlCheb(cufftComplex4 *gridTorqueCheb, cufftComplex4* gridTorqueCurlCheb, real3 L, int3 n){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	const int2 ik = make_int2(id%(n.x/2+1), id/(n.x/2+1));
	if(id >= n.y*(n.x/2+1)){
	  return;
	}
	const auto kn = computeWaveNumber(id, n.x, n.y);
	const auto k = computeWaveVector(kn, make_real2(L));
	auto torque = make_third_index_iterator(gridTorqueCheb, ik.x, ik.y, Index3D(n.x/2+1, n.y, 2*n.z-2));
	auto curl = make_third_index_iterator(gridTorqueCurlCheb, ik.x, ik.y, Index3D(n.x/2+1, n.y, 2*n.z-2));
	//First sum the terms that do not depend on the derivative in Z
	const real half = real(0.5);
	const bool isUnpairedX = ik.x == (n.x - ik.x);
	const bool isUnpairedY = ik.y == (n.y - ik.y);
	real Dx = isUnpairedX?0:k.x;
	real Dy = isUnpairedY?0:k.y;
	fori(0, n.z){
	  const auto T = torque[i];
	   curl[i].x += {-half*Dy*T.z.y, half*Dy*T.z.x};
	   curl[i].y += {half*Dx*T.z.y, -half*Dx*T.z.x};
	   curl[i].z += {half*(-Dx*T.y.y + Dy*T.x.y),
	                 half*(Dx*T.y.x - Dy*T.x.x)};
	}
	//Overwrite input torque with Z derivatives
	chebyshevDerivate(torque, torque, n.z, real(0.5)*L.z);
	//Sum the rest of the terms
	fori(0, n.z){
	  const auto DzT = torque[i];
	  curl[i].x += -half*DzT.y;
	  curl[i].y += half*DzT.x;
	}
      }

    }

    //Spread the curl of the torques to the grid and add it to the fluid forcing in Cheb space
    void DPStokes::addSpreadTorquesFourier(real4* pos, real4* torques, int numberParticles,
					   cached_vector<cufftComplex4> &gridForceCheb, cudaStream_t st){
      if(torques == nullptr) return;
      System::log<System::DEBUG2>("[DPStokes] Spreading torques");
      const int3 n = grid.cellDim;
      cached_vector<real4> gridTorque(2*(n.x/2+1)*n.y*(2*n.z-2));
      thrust::fill(thrust::cuda::par.on(st), gridTorque.begin(), gridTorque.end(), real4());
      auto d_gridTorque = thrust::raw_pointer_cast(gridTorque.data());
      IBM<KernelTorque, Grid> ibm(kernelTorque, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
      ibm.spread(pos, torques, d_gridTorque, numberParticles, st);
      auto gridTorqueCheb = fct->forwardTransform(gridTorque, st);
      auto d_gridForceCheb = thrust::raw_pointer_cast(gridForceCheb.data());
      auto d_gridTorqueCheb = thrust::raw_pointer_cast(gridTorqueCheb.data());
      const int blockSize = 128;
      const int numberSystems = n.y*(n.x/2+1);
      const int numberBlocks = numberSystems/blockSize+1;
      detail::addTorqueCurlCheb<<<numberBlocks, blockSize, 0, st>>>(d_gridTorqueCheb, d_gridForceCheb,
								    make_real3(Lx, Ly, H), n);
      CudaCheckError();
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
      const int blockSize = 64;
      const int numberBlocks = numberSystems/blockSize+1;
      solveBVPVelocityD<<<numberBlocks, blockSize, 0,st>>>(bvpSolver->getGPUSolver(), n.x, n.y, n.z,
							   make_real2(Lx, Ly), H*0.5,
							   d_gridForceFourier,
							   d_gridVelocityFourier,
							   tmp,
							   thrust::raw_pointer_cast(tmp_storage.data()),
							   d_precomputedPressure, d_precomputedVelocity,
							   viscosity, mode);
      CudaCheckError();
      System::log<System::DEBUG2>("[DPStokes] BVP solve done");
    }
    namespace detail{
      struct ToReal3{
	template<class T>
	__host__ __device__ uammd::real3 operator()(T i){
	  auto pr3 = uammd::make_real3(i);
	  return pr3;
	}
      };

    }
    cached_vector<real3> DPStokes::interpolateVelocity(cached_vector<real4> &gridData, real4* pos,
						       int numberParticles, cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] Interpolating forces and energies");
      cached_vector<real3> particleVels(numberParticles);
      thrust::fill(thrust::cuda::par.on(st), particleVels.begin(), particleVels.end(), real3());
      real4* d_gridVelocity = thrust::raw_pointer_cast(gridData.data());
      auto dgrid3 = thrust::make_transform_iterator(d_gridVelocity, detail::ToReal3());
      const int3 n = grid.cellDim;
      IBM<Kernel, Grid> ibm(kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
      ibm.gather(pos, thrust::raw_pointer_cast(particleVels.data()),
		 dgrid3,
		 *qw, IBM_ns::DefaultWeightCompute(),
		 numberParticles, st);
      CudaCheckError();
      return particleVels;
    }

    namespace detail{
      //Compute the curl of the velocity, V, in chebyshev space. This is equal to the angular velocity
      // 0.5\nabla \times V = 0.5 (i*k_x i*k_y \partial_z)\times (V_x V_y V_z) =
      // = 0.5( i*k_y*V_z - \partial_z(V_y), \partial_z(V_x) - i*k_x*V_z, i*k_x*V_y - i*k_y*V_x)
      //Overwrite the output vector with the angular velocities in Cheb space
      //The input velocity vector is overwritten
      __global__ void computeVelocityCurlCheb(cufftComplex4 *gridVelsCheb, cufftComplex4* gridAngVelCheb,
					      real3 L, int3 n){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	const int2 ik = make_int2(id%(n.x/2+1), id/(n.x/2+1));
	if(id >= n.y*(n.x/2+1)){
	  return;
	}
	auto kn = computeWaveNumber(id, n.x, n.y);
	auto k = computeWaveVector(kn, make_real2(L));
	auto vels = make_third_index_iterator(gridVelsCheb, ik.x, ik.y, Index3D(n.x/2+1, n.y, 2*n.z-2));
	auto curl = make_third_index_iterator(gridAngVelCheb, ik.x, ik.y, Index3D(n.x/2+1, n.y, 2*n.z-2));
	//First sum the terms that do not depend on the derivative in Z
	const real half = real(0.5);
	const bool isUnpairedX = ik.x == (n.x - ik.x);
	const bool isUnpairedY = ik.y == (n.y - ik.y);
	real Dx = isUnpairedX?0:k.x;
	real Dy = isUnpairedY?0:k.y;
	fori(0, n.z){
	  const auto T = vels[i];
	  curl[i].x = {-half*Dy*T.z.y, half*Dy*T.z.x};
	  curl[i].y = {half*Dx*T.z.y, -half*Dx*T.z.x};
	  curl[i].z = {half*(-Dx*T.y.y + Dy*T.x.y),
	               half*(Dx*T.y.x - Dy*T.x.x)};
	}
	//Overwrite input torque with Z derivatives
	chebyshevDerivate(vels, vels, n.z, real(0.5)*L.z);
	//Sum the rest of the terms
	fori(0, n.z){
	  const auto DzT = vels[i];
	  curl[i].x += -half*DzT.y;
	  curl[i].y += half*DzT.x;
	}
      }
    }

    cached_vector<cufftComplex4> DPStokes::computeGridAngularVelocityCheb(cached_vector<cufftComplex4> &gridVelsCheb,
									  cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] Computing angular velocities as curl of velocity");
      const int3 n = grid.cellDim;
      const int blockSize = 128;
      const int numberSystems = n.y*(n.x/2+1);
      const int numberBlocks = numberSystems/blockSize+1;
      //The kernel overwrites the input vector, thus the copy
      cached_vector<cufftComplex4> gridVelsChebCopy(gridVelsCheb.size());
      thrust::copy(gridVelsCheb.begin(), gridVelsCheb.end(), gridVelsChebCopy.begin());
      auto d_gridVelsCheb = thrust::raw_pointer_cast(gridVelsChebCopy.data());
      cached_vector<cufftComplex4> gridAngVelsCheb(gridVelsChebCopy.size());
      auto d_gridAngVelsCheb = thrust::raw_pointer_cast(gridAngVelsCheb.data());
      detail::computeVelocityCurlCheb<<<numberBlocks, blockSize, 0, st>>>(d_gridVelsCheb, d_gridAngVelsCheb,
									  make_real3(Lx, Ly, H), n);
      CudaCheckError();
      return gridAngVelsCheb;
    }

    cached_vector<real3> DPStokes::interpolateAngularVelocity(cached_vector<real4> &gridAngVels,
							      real4* pos,
							      int numberParticles, cudaStream_t st){
      System::log<System::DEBUG2>("[DPStokes] Interpolating angular velocities");
      const int3 n = grid.cellDim;
      cached_vector<real3> particleAngVels(numberParticles);
      thrust::fill(thrust::cuda::par.on(st), particleAngVels.begin(), particleAngVels.end(), real3());
      real4* d_gridAngVelocity = thrust::raw_pointer_cast(gridAngVels.data());
      auto dgrid3 = thrust::make_transform_iterator(d_gridAngVelocity, detail::ToReal3());
      IBM<KernelTorque, Grid> ibm(kernelTorque, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
      ibm.gather(pos, thrust::raw_pointer_cast(particleAngVels.data()),
		 dgrid3,
		 *qw, IBM_ns::DefaultWeightCompute(),
		 numberParticles, st);
      CudaCheckError();
      return particleAngVels;
    }


  }
}
