#ifndef ICM_COMPRESSIBLE_SPREADINTERP_CUH
#define ICM_COMPRESSIBLE_SPREADINTERP_CUH

#include"Integrator/Hydro/ICM_Compressible.cuh"
#include"misc/IBM.cuh"
#include "misc/IBM_kernels.cuh"
#include"utils.cuh"
namespace uammd{
  namespace Hydro{
    namespace icm_compressible{
      namespace staggered{
	struct ShiftTransform{
	  real3 shift;
	  ShiftTransform(real3 shift):shift(shift){}

	  __device__ auto operator()(real4 p){
	    return make_real3(p)-shift;
	  }

	};

	template<class PositionIterator>
	auto make_shift_iterator(const PositionIterator &positions, real3 shift){
	  return thrust::make_transform_iterator(positions, ShiftTransform(shift));
	}

	template<class ParticleIterator, class PositionIterator, class Kernel>
	auto spreadParticleForces(const ParticleIterator &particleData, const PositionIterator &positions,
				  std::shared_ptr<Kernel> kernel,
				  int numberParticles, Grid grid){
	  DataXYZ particleDataXYZ(particleData, numberParticles);
	  DataXYZ gridData(grid.getNumberCells());
	  gridData.fillWithZero();
	  const real3 h = grid.cellSize;
	  IBM<Kernel> ibm(kernel, grid);
	  auto posX = make_shift_iterator(positions, {real(0.5)*h.x, 0, 0});
	  ibm.spread(posX, particleDataXYZ.x(), gridData.x(), numberParticles);
	  auto posY = make_shift_iterator(positions, {0, real(0.5)*h.y, 0});
	  ibm.spread(posY, particleDataXYZ.y(), gridData.y(), numberParticles);
	  auto posZ = make_shift_iterator(positions, {0, 0, real(0.5)*h.z});
	  ibm.spread(posZ, particleDataXYZ.z(), gridData.z(), numberParticles);
	  return gridData;
}

	template<class PositionIterator, class Kernel>
	auto interpolateFluidVelocities(const DataXYZ &gridData, const PositionIterator &positions,
					std::shared_ptr<Kernel> kernel,
					int numberParticles, Grid grid){
	  DataXYZ particleDataXYZ(numberParticles);
	  particleDataXYZ.fillWithZero();
	  const real3 h = grid.cellSize;
	  IBM<Kernel> ibm(kernel, grid);
	  auto posX = make_shift_iterator(positions, {real(0.5)*h.x, 0, 0});
	  ibm.gather(posX, particleDataXYZ.x(), gridData.x(), numberParticles);
	  auto posY = make_shift_iterator(positions, {0, real(0.5)*h.y, 0});
	  ibm.gather(posY, particleDataXYZ.y(), gridData.y(), numberParticles);
	  auto posZ = make_shift_iterator(positions, {0, 0, real(0.5)*h.z});
	  ibm.gather(posZ, particleDataXYZ.z(), gridData.z(), numberParticles);
	  return particleDataXYZ;
	}

      }
      namespace regular{
	template<class ParticleIterator, class PositionIterator, class Kernel>
	auto spreadParticleForces(ParticleIterator &particleData, PositionIterator &positions,
				  std::shared_ptr<Kernel> kernel,
				  int numberParticles, Grid grid){
	  DataXYZ particleDataXYZ(particleData, numberParticles);
	  DataXYZ gridData(grid.getNumberCells());
	  gridData.fillWithZero();
	  const real3 h = grid.cellSize;
	  IBM<Kernel> ibm(kernel, grid);
	  ibm.spread(positions, particleDataXYZ.x(), gridData.x(), numberParticles);
	  ibm.spread(positions, particleDataXYZ.y(), gridData.y(), numberParticles);
	  ibm.spread(positions, particleDataXYZ.z(), gridData.z(), numberParticles);
	  return gridData;
	}

	template<class ParticleIterator, class PositionIterator, class Kernel>
	auto interpolateFluidVelocities(ParticleIterator &particleData, PositionIterator &positions,
					std::shared_ptr<Kernel> kernel,
					int numberParticles, Grid grid){
	  DataXYZ particleDataXYZ(particleData, numberParticles);
	  DataXYZ gridData(grid.getNumberCells());
	  gridData.fillWithZero();
	  const real3 h = grid.cellSize;
	  IBM<Kernel> ibm(kernel, grid);
	  ibm.gather(positions, particleDataXYZ.x(), gridData.x(), numberParticles);
	  ibm.gather(positions, particleDataXYZ.y(), gridData.y(), numberParticles);
	  ibm.gather(positions, particleDataXYZ.z(), gridData.z(), numberParticles);
	  return gridData;
	}
      }
    }
  }
}
#endif
