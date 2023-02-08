#ifndef ICM_COMPRESSIBLE_SPREADINTERP_CUH
#define ICM_COMPRESSIBLE_SPREADINTERP_CUH

#include"Integrator/Hydro/ICM_Compressible.cuh"
#include"misc/IBM.cuh"
#include "misc/IBM_kernels.cuh"
#include"utils.cuh"
namespace uammd{
  namespace Hydro{
    namespace icm_compressible{

      //Cell to index transformation for the IBM module, takes into account ghost cells
      struct LinearIndexGhost3D{
	LinearIndexGhost3D(int3 n):n(n){}

	//Returns the index of a certain cell, c, including ghost cells
	inline __device__ int operator()(int3 c) const{
	  return linearIndex3D(c, n);
	}

      private:
	const int3 n;
      };

      namespace staggered{
	struct ShiftTransform{
	  real3 shift;
	  ShiftTransform(real3 shift):shift(shift){}

	  __device__ auto operator()(real4 p){
	    return make_real3(p)+shift;
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
	  const int3 n = grid.cellDim;
	  DataXYZ gridData(n.x*n.y*n.z);
	  gridData.fillWithZero();
	  if(numberParticles > 0){
	    DataXYZ particleDataXYZ(particleData, numberParticles);
	    const real3 h = grid.cellSize;
	    IBM<Kernel, Grid> ibm(kernel, grid);
	    auto posX = make_shift_iterator(positions, {-real(0.5)*h.x, 0, 0});
	    ibm.spread(posX, particleDataXYZ.x(), gridData.x(), numberParticles);
	    auto posY = make_shift_iterator(positions, {0, -real(0.5)*h.y, 0});
	    ibm.spread(posY, particleDataXYZ.y(), gridData.y(), numberParticles);
	    auto posZ = make_shift_iterator(positions, {0, 0, -real(0.5)*h.z});
	    ibm.spread(posZ, particleDataXYZ.z(), gridData.z(), numberParticles);
	  }
	  return gridData;
	}

	template<class PositionIterator, class Kernel>
	auto interpolateFluidVelocities(const DataXYZ &gridData, const PositionIterator &positions,
					std::shared_ptr<Kernel> kernel,
					int numberParticles, Grid grid){
	  DataXYZ particleDataXYZ(numberParticles);
	  particleDataXYZ.fillWithZero();
	  if(numberParticles > 0){
	    const int3 n = grid.cellDim;
	    const real3 h = grid.cellSize;
	    IBM<Kernel, Grid, LinearIndexGhost3D> ibm(kernel, grid, LinearIndexGhost3D(n));
	    auto posX = make_shift_iterator(positions, {-real(0.5)*h.x, 0, 0});
	    ibm.gather(posX, particleDataXYZ.x(), gridData.x(), numberParticles);
	    auto posY = make_shift_iterator(positions, {0, -real(0.5)*h.y, 0});
	    ibm.gather(posY, particleDataXYZ.y(), gridData.y(), numberParticles);
	    auto posZ = make_shift_iterator(positions, {0, 0, -real(0.5)*h.z});
	    ibm.gather(posZ, particleDataXYZ.z(), gridData.z(), numberParticles);
	  }
	  return particleDataXYZ;
	}

      }
    }
  }
}
#endif
