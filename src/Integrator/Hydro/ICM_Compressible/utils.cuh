/*Raul P. Pelaez 2022. Utilities for the Compressible Inertial Coupling Method.

 */
#ifndef ICM_COMPRESSIBLE_UTILS_CUH
#define ICM_COMPRESSIBLE_UTILS_CUH
#include "utils/utils.h"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include "utils/container.h"
#include "utils/Grid.cuh"
namespace uammd{
  namespace Hydro{
    namespace icm_compressible{

      template<class T>
      using cached_vector = uammd::uninitialized_cached_vector<T>;
	//thrust::device_vector<T, System::allocator_thrust<T>>;

      struct AoSToSoAReal3{
	template<class VecType>
	__device__ thrust::tuple<real,real,real> operator()(VecType v){
	  return thrust::make_tuple(v.x, v.y, v.z);
	}
      };

      struct SoAToAoSReal3{
	__device__ real3 operator()(thrust::tuple<real, real, real> v){
	  return {thrust::get<0>(v), thrust::get<1>(v), thrust::get<2>(v)};
	}
      };

      struct ToReal3{template<class T> __device__ real3 operator()(T v){return make_real3(v);}};

      struct DataXYZ{
        uninitialized_cached_vector<real> m_x, m_y, m_z;

	DataXYZ():DataXYZ(0){}

	template<class VectorTypeIterator>
	DataXYZ(VectorTypeIterator &input, int size):DataXYZ(size){
	  auto zip = thrust::make_zip_iterator(thrust::make_tuple(m_x.begin(), m_y.begin(), m_z.begin()));
	  thrust::transform(thrust::cuda::par, input, input + size, zip, AoSToSoAReal3());
	}

	DataXYZ(int size){
	  resize(size);
	  fillWithZero();
	}

	void resize(int newSize){
	  m_x.resize(newSize);
	  m_y.resize(newSize);
	  m_z.resize(newSize);
	}

	void fillWithZero() const{
	  thrust::fill(m_x.begin(), m_x.end(), 0);
	  thrust::fill(m_y.begin(), m_y.end(), 0);
	  thrust::fill(m_z.begin(), m_z.end(), 0);
	}

	using Iterator = real*;
	Iterator x()const{return thrust::raw_pointer_cast(m_x.data());}
	Iterator y()const{return thrust::raw_pointer_cast(m_y.data());}
	Iterator z()const{return thrust::raw_pointer_cast(m_z.data());}

        auto xyz() const{
	  auto zip = thrust::make_zip_iterator(thrust::make_tuple(x(), y(), z()));
	  const auto tr = thrust::make_transform_iterator(zip, SoAToAoSReal3());
	  return tr;
	}

	void swap(DataXYZ &another){
	  m_x.swap(another.m_x);
	  m_y.swap(another.m_y);
	  m_z.swap(another.m_z);
	}

	void clear(){
	  m_x.clear();
	  m_y.clear();
	  m_z.clear();
	}

	auto size() const{
	  return this->m_x.size();
	}
      };

      class DataXYZPtr{
	DataXYZ::Iterator m_x,m_y,m_z;
      public:
	DataXYZPtr(const DataXYZ & data):
	  m_x(data.x()), m_y(data.y()),m_z(data.z()){}

	using Iterator = real*;
	__host__ __device__ Iterator x()const{return m_x;}
	__host__ __device__ Iterator y()const{return m_y;}
	__host__ __device__ Iterator z()const{return m_z;}

	__host__ __device__ auto xyz() const{
	  auto zip = thrust::make_zip_iterator(thrust::make_tuple(x(), y(), z()));
	  const auto tr = thrust::make_transform_iterator(zip, SoAToAoSReal3());
	  return tr;
	}

      };

      struct FluidPointers{
	FluidPointers(){}
	template<class RealContainer>
	FluidPointers(const RealContainer &dens, const DataXYZ &vel, const DataXYZ &momentum):
	  density(const_cast<real*>(thrust::raw_pointer_cast(dens.data()))),
	  velocityX(const_cast<real*>(vel.x())), velocityY(const_cast<real*>(vel.y())), velocityZ(const_cast<real*>(vel.z())),
	  momentumX(const_cast<real*>(momentum.x())), momentumY(const_cast<real*>(momentum.y())), momentumZ(const_cast<real*>(momentum.z())){}
	real* density;
	DataXYZ::Iterator velocityX, velocityY, velocityZ;
	DataXYZ::Iterator momentumX, momentumY, momentumZ;
      };

      struct FluidData{
	FluidData(int3 n){
	  resize(n);
	}

	FluidData(): FluidData({0,0,0}){}

	DataXYZ velocity, momentum;
	cached_vector<real> density;

	FluidPointers getPointers() const{
	  return FluidPointers(density, velocity, momentum);
	}

	void resize(int3 n){
	  int newSize = n.x*n.y*n.z;
	  momentum.resize(newSize);
	  velocity.resize(newSize);
	  density.resize(newSize);
	}

	void clear(){
	  momentum.clear();
	  velocity.clear();
	  density.clear();
	}
      };

      struct FluidTimePack{
	FluidPointers timeA, timeB, timeC;
      };

      struct FluidParameters{
	real shearViscosity, bulkViscosity;
	real dt;
      };

      __host__ __device__ int3 getCellFromThreadId(int id, int3 n){
	const int3 cell = make_int3(id%n.x, (id/n.x)%n.y, id/(n.x*n.y));
	return cell;
      }

      //Get the unique linear index of a cell, takes into account ghost cells.
      //That is, the cell =(0,0,0) is not located at index i=0, rather at index 1+(1+(n.y+2))*(n.x+2)
      __host__ __device__ int linearIndex3D(int3 cell, int3 n){
	//Ghost cells are encoded as an extra layer around the grid
	cell += {1,1,1};
	n += {2,2,2};
	return cell.x + (cell.y + cell.z*n.y)*n.x;
      }

      __host__ __device__ int linearIndexGhost3D(int3 cell, int3 n){
	return cell.x + (cell.y + cell.z*n.y)*n.x;
      }

      //Folds the index of a cell in a certain direction back into the domain
      inline __host__  __device__ int pbc_cell_coord(int cell, int ncells){
	if(cell <= -1) cell += ncells;
	else if(cell >= ncells) cell -= ncells;
	return cell;
      }

      //Folds the index of a cell back into the domain
      inline __host__  __device__ int3 pbc_cell(int3 cell, int3 cellDim){
	int3 cellPBC;
	cellPBC.x = pbc_cell_coord(cell.x, cellDim.x);
	cellPBC.y = pbc_cell_coord(cell.y, cellDim.y);
	cellPBC.z = pbc_cell_coord(cell.z, cellDim.z);
	return cellPBC;
      }

      template<class ScalarIterator>
      __device__ real fetchScalar(ScalarIterator scalar, int3 cell, int3 n){
	//cell = pbc_cell(cell, n); //With ghost cells the limits of the grid are never accessed.
	int ic = linearIndex3D(cell, n);
	return scalar[ic];
      }

      enum class subgrid{x,y,z};

      template<subgrid direction>
      __device__ auto getVelocityPointer(FluidPointers fluid){
	if(direction == subgrid::x) return fluid.velocityX;
	if(direction == subgrid::y) return fluid.velocityY;
	if(direction == subgrid::z) return fluid.velocityZ;
      }

      template<subgrid direction>
      __device__ auto getMomentumPointer(FluidPointers fluid){
	if(direction == subgrid::x) return fluid.momentumX;
	if(direction == subgrid::y) return fluid.momentumY;
	if(direction == subgrid::z) return fluid.momentumZ;
      }

    }
  }
}
#endif
