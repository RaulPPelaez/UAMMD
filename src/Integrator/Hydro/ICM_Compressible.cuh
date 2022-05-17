#ifndef UAMMD_ICM_COMPRESSIBLE_CUH
#define UAMMD_ICM_COMPRESSIBLE_CUH
#include"uammd.cuh"
#include"Integrator/Integrator.cuh"
#include"utils/container.h"
#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "This code requires the CUDA flag --extended-lambda to be enabled"
#endif
namespace uammd{
  namespace Hydro{

    namespace icm_compressible{
      //Equation of state      
      struct DensityToPressure{
	real isothermalSpeedOfSound = 1.0;
	__device__ real operator()(real density){
	  return isothermalSpeedOfSound*isothermalSpeedOfSound*density;
	}
      };
      
      template<class T>
      using cached_vector = uammd::uninitialized_cached_vector<T>;

      struct AoSToSoAReal3{
	template<class VecType>
	__device__ auto operator()(VecType v){
	  return thrust::make_tuple(v.x, v.y, v.z);
	}
      };

      struct SoAToAoSReal3{       
	__device__ auto operator()(thrust::tuple<real, real, real> v){
	  return make_real3(thrust::get<0>(v), thrust::get<1>(v), thrust::get<2>(v));
	}
      };

      struct ToReal3{template<class T> __device__ real3 operator()(T v){return make_real3(v);}};
      
      struct DataXYZ{
        uninitialized_cached_vector<real> m_x, m_y, m_z;
	
	DataXYZ():DataXYZ(0){}

	template<class VectorTypeIterator>
	DataXYZ(VectorTypeIterator &input, int size):DataXYZ(size){
	  auto zip = thrust::make_zip_iterator(thrust::make_tuple(m_x.begin(), m_y.begin(), m_z.begin()));
	  thrust::transform(input, input + size, zip, AoSToSoAReal3());
	}
		
	DataXYZ(int size){
	  resize(size);
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

	auto size(){
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
	FluidPointers(RealContainer &dens, DataXYZ &vel):
	  density(thrust::raw_pointer_cast(dens.data())),
	  velocityX(vel.x()), velocityY(vel.y()), velocityZ(vel.z()){}
	real* density;
	DataXYZ::Iterator velocityX, velocityY, velocityZ;
      };

      struct FluidData{
	FluidData(Grid grid):
	  velocity(grid.getNumberCells()),
	  density(grid.getNumberCells()){}
	DataXYZ velocity;
	cached_vector<real> density;

	FluidPointers getPointers(){
	  return FluidPointers(density, velocity);
	}

	void clear(){
	  velocity.clear();
	  density.clear();
	}
      };

    }
    
    class ICM_Compressible: public Integrator{
      template<class T>
      using cached_vector = icm_compressible::cached_vector<T>;
      using DataXYZ = icm_compressible::DataXYZ;
      using FluidPointers = icm_compressible::FluidPointers;
      using DensityToPressure = icm_compressible::DensityToPressure;
    public:
      struct Parameters{
	real shearViscosity = -1;
	real bulkViscosity = -1;
	real speedOfSound = -1;
	real temperature = 0;
	real hydrodynamicRadius = -1;
	real dt = -1;
	Box box;
	uint seed = 0;
      };
      
      ICM_Compressible(std::shared_ptr<ParticleData> pd, Parameters par):
	Integrator(pd, "ICM::Compressible"){
	densityToPressure = std::make_shared<DensityToPressure>();
	dt = par.dt;
	shearViscosity = par.shearViscosity;
	bulkViscosity = par.bulkViscosity;
	temperature = par.temperature;
	seed = (par.seed==0)?sys->rng().next32():par.seed;
	int3 ncells = make_int3(par.box.boxSize/(0.91*par.hydrodynamicRadius));
	grid = Grid(par.box, ncells);
	densityToPressure->isothermalSpeedOfSound = par.speedOfSound;
	currentFluidDensity.resize(grid.getNumberCells());
	thrust::fill(currentFluidDensity.begin(), currentFluidDensity.end(), 1.0);
	currentFluidVelocity.resize(grid.getNumberCells());
	currentFluidVelocity.fillWithZero();

	System::log<System::MESSAGE>("[ICM_Compressible] dt: %g", dt);
	System::log<System::MESSAGE>("[ICM_Compressible] shear viscosity: %g", shearViscosity);
	System::log<System::MESSAGE>("[ICM_Compressible] bulk viscosity: %g", bulkViscosity);
	System::log<System::MESSAGE>("[ICM_Compressible] isothermal speed of sound: %g", par.speedOfSound);
	System::log<System::MESSAGE>("[ICM_Compressible] temperature: %g", temperature);
	
	System::log<System::MESSAGE>("[ICM_Compressible] Box size: %g %g %g", par.box.boxSize.x, par.box.boxSize.y, par.box.boxSize.z);
	System::log<System::MESSAGE>("[ICM_Compressible] Fluid cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
	System::log<System::MESSAGE>("[ICM_Compressible] seed: %ux", seed);
	
      }

      void forwardTime() override;

    private:
      Grid grid;
      DataXYZ currentFluidVelocity;
      cached_vector<real> currentFluidDensity;

      auto storeCurrentPositions();
      void forwardPositionsToHalfStep();
      auto computeCurrentFluidForcing();
       void updateParticleForces();
       auto spreadCurrentParticleForcesToFluid();
      
      auto interpolateFluidVelocityToParticles(const DataXYZ &fluidVelocities);
      void forwardFluidDensityAndVelocityToNextStep(const DataXYZ &fluidForcingAtHalfStep);
      auto computeStochasticTensor();
      void updateFluidWithRungeKutta3(const DataXYZ &fluidForcingAtHalfStep,
				      cached_vector<real2> &fluidStochasticTensor);
      template<int subStep>
      auto callRungeKuttaSubStep(const DataXYZ &fluidForcingAtHalfStep,
				 cached_vector<real2> &fluidStochasticTensor,
				 FluidPointers fluidAtSubTime = FluidPointers());
      void addFluidExternalForcing(DataXYZ &fluidForcingAtHalfStep){}
      void forwardPositionsToNextStep(cached_vector<real4> currentPositions, DataXYZ &fluidVelocitiesAtN);

      int steps = 0;
      real dt;
      real shearViscosity, bulkViscosity;
      std::shared_ptr<DensityToPressure> densityToPressure;
      real temperature;
      uint seed = 1234;
    };

  }  
}
#include"ICM_Compressible.cu"
#endif




