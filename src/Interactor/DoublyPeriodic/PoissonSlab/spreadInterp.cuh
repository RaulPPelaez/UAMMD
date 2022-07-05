/*Raul P. Pelaez 2020. Spreading and interpolation for the Doubly Periodic Poisson solver. Slab geometry
*/

#ifndef DPPOISSON_SPREADINTERP_CUH
#define DPPOISSON_SPREADINTERP_CUH

#include "ParticleData/ParticleData.cuh"
#include "global/defines.h"
#include "misc/ChevyshevUtils.cuh"
#include "misc/IBM.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftDebug.h"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include <iostream>
#include <thrust/iterator/discard_iterator.h>

namespace uammd{
  namespace DPPoissonSlab_ns{

    struct Gaussian{
      int3 support;
      Gaussian(real tolerance, real width, real h, real H, real He, real nz, int supportxy):
	nz(nz){
	this-> prefactor = cbrt(pow(2*M_PI*width*width, -1.5));
	this-> tau = -1.0/(2.0*width*width);
	rmax = supportxy*h*0.5;//5.0*h;//sqrt(log(tolerance/prefactor)/tau);
	support.x = supportxy>0?(supportxy):std::max(3, int(2*rmax/h + 0.5)+1);
	support.y = support.x;
	this->Htot = H + 4*He;
	int czmax = int((nz-1)*(acos(2.0*(0.5*H + He)/Htot)/real(M_PI)));
	support.z = 2*czmax;
	if(support.z%2==0){
	  support.z--;
	}
      }

      inline __host__  __device__ int3 getMaxSupport() const{
	return make_int3(support.x, support.y, support.z);
      }

      inline __host__  __device__ int3 getSupport(real3 pos, int3 cell) const{
	real ch = real(0.5)*Htot*cospi((real(cell.z))/(nz-1));
	int czt = int((nz)*(acos(real(2.0)*(ch+rmax)/Htot)/real(M_PI)));
	int czb = int((nz)*(acos(real(2.0)*(ch-rmax)/Htot)/real(M_PI)));
	int sz = 2*thrust::max(cell.z - czt, czb - cell.z)+1;
	return make_int3(support.x, support.y, sz);
      }

      inline __host__  __device__ real phi(real r, real3 pos) const{
	return (abs(r)>=rmax)?0:(prefactor*exp(tau*r*r));
      }

      inline __host__  __device__ real delta(real3 rvec, real3 h) const{
	const real r2 = dot(rvec, rvec);
	return (abs(rvec.z)>=rmax)?0:(prefactor*prefactor*prefactor*exp(tau*r2));
      }

    private:
      real prefactor;
      real tau;
      real rmax;
      int nz;
      real Htot;
    };

    template<class PosIterator, class ChargeIterator>
    struct ChargeGroup{
      const PosIterator pos;
      const ChargeIterator charge;
      const int numberParticles;
      ChargeGroup(const PosIterator p, const ChargeIterator c, int N):
	pos(p), charge(c), numberParticles(N){}
    };

    template<class PosIterator, class ChargeIterator>
    ChargeGroup<PosIterator, ChargeIterator> make_charge_group(const PosIterator &pos,
							       const ChargeIterator &charge,
							       int numberParticles){
      return ChargeGroup<PosIterator, ChargeIterator>(pos, charge, numberParticles);
    }

    struct ImagePositionTransform{
      real wallZ;
      ImagePositionTransform(real wallZ):wallZ(wallZ){}
      __device__ __host__ real4 operator()(real4 ipos){
	auto pos = ipos;
	pos.z = real(2.0)*wallZ - pos.z;
	return pos;
      }
    };

    struct ImageChargeTransform{
      real* charges;
      real permivityRatio;
      ImageChargeTransform(real* c, real permivityRatio):
	charges(c), permivityRatio(permivityRatio){}

      __device__ __host__ real operator()(int i){
	real q = -charges[i];
	real ep = permivityRatio;
	return q*(ep-1)/(ep+1);
      }

    };

    namespace detail{
      struct IsPositionCloseToWall{
	real halfH, threshold;
	real4* pos;

	IsPositionCloseToWall(real H, real threshold, real4 *pos):
	  halfH(0.5*H), threshold(threshold), pos(pos){
	}

	__device__ bool operator()(int i){
	  real z = pos[i].z;
	  return  abs(halfH - abs(z)) <= threshold;
	}
      };

      template <class PositionIterator>
      struct HasImageChargeWithWall{
        const real wallZ, threshold;
	const PositionIterator pos;
        HasImageChargeWithWall(real wallZ, real threshold, PositionIterator pos)
            : wallZ(wallZ), threshold(threshold), pos(pos) {}

        __device__ bool operator()(int i){
	  real z = pos[i].z;
	  return abs(wallZ - z) <= threshold;
	}
      };
    }
    class SeparatedCharges{

    public:
      cached_vector<int> nearWalls;
      cached_vector<int> farFromWalls;
      cached_vector<int> imagesTop;
      cached_vector<int> imagesBottom;

      template<class Container>
      void separate(const Container &positions, real H,
		    real thresholdDistance, cudaStream_t st){
	detail::IsPositionCloseToWall predicate(H, thresholdDistance, positions.begin());
	auto cit = thrust::make_counting_iterator<int>(0);
	int numberParticles = positions.size();
	nearWalls.resize(numberParticles);
	farFromWalls.resize(numberParticles);
	System::log<System::DEBUG2>("Searching particles close and far to walls.");
	auto lastParticles = thrust::partition_copy(thrust::cuda::par.on(st),
						    cit, cit + numberParticles,
						    nearWalls.begin(),
						    farFromWalls.begin(),
						    predicate);
	int nnear = thrust::distance(nearWalls.begin(), lastParticles.first);
	nearWalls.resize(nnear);
	int nfar = thrust::distance(farFromWalls.begin(), lastParticles.second);
	farFromWalls.resize(nfar);
	identifyImageCharges(positions, H, thresholdDistance, nnear, st);
	CudaCheckError();
      }

    private:
      template<class Container>
      void identifyImageCharges(const Container &positions, real H,
				real thresholdDistance, int nnear, cudaStream_t st){
	auto positionsNearWalls = positions.begin();
	auto cit = nearWalls.begin();
	imagesTop.resize(nnear);
	System::log<System::DEBUG2>("Searching for top images");
	{
	  detail::HasImageChargeWithWall<decltype(positionsNearWalls)> hasImageCharge(0.5*H, thresholdDistance, positionsNearWalls);
	  auto lastParticle = thrust::partition_copy(thrust::cuda::par.on(st),
						     cit, cit + nnear,
						     imagesTop.begin(), thrust::make_discard_iterator(),
						     hasImageCharge);
	  imagesTop.resize(thrust::distance(imagesTop.begin(), lastParticle.first));
	}
	imagesBottom.resize(nnear);
	System::log<System::DEBUG2>("Searching for bottom images");
	{
	  detail::HasImageChargeWithWall<decltype(positionsNearWalls)> hasImageCharge(-0.5*H, thresholdDistance, positionsNearWalls);
	  auto lastParticle = thrust::partition_copy(thrust::cuda::par.on(st),
						     cit, cit + nnear,
						     imagesBottom.begin(), thrust::make_discard_iterator(),
						     hasImageCharge);
	  imagesBottom.resize(thrust::distance(imagesBottom.begin(), lastParticle.first));
	}
      }

    };

    struct toReal4{
      inline __device__ real4 operator()(real3 a) const{
	return make_real4(a);
      }
    };

    struct FieldPotential2ForceEnergy{
      real4* force;
      real* energy;
      real* charge;
      int i;

      FieldPotential2ForceEnergy(real4* f, real* e, real *q):force(f), energy(e), charge(q), i(-1){}

      __device__ FieldPotential2ForceEnergy operator()(int ai){
	this->i = ai;
	return *this;
      }

      __device__ void operator += (real4 fande) const{
	force[i] += -charge[i]*make_real4(fande.x, fande.y, fande.z, 0);
	energy[i] += -charge[i]*fande.w;
      }
    };

    class SpreadInterpolateCharges{
      using Grid = chebyshev::doublyperiodic::Grid;
      using QuadratureWeights = chebyshev::doublyperiodic::QuadratureWeights;
      using Kernel = DPPoissonSlab_ns::Gaussian;

      real H;
      Grid grid;
      std::shared_ptr<Kernel> kernel;
      std::shared_ptr<QuadratureWeights> qw;

      std::shared_ptr<System> sys;
      std::shared_ptr<ParticleData> pd;

    public:

      struct Parameters{
	Grid grid;
	real H, He;
	real gaussianWidth;
	real tolerance = 1e-4;
	int support = -1;
	int maximumSupport = -1;
      };

      SpreadInterpolateCharges(std::shared_ptr<System> sys, std::shared_ptr<ParticleData> pd, Parameters par):
	sys(sys), pd(pd), grid(par.grid), H(par.H){
	sys->log<System::DEBUG>("[SpreadInterpolate] Initialized");
	initializeKernel(par.tolerance, par.gaussianWidth, par.He, par.maximumSupport, par.support);
	initializeQuadratureWeights();
      }

      cached_vector<real> spreadChargesNearWalls(SeparatedCharges &sep, cudaStream_t st){
	sys->log<System::DEBUG>("Spreading %d particles near walls", sep.nearWalls.size());
	auto n = grid.cellDim;
	cached_vector<real> gridCharges((2*(n.x/2+1))*n.y*(2*n.z-2));
	thrust::fill(thrust::cuda::par.on(st), gridCharges.begin(), gridCharges.end(), real());
	int* indices = thrust::raw_pointer_cast(sep.nearWalls.data());
	spreadFromIndices(indices, thrust::raw_pointer_cast(gridCharges.data()), sep.nearWalls.size(), st);
	return gridCharges;
      }

      void spreadChargesFarFromWallAdd(SeparatedCharges &sep, cached_vector<real> &gridCharges, cudaStream_t st){
	sys->log<System::DEBUG>("Spreading %d particles far from walls", sep.farFromWalls.size());
	real* d_gridCharges = thrust::raw_pointer_cast(gridCharges.data());
	int* indices = thrust::raw_pointer_cast(sep.farFromWalls.data());
	spreadFromIndices(indices, d_gridCharges, sep.farFromWalls.size(), st);
      }

      void spreadImageChargesAdd(SeparatedCharges &sep, cached_vector<real> &gridCharges, Permitivity permitivity, cudaStream_t st){
	sys->log<System::DEBUG>("Spreading %d (top) and %d (bottom) particle images",
				sep.imagesTop.size(), sep.imagesBottom.size());
	spreadBottomWallImages(sep, gridCharges, permitivity, st);
	spreadTopWallImages(sep, gridCharges, permitivity, st);
      }

      template<class Real4Container>
      void interpolateFieldsToParticles(Real4Container &gridFieldPotential, cudaStream_t st){
	sys->log<System::DEBUG2>("[DPPoissonSlab] Interpolating forces and energies");
	int numberParticles = pd->getNumParticles();
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto charge = pd->getCharge(access::location::gpu, access::mode::read);
	auto forces = pd->getForce(access::location::gpu, access::mode::readwrite);
	auto energies = pd->getEnergy(access::location::gpu, access::mode::readwrite);
	real4* d_gridForcesEnergies = (real4*)thrust::raw_pointer_cast(gridFieldPotential.data());
	auto Ep2fe = DPPoissonSlab_ns::FieldPotential2ForceEnergy(forces.begin(), energies.begin(), charge.begin());
	auto f_tr = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), Ep2fe);
	int3 n = grid.cellDim;
	IBM<Kernel, Grid> ibm(kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
	IBM_ns::DefaultWeightCompute wc; //If non-default QuadratureWeights are used, a weight compute must also be passed.
	ibm.gather(pos.begin(), f_tr, d_gridForcesEnergies, *qw, wc, numberParticles, st);
	CudaCheckError();
      }

    private:

      void initializeKernel(real tolerance, real gaussianWidth, real He, int maximumSupport, int support){
	sys->log<System::DEBUG>("[DPPoissonSlab] Initialize kernel");
	double h = grid.cellSize.x;
	auto n = grid.cellDim;
	this->kernel = std::make_shared<Kernel>(tolerance, gaussianWidth, h, H, He, n.z, support);
	sys->log<System::MESSAGE>("[DPPoissonSlab] Kernel support %g (%d %d %d cells)", kernel->support.x*h, kernel->support.x, kernel->support.y, kernel->support.z);
	sys->log<System::MESSAGE>("[DPPoissonSlab] Maximum support allowed: %d", maximumSupport);
	sys->log<System::MESSAGE>("[DPPoissonSlab] Kernel width %g", gaussianWidth);
	sys->log<System::MESSAGE>("[DPPoissonSlab] Grid XY spacing %g", h);
	sys->log<System::MESSAGE>("[DPPoissonSlab] Spread weight at h: %g", kernel->delta(make_real3(0,0,h), make_real3(h)));
	sys->log<System::MESSAGE>("[DPPoissonSlab] Spread at maximum distance (%g) : %g", h*(kernel->support.x/2), kernel->delta(make_real3(0,0, h*(kernel->support.x/2)-1e-5), make_real3(h)));
      }

      void initializeQuadratureWeights(){
	sys->log<System::DEBUG>("[DPPoissonSlab Spread] Initialize quadrature weights");
	real lz =  grid.box.boxSize.z;
	real hx = grid.cellSize.x;
	real hy = grid.cellSize.y;
	int nz =  grid.cellDim.z;
	qw = std::make_shared<QuadratureWeights>(lz, hx, hy, nz);
      }

      void spreadBottomWallImages(SeparatedCharges &sep, cached_vector<real> &gridCharges, Permitivity permitivity, cudaStream_t st){
	int* bottomImagesIndices = thrust::raw_pointer_cast(sep.imagesBottom.data());
	spreadImagesWithIndices(bottomImagesIndices, gridCharges, permitivity.bottom/permitivity.inside, sep.imagesBottom.size(), -0.5*H, st);
      }

      void spreadTopWallImages(SeparatedCharges &sep, cached_vector<real> &gridCharges, Permitivity permitivity, cudaStream_t st){
	int* topImagesIndices = thrust::raw_pointer_cast(sep.imagesTop.data());
	spreadImagesWithIndices(topImagesIndices, gridCharges, permitivity.top/permitivity.inside, sep.imagesTop.size(), 0.5*H, st);
      }

      void spreadImagesWithIndices(int* indices, cached_vector<real> &gridCharges, real permitivityRatio, int numberImages, real wallZ, cudaStream_t st){
	real* d_gridCharges = thrust::raw_pointer_cast(gridCharges.data());
	auto charges = pd->getCharge(access::location::gpu, access::mode::read);
	auto chargeImages = thrust::make_transform_iterator(indices,
			    ImageChargeTransform(charges.raw(), permitivityRatio));
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto pos_perm = thrust::make_permutation_iterator(pos, indices);
	auto posImages = thrust::make_transform_iterator(pos_perm, ImagePositionTransform(wallZ));
	auto group = make_charge_group(posImages, chargeImages, numberImages);
	spreadGroup(d_gridCharges, group, st);
      }

      void spreadFromIndices(int* indices, real* gridCharges, int numberParticles, cudaStream_t st){
	sys->log<System::DEBUG>("Spreading from indices");
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto charges = pd->getCharge(access::location::gpu, access::mode::read);
	auto pos_perm = thrust::make_permutation_iterator(pos, indices);
	auto charge_perm = thrust::make_permutation_iterator(charges.begin(), indices);
	auto group = make_charge_group(pos_perm, charge_perm, numberParticles);
	spreadGroup(gridCharges, group, st);
      }

      template<class ChargeGroup>
      void spreadGroup(real* d_gridCharges, ChargeGroup group, cudaStream_t st){
	if(group.numberParticles == 0) return;
	sys->log<System::DEBUG>("Spreading %d particles", group.numberParticles);
	auto minusChargeIterator = thrust::make_transform_iterator(group.charge, thrust::negate<real>());
	int3 n = grid.cellDim;
	IBM<Kernel, Grid> ibm(kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
	ibm.spread(group.pos, minusChargeIterator, d_gridCharges, group.numberParticles, st);
	CudaCheckError();
      }
    };
  }
}
#endif
