/*Raul P. Pelaez 2019-2020. Spectral Poisson solver with Ewald splitting.
TODO:
100- Heuristics for selecting cut offs with tolerance need reevaluation
50- In place C2R cufft does not work. I believe cufft just does not allow it, but I am not sure.
 */

#include"SpectralEwaldPoisson.cuh"
#include"utils/cufftDebug.h"
#include <thrust/iterator/zip_iterator.h>
namespace uammd{

  namespace Poisson_ns{

    __host__ __device__ real greensFunction(real r2, real gw, real split, real epsilon){
      double G = 0;
      if(r2>gw*gw*gw*gw){
	double r = sqrt(r2);
	G = (1.0/(4.0*M_PI*epsilon*r)*(erf(r/(2*gw)) - erf(r/sqrt(4*gw*gw+1/(split*split)))));
      }
      else{
	const double pi32 = pow(M_PI,1.5);
	const double gw2 = gw*gw;
	const double invsp2 = 1.0/(split*split);
	const double selfterm = 1.0/(4*pi32*gw) - 1.0/(2*pi32*sqrt(4*gw2+invsp2));
	const double r2term = 1.0/(6.0*pi32*pow(4.0*gw2 + invsp2, 1.5)) - 1.0/(48.0*pi32*gw2*gw);
	const double r4term = 1.0/(640.0*pi32*gw2*gw2*gw) - 1.0/(20.0*pi32*pow(4*gw2+invsp2,2.5));
	G = 1.0/epsilon*(selfterm+r2*r2term + r2*r2*r4term);
      }
      return G;
    }

    __device__ __host__ real greensFunctionField(real r, real gw, real split, real epsilon){
      double r2 = r*r;
      double gw2 = gw*gw;
      double newgw = sqrt(gw2+1/(4.0*split*split));
      double newgw2 = newgw*newgw;
      double fmod = 0;
      if(r2>gw*gw*gw*gw){
	double invrterm = exp(-0.25*r2/newgw2)/sqrt(M_PI*newgw2) - exp(-0.25*r2/gw2)/sqrt(M_PI*gw2);
	double invr2term = erf(0.5*r/newgw) - erf(0.5*r/gw);

	fmod += 1/(4*M_PI)*( invrterm/r - invr2term/r2);
      }
      else if (r2>0){
	const double pi32 = pow(M_PI, 1.5);
	double rterm = 1/(24*pi32)*(1.0/(gw2*gw) - 1/(newgw2*newgw));
	double r3term = 1/(160*pi32)*(1.0/(newgw2*newgw2*newgw) - 1.0/(gw2*gw2*gw));
	fmod += r*rterm+r2*r*r3term;
      }
      return fmod/epsilon;
    }

    template<class T> std::shared_ptr<T> allocateTemporaryArray(size_t numberElements){
      auto alloc = System::getTemporaryDeviceAllocator<T>();
      return std::shared_ptr<T>(alloc.allocate(numberElements), [=](T* ptr){ alloc.deallocate(ptr);});
    }
  }

  Poisson::Poisson(shared_ptr<ParticleData> pd,
		   shared_ptr<ParticleGroup> pg,
		   shared_ptr<System> sys,
		   Poisson::Parameters par):
    Interactor(pd, pg, sys, "IBM::Poisson"),
    epsilon(par.epsilon),
    box(par.box),
    split(par.split),
    gw(par.gw),
    tolerance(par.tolerance){
    double h;
    double farFieldGaussianWidth = par.gw;
    if(par.split > 0) farFieldGaussianWidth = sqrt(par.gw*par.gw+1.0/(4.0*par.split*par.split));
    if(par.upsampling>0) h = 1.0/par.upsampling;
    else h = (1.3 - std::min((-log10(par.tolerance))/10.0, 0.9))*farFieldGaussianWidth;
    h = std::min(h, box.boxSize.x/16.0);
    sys->log<System::MESSAGE>("[Poisson] Proposed h: %g", h);
    {
      int3 cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h));
      grid = Grid(par.box, cellDim);
      h = grid.cellSize.x;
    }
    sys->log<System::MESSAGE>("[Poisson] Selected h: %g", h);
    int ncells = grid.getNumberCells();
    this->kernel = std::make_shared<Kernel>(par.tolerance, farFieldGaussianWidth, h);
    if(kernel->support > grid.cellDim.x/2-1){
      sys->log<System::ERROR>("[Poisson] Kernel support is too large for this configuration, try increasing splitting parameter or decrasing tolerance");
      throw std::invalid_argument("[Poisson] Kernel support is too large");
    }
    kernel->support = std::min(kernel->support, grid.cellDim.x/2-2);
    if(split>0){
      long double E=1;
      long double r = farFieldGaussianWidth;
      while(abs(E)>par.tolerance){
	r+=0.001l;
	E = Poisson_ns::greensFunction(r*r, gw, split, epsilon);
      }
      nearFieldCutOff = r;
      if(nearFieldCutOff > box.boxSize.x/2.0){
	sys->log<System::ERROR>("[Poisson] Near field cut off is too large, increase splitting parameter.");
	throw std::invalid_argument("[Poisson] Near field cut off is too large");
      }
    }
    sys->log<System::MESSAGE>("[Poisson] tolerance: %g", par.tolerance);
    sys->log<System::MESSAGE>("[Poisson] support: %d", kernel->support);
    sys->log<System::MESSAGE>("[Poisson] epsilon: %g", epsilon);
    sys->log<System::MESSAGE>("[Poisson] Gaussian source width: %g", par.gw);
    sys->log<System::MESSAGE>("[Poisson] cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
    sys->log<System::MESSAGE>("[Poisson] box size: %g %g %g", box.boxSize.x, box.boxSize.y, box.boxSize.z);
    if(par.split> 0){
      sys->log<System::MESSAGE>("[Poisson] Ewald split mode enabled");
      sys->log<System::MESSAGE>("[Poisson] split: %g", par.split);
      sys->log<System::MESSAGE>("[Poisson] Far field width: %g, (%g times original width)",
				farFieldGaussianWidth, 1/par.gw*sqrt(1/(4*par.split*par.split)+par.gw*par.gw));
      sys->log<System::MESSAGE>("[Poisson] Near field cut off: %g", nearFieldCutOff);
    }
    CudaSafeCall(cudaStreamCreate(&st));
    initCuFFT();
    if(split){
      //TODO: I need a better heuristic to select the table size
      int Ntable = std::max(4096, std::min(1<<16, int(nearFieldCutOff/(gw*tolerance*1e3))));
      sys->log<System::MESSAGE>("[Poisson] Elements in near field table: %d", Ntable);
      nearFieldGreensFunctionTable.resize(Ntable);
      real* ptr = thrust::raw_pointer_cast(nearFieldGreensFunctionTable.data());
      nearFieldGreensFunction = std::make_shared<TabulatedFunction<real>>(ptr, Ntable, 0, nearFieldCutOff,
					[=](real r){
					  return Poisson_ns::greensFunctionField(r, gw, split, epsilon);
					});
      nearFieldPotentialGreensFunctionTable.resize(Ntable);
      ptr = thrust::raw_pointer_cast(nearFieldPotentialGreensFunctionTable.data());
      nearFieldPotentialGreensFunction = std::make_shared<TabulatedFunction<real>>(ptr, Ntable, 0, nearFieldCutOff*nearFieldCutOff,
					[=](real r2){
					  return Poisson_ns::greensFunction(r2, gw, split, epsilon);
					});

    }
    CudaCheckError();
  }

  Poisson::~Poisson(){
    cudaStreamDestroy(st);
  }

  void Poisson::initCuFFT(){
    CufftSafeCall(cufftCreate(&cufft_plan_forward));
    CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
    CufftSafeCall(cufftCreate(&cufft_plan_inverse));
    CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));
    //Required storage for the plans
    size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
    /*Set up cuFFT*/
    //This sizes have to be reversed according to the cufft docs
    int3 cdtmp = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x};
    int3 inembed = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x/2+1};
    int3 oembed = {grid.cellDim.z, grid.cellDim.y, 2*(grid.cellDim.x/2+1)};
    //A single forward fft for the charges
    CufftSafeCall(cufftMakePlan3d(cufft_plan_forward, cdtmp.x, cdtmp.y, cdtmp.z,
				  CUFFT_Real2Complex<real>::value,
				  &cufftWorkSizef));
    sys->log<System::DEBUG>("[BDHI::Poisson] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
    //Force and energy in a single transform
    /*I want to make four 3D FFTs, each one using one of the three interleaved coordinates*/
    CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
				    3, &cdtmp.x, /*Three dimensional FFT*/
				    &inembed.x,
				    /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				    4, 1, //Each element separated by four others fx0 fy0 fz0 e0 fx1 fy1 fz1 e1...
				    &oembed.x,
				    4, 1,
				    CUFFT_Complex2Real<real>::value, 4,
				    &cufftWorkSizei));
    size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);
    size_t free_mem, total_mem;
    CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
    sys->log<System::DEBUG>("[BDHI::Poisson] Necessary work space for cuFFT: %s, available: %s, total: %s",
			    printUtils::prettySize(cufftWorkSize).c_str(),
			    printUtils::prettySize(free_mem).c_str(),
			    printUtils::prettySize(total_mem).c_str());
    try{
      cufftWorkArea.resize(cufftWorkSize);
    }
    catch(...){
      sys->log<System::ERROR>("[BDHI::Poisson] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
				 printUtils::prettySize(free_mem).c_str(),
				 printUtils::prettySize(cufftWorkSize).c_str());
      throw std::bad_alloc();
    }

    auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
    CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
    CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
  }

  namespace Poisson_ns{

    struct NearFieldEnergyTransverser{
      using returnInfo = real;

      NearFieldEnergyTransverser(real* energy_ptr, real* charge, TabulatedFunction<real> gf, Box box):
	energy_ptr(energy_ptr), charge(charge), greensFunction(gf), box(box){}

      inline __device__ returnInfo zero() const{ return 0;}

      inline __device__ real getInfo(int pi) const{ return charge[pi];}

      inline __device__ returnInfo compute(const real4 &pi, const real4 &pj, real chargei, real chargej) const{
	real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
	real r2 = dot(rij, rij);
	return chargei*chargej*greensFunction(r2);
      }

      inline __device__ void accumulate(returnInfo &total, const returnInfo &current) const {total += current;}

      inline __device__ void set(uint pi, const returnInfo &total) const {energy_ptr[pi] += total;}
    private:
      real* energy_ptr;
      real* charge;
      TabulatedFunction<real> greensFunction;
      Box box;
    };

    struct NearFieldForceTransverser{
      using returnInfo = real3;

      NearFieldForceTransverser(real4* force_ptr, real* charge, TabulatedFunction<real> gff, Box box):
	 force_ptr(force_ptr), greensFunctionField(gff), charge(charge), box(box){}

      inline __device__ returnInfo zero() const{ return returnInfo();}

      inline __device__ real getInfo(int pi) const{ return charge[pi];}

      inline __device__ returnInfo compute(const real4 &pi, const real4 &pj, real chargei, real chargej) const{
	real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
	real r2 = dot(rij, rij);
	real r = sqrt(r2);
	real fmod = -chargei*chargej*greensFunctionField(r);
	return (r2>0)?(fmod*rij/r):real3();
      }
      inline __device__ void accumulate(returnInfo &total, const returnInfo &current) const {total += current;}
      inline __device__ void set(uint pi, const returnInfo &total) const {force_ptr[pi] += make_real4(total);}
    private:
      TabulatedFunction<real> greensFunctionField;
      real4* force_ptr;
      real* charge;
      Box box;
    };

  }

  void Poisson::farField(cudaStream_t st){
    sys->log<System::DEBUG2>("[Poisson] Far field computation");
    int3 n = grid.cellDim;
    auto gridCharges = Poisson_ns::allocateTemporaryArray<real>(2*(n.x/2+1)*n.y*n.z);
    auto  gridFieldPotentialFourier = Poisson_ns::allocateTemporaryArray<cufftComplex4>((n.x/2+1)*n.y*n.z);
    thrust::fill(thrust::cuda::par.on(st), gridCharges.get(), gridCharges.get() + 2*(n.x/2+1)*n.y*n.z, 0);
    spreadCharges(gridCharges.get());
    forwardTransformCharge(gridCharges.get(), (cufftComplex*) gridCharges.get());
    convolveFourier((cufftComplex*) gridCharges.get(), gridFieldPotentialFourier.get());
    gridCharges.reset();
    auto gridFieldPotential = Poisson_ns::allocateTemporaryArray<real4>(2*(n.x/2+1)*n.y*n.z);
    inverseTransform((cufftComplex*)gridFieldPotentialFourier.get(), (real*) gridFieldPotential.get());
    interpolateFields(gridFieldPotential.get());
  }

  void Poisson::nearFieldForce(cudaStream_t st){
    if(split>0){
      sys->log<System::DEBUG2>("[Poisson] Near field force computation");
      if(!nl) nl = std::make_shared<NeighbourList>(pd, pg, sys);
      nl->update(box, nearFieldCutOff, st);
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
      auto charge = pd->getCharge(access::location::gpu, access::mode::read);
      auto tr = Poisson_ns::NearFieldForceTransverser(force.begin(), charge.begin(), *nearFieldGreensFunction, box);
      nl->transverseList(tr, st);
    }
  }

  void Poisson::sumForce(cudaStream_t st){
    sys->log<System::DEBUG2>("[Poisson] Sum Force");
    farField(st);
    nearFieldForce(st);
  }

  void Poisson::nearFieldEnergy(cudaStream_t st){
    if(split>0){
      int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG2>("[Poisson] Near field energy computation");
      if(!nl) nl = std::make_shared<NeighbourList>(pd, pg, sys);
      nl->update(box, nearFieldCutOff, st);
      auto charge = pd->getCharge(access::location::gpu, access::mode::read);
      auto energy = pd->getEnergy(access::location::gpu, access::mode::read);
      auto tr = Poisson_ns::NearFieldEnergyTransverser(energy.begin(), charge.begin(), *nearFieldPotentialGreensFunction, box);
      nl->transverseList(tr, st);
    }
  }

  real Poisson::sumEnergy(){
    sys->log<System::DEBUG2>("[Poisson] Sum Energy");
    cudaStream_t st = 0;
    farField(st);
    nearFieldEnergy(st);
    return 0;
  }

  namespace Poisson_ns{
    using cufftComplex4 = cufftComplex4_t<real>;//Poisson::cufftComplex4;
    using cufftComplex = cufftComplex_t<real>;//Poisson::cufftComplex;

    template<class vec3>
    inline __device__ vec3 cellToWaveNumber(const int3 &cell, const int3 &cellDim, const vec3 &L){
      const vec3 pi2invL = (real(2.0)*real(M_PI))/L;
      vec3 k = {cell.x*pi2invL.x,
	cell.y*pi2invL.y,
	cell.z*pi2invL.z};
      if(cell.x >= (cellDim.x/2+1)) k.x -= real(cellDim.x)*pi2invL.x;
      if(cell.y >= (cellDim.y/2+1)) k.y -= real(cellDim.y)*pi2invL.y;
      if(cell.z >= (cellDim.z/2+1)) k.z -= real(cellDim.z)*pi2invL.z;
      return k;
    }

    __device__ bool isNyquist(int3 cell, int3 cellDim){
      bool isXnyquist = (cell.x == cellDim.x - cell.x) && (cellDim.x%2 == 0);
      bool isYnyquist = (cell.y == cellDim.y - cell.y) && (cellDim.y%2 == 0);
      bool isZnyquist = (cell.z == cellDim.z - cell.z) && (cellDim.z%2 == 0);
      return (isXnyquist && cell.y==0   && cell.z==0)  or  //1
	(isXnyquist && isYnyquist  && cell.z==0)  or  //2
	(cell.x==0  && isYnyquist  && cell.z==0)  or  //3
	(isXnyquist && cell.y==0   && isZnyquist) or  //4
	(cell.x==0  && cell.y==0   && isZnyquist) or  //5
	(cell.x==0  && isYnyquist  && isZnyquist) or  //6
	(isXnyquist && isYnyquist  && isZnyquist);    //7
    }

    __global__ void chargeFourier2FieldAndPotential(cufftComplex* gridCharges,
						    cufftComplex4* gridFieldPotential,
						    real epsilon,
						    Grid grid){
      int3 cell;
      cell.x = blockIdx.x*blockDim.x + threadIdx.x;
      cell.y = blockIdx.y*blockDim.y + threadIdx.y;
      cell.z = blockIdx.z*blockDim.z + threadIdx.z;
      if(cell.x>=(grid.cellDim.x/2+1)) return;
      if(cell.y>=grid.cellDim.y) return;
      if(cell.z>=grid.cellDim.z) return;
      const real3 k = cellToWaveNumber(cell, grid.cellDim, grid.box.boxSize);
      const real k2 = dot(k, k);
      if(cell.x == 0 and cell.y == 0 and cell.z == 0){
	gridFieldPotential[0] = cufftComplex4();
	return;
      }
      const int i_icell = cell.x + (cell.y + cell.z*grid.cellDim.y)*(grid.cellDim.x/2+1);
      cufftComplex4 fieldPotential = cufftComplex4();
      const bool nyquist = isNyquist(cell, grid.cellDim);
      if(not nyquist){
	const cufftComplex fk = gridCharges[i_icell];
	const real B = real(1.0)/(k2*epsilon*grid.getNumberCells());
       	fieldPotential.x.x = k.x*fk.y*B; fieldPotential.x.y = -k.x*fk.x*B;
       	fieldPotential.y.x = k.y*fk.y*B; fieldPotential.y.y = -k.y*fk.x*B;
       	fieldPotential.z.x = k.z*fk.y*B; fieldPotential.z.y = -k.z*fk.x*B;
	fieldPotential.w = fk*B;
      }
      gridFieldPotential[i_icell] = fieldPotential;
    }

  }

  void Poisson::spreadCharges(real* gridCharges){
    sys->log<System::DEBUG2>("[Poisson] Spreading charges");
    int numberParticles = pg->getNumberParticles();
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto charges = pd->getCharge(access::location::gpu, access::mode::read);
    int3 n = grid.cellDim;
    IBM<Kernel> ibm(sys, kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
    ibm.spread(pos.begin(), charges.begin(), gridCharges, numberParticles, st);
    CudaCheckError();
  }

  void Poisson::forwardTransformCharge(real *gridCharges, cufftComplex* gridChargesFourier){
    CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
    sys->log<System::DEBUG2>("[Poisson] Taking grid to wave space");
    CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, gridCharges, gridChargesFourier));
  }

  void Poisson::convolveFourier(cufftComplex* gridChargesFourier, cufftComplex4* gridFieldPotentialFourier){
    sys->log<System::DEBUG2>("[Poisson] Wave space convolution");
    dim3 NthreadsCells = dim3(8,8,8);
    dim3 NblocksCells;
    int ncellsx = grid.cellDim.x/2+1;
    NblocksCells.x= (ncellsx/NthreadsCells.x + ((ncellsx%NthreadsCells.x)?1:0));
    NblocksCells.y= grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
    NblocksCells.z= grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
    Poisson_ns::chargeFourier2FieldAndPotential<<<NblocksCells, NthreadsCells, 0, st>>>
      (gridChargesFourier, gridFieldPotentialFourier, epsilon, grid);
    CudaCheckError();
  }

  void Poisson::inverseTransform(cufftComplex* gridFieldPotentialFourier, real* gridFieldPotential){
    sys->log<System::DEBUG2>("[Poisson] Force to real space");
    CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
    CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, gridFieldPotentialFourier, gridFieldPotential));
    CudaCheckError();
  }

  namespace Poisson_ns{

    struct UnZip2Real4{

      real4* force;
      real* energy;
      real* charges;
      int i;

      UnZip2Real4(real* charges, real4* f, real* e):charges(charges),force(f), energy(e), i(-1){}

      __device__ UnZip2Real4 operator()(int ai){
	this->i = ai;
	return *this;
      }

      __device__ void operator += (real4 fande) const{
	if(force) force[i] += charges[i]*make_real4(fande.x, fande.y, fande.z, 0);
	if(energy) energy[i] += charges[i]*fande.w;
      }

    };

  }

  void Poisson::interpolateFields(real4* gridFieldPotential){
    sys->log<System::DEBUG2>("[Poisson] Interpolating forces and energies");
    int numberParticles = pg->getNumberParticles();
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto charge = pd->getCharge(access::location::gpu, access::mode::read);
    auto forces = pd->getForce(access::location::gpu, access::mode::readwrite);
    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
    auto gridData2ForceAndEnergy = Poisson_ns::UnZip2Real4(charge.begin(), forces.begin(), energy.begin());
    auto f_tr = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), gridData2ForceAndEnergy);
    int3 n = grid.cellDim;
    IBM<Kernel> ibm(sys, kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
    ibm.gather(pos.begin(), f_tr, gridFieldPotential, numberParticles, st);
  }


}
