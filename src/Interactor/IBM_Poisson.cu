/*Raul P. Pelaez 2019-2020. Spectral Poisson solver with Ewald splitting.
TODO:
100- Heuristics for selecting cut offs with tolerance need reevaluation
 */

#include"IBM_Poisson.cuh"
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
      nearFieldGreensFunction = TabulatedFunction<real>(ptr, Ntable, 0, nearFieldCutOff,
					[=](real r){
					  return Poisson_ns::greensFunctionField(r, gw, split, epsilon);
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

      NearFieldEnergyTransverser(real* potential_ptr, real* charge,
				 real ep, real sp, real gw, Box box):
	potential_ptr(potential_ptr), charge(charge),
	epsilon(ep), split(sp), gw(gw), box(box){}

      inline __device__ returnInfo zero() const{ return 0.0f;}
      inline __device__ real getInfo(int pi) const{ return charge[pi];}

      inline __device__ returnInfo compute(const real4 &pi, const real4 &pj, real chargei, real chargej) const{
	real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
	real r2 = dot(rij, rij);
	return chargej*greensFunction(r2, gw, split, epsilon);
      }

      inline __device__ void accumulate(returnInfo &total, const returnInfo &current) const {total += current;}

      inline __device__ void set(uint pi, const returnInfo &total) const {potential_ptr[pi] += total;}
    private:
      real* potential_ptr;
      real* charge;
      real epsilon, split, gw;
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
    resetGridData();
    spreadCharges();
    forwardTransformCharge();
    convolveFourier();
    inverseTransform();
    interpolateFields();
  }

  void Poisson::resetGridData(){
    try{
      int ncells = grid.cellDim.y*grid.cellDim.z*(grid.cellDim.x/2+1);
      gridCharges.resize(2*ncells);
      gridFieldPotential.resize(ncells);
      thrust::fill(thrust::cuda::par.on(st),
		   gridCharges.begin(), gridCharges.end(),
		   std::iterator_traits<decltype(gridCharges.begin())>::value_type());
    }
    catch(thrust::system_error &e){
      sys->log<System::ERROR>("[Poisson] Thrust could not reset grid data with error &s", e.what());
      throw;
    }
  }

  void Poisson::sumForce(cudaStream_t st){
    sys->log<System::DEBUG2>("[Poisson] Sum Force");
    farField(st);
    if(split>0){
      sys->log<System::DEBUG2>("[Poisson] Near field force computation");
      if(!nl) nl = std::make_shared<NeighbourList>(pd, pg, sys);
      nl->update(box, nearFieldCutOff, st);
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
      auto charge = pd->getCharge(access::location::gpu, access::mode::read);
      auto tr = Poisson_ns::NearFieldForceTransverser(force.begin(), charge.begin(), nearFieldGreensFunction, box);
      nl->transverseList(tr, st);
    }
  }

  namespace Poisson_ns{
    struct Potential2Energy{
      real originPotential;
      Potential2Energy(real originPotential): originPotential(originPotential){}

      __device__ real operator()(thrust::tuple<real,real> chargeAndPotential){
	real q = thrust::get<0>(chargeAndPotential);
	real phi = thrust::get<1>(chargeAndPotential) - originPotential;
	return q*phi;
      }
    };
  }

  real Poisson::sumEnergy(){
    sys->log<System::DEBUG2>("[Poisson] Sum Energy");
    cudaStream_t st = 0;
    farField(st);
    int numberParticles = pg->getNumberParticles();
    if(split>0){
      sys->log<System::DEBUG2>("[Poisson] Near field energy computation");
      if(!nl) nl = std::make_shared<NeighbourList>(pd, pg, sys);
      nl->update(box, nearFieldCutOff, st);
      auto potential = thrust::raw_pointer_cast(potentialAtCharges.data());
      auto charge = pd->getCharge(access::location::gpu, access::mode::read);
      auto tr = Poisson_ns::NearFieldEnergyTransverser(potential, charge.begin(), epsilon, split, gw, box);
      nl->transverseList(tr, st);
    }
    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
    auto charge = pd->getCharge(access::location::gpu, access::mode::read);
    real originPotential = measurePotentialAtOrigin();
    auto chargeAndPotential = thrust::make_zip_iterator(thrust::make_tuple(charge.begin(), potentialAtCharges.begin()));
    thrust::transform(thrust::cuda::par.on(st),
		      chargeAndPotential, chargeAndPotential + numberParticles, energy.begin(),
		      Poisson_ns::Potential2Energy(originPotential));
    return 0;
  }

  namespace Poisson_ns{
    using cufftComplex3 = Poisson::cufftComplex3;
    using cufftComplex4 = Poisson::cufftComplex4;
    using cufftComplex = Poisson::cufftComplex;

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
      if(k2 == 0){
	gridFieldPotential[0] = cufftComplex4();
	return;
      }
      const int i_icell = cell.x + (cell.y + cell.z*grid.cellDim.y)*(grid.cellDim.x/2+1);
      cufftComplex4 fieldPotential = cufftComplex4();
      bool nyquist = isNyquist(cell, grid.cellDim);
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

  void Poisson::spreadCharges(){
    sys->log<System::DEBUG2>("[Poisson] Spreading charges");
    int numberParticles = pg->getNumberParticles();
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto charges = pd->getCharge(access::location::gpu, access::mode::read);
    real* d_gridCharges = (real*)thrust::raw_pointer_cast(gridCharges.data());
    int3 n = grid.cellDim;
    IBM<Kernel> ibm(sys, kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
    ibm.spread(pos.begin(), charges.begin(), d_gridCharges, numberParticles, st);
    CudaCheckError();
  }

  void Poisson::forwardTransformCharge(){
    CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
    auto d_gridCharges = thrust::raw_pointer_cast(gridCharges.data());
    auto d_gridChargesFourier = thrust::raw_pointer_cast(gridCharges.data());
    sys->log<System::DEBUG2>("[Poisson] Taking grid to wave space");
    auto cufftStatus = cufftExecReal2Complex<real>(cufft_plan_forward, (cufftReal*)d_gridCharges, (cufftComplex*)d_gridChargesFourier);
    if(cufftStatus != CUFFT_SUCCESS){
      sys->log<System::ERROR>("[Poisson] Error in forward CUFFT");
      throw std::runtime_error("CUFFT Error");
    }
  }

  void Poisson::inverseTransform(){
    sys->log<System::DEBUG2>("[Poisson] Force to real space");
    CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
    auto d_gridFieldPotential = thrust::raw_pointer_cast(gridFieldPotential.data());
    auto d_gridFieldPotentialFourier = thrust::raw_pointer_cast(gridFieldPotential.data());
    auto cufftStatus = cufftExecComplex2Real<real>(cufft_plan_inverse,
						   (cufftComplex*)d_gridFieldPotentialFourier,
						   (cufftReal*)d_gridFieldPotential);
    if(cufftStatus != CUFFT_SUCCESS){
      sys->log<System::ERROR>("[Poisson] Error in inverse CUFFT");
      throw std::runtime_error("CUFFT Error");
    }
    CudaCheckError();
  }

  void Poisson::convolveFourier(){
    auto d_gridChargesFourier = thrust::raw_pointer_cast(gridCharges.data());
    cufftComplex4* d_gridFieldPotentialFourier = thrust::raw_pointer_cast(gridFieldPotential.data());
    sys->log<System::DEBUG2>("[Poisson] Wave space convolution");
    dim3 NthreadsCells = dim3(8,8,8);
    dim3 NblocksCells;
    int ncellsx = grid.cellDim.x/2+1;
    NblocksCells.x= (ncellsx/NthreadsCells.x + ((ncellsx%NthreadsCells.x)?1:0));
    NblocksCells.y= grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
    NblocksCells.z= grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
    Poisson_ns::chargeFourier2FieldAndPotential<<<NblocksCells, NthreadsCells, 0, st>>>
      ((cufftComplex*) d_gridChargesFourier,
       d_gridFieldPotentialFourier,
       epsilon,
       grid);
    CudaCheckError();
  }

  namespace Poisson_ns{

    struct toReal4{
      __device__ real4 operator()(real3 a){
	return make_real4(a);
      }
    };

    struct UnZip2Real4{

      real4* force;
      real* potential;
      real* charges;
      int i;

      UnZip2Real4(real* charges, real4* f, real* phi):charges(charges),force(f), potential(phi), i(-1){}
      __device__ UnZip2Real4 operator()(int ai){
	this->i = ai;
	return *this;
      }

      __device__ void operator += (real4 fande){
	if(force) force[i] += charges[i]*make_real4(fande.x, fande.y, fande.z, 0);
	if(potential) potential[i] = fande.w;
      }

    };

  }

  void Poisson::interpolateFields(){
    sys->log<System::DEBUG2>("[Poisson] Interpolating forces and energies");
    int numberParticles = pg->getNumberParticles();
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto charge = pd->getCharge(access::location::gpu, access::mode::read);
    auto forces = pd->getForce(access::location::gpu, access::mode::readwrite);
    potentialAtCharges.resize(numberParticles);
    auto potential = thrust::raw_pointer_cast(potentialAtCharges.data());
    auto d_gridFieldPotential = (real4*)thrust::raw_pointer_cast(gridFieldPotential.data());
    auto gridData2ForceAndPotential = Poisson_ns::UnZip2Real4(charge.begin(), forces.begin(), potential);
    auto f_tr = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), gridData2ForceAndPotential);
    int3 n = grid.cellDim;
    IBM<Kernel> ibm(sys, kernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
    ibm.gather(pos.begin(), f_tr, d_gridFieldPotential, numberParticles, st);
  }

  real Poisson::measurePotentialAtOrigin(){
    auto phiFar = measurePotentialAtOriginFarField();
    auto phiNear = measurePotentialAtOriginNearField();
    return phiFar + phiNear;
  }

  real Poisson::measurePotentialAtOriginFarField(){
    real4* d_gridFieldPotential = (real4*)thrust::raw_pointer_cast(gridFieldPotential.data());
    originPotential.resize(1);
    originPotential[0] = 0;
    auto originPotential_ptr = thrust::raw_pointer_cast(originPotential.data());
    auto f_tr = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
						Poisson_ns::UnZip2Real4(nullptr, nullptr, originPotential_ptr));
    auto originPos = thrust::make_constant_iterator<real4>(real4());
    real h = grid.cellSize.x;
    auto kernel = std::make_shared<Kernel>(tolerance, 1.0/(2.0*split), h);
    IBM<Kernel> ibmOrigin(sys, kernel, grid);
    ibmOrigin.gather(originPos, f_tr, d_gridFieldPotential, 1, st);
    real phifar = originPotential[0];
    return phifar;
  }

  namespace Poisson_ns{

    struct PotentialAtOrigin{
      real3 origin;
      Box box;
      real gw, split, epsilon;
      PotentialAtOrigin(Box box, real3 origin, real gw, real split, real epsilon):box(box), origin(origin){}

      __device__ real operator()(thrust::tuple<real4, real> posCharge){
	real3 pos = make_real3(thrust::get<0>(posCharge));
	real3 rij = box.apply_pbc(pos - origin);
	real r2 =  dot(rij, rij);
	real q = thrust::get<1>(posCharge);
	real phi = q*greensFunction(r2, gw, split, epsilon);
	return phi;
      }

    };

  }

  real Poisson::measurePotentialAtOriginNearField(){
    real phiNear = 0;
    if(split){
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto charge = pd->getCharge(access::location::gpu, access::mode::read);
      auto posCharge = thrust::make_zip_iterator(thrust::make_tuple(pos.begin(), charge.begin()));
      int numberParticles = pg->getNumberParticles();
      phiNear = thrust::transform_reduce(thrust::cuda::par,
					 posCharge, posCharge + numberParticles,
					 Poisson_ns::PotentialAtOrigin(grid.box, real3(), gw/sqrt(2), split, epsilon),
					 0, thrust::plus<real>());
    }
    return phiNear;
  }


}
