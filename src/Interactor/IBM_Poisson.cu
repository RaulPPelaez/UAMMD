/*Raul P. Pelaez 2019. Spectral Poisson solver
 */

#include"IBM_Poisson.cuh"
#include"utils/cufftPrecisionAgnostic.h"
#include"utils/cufftDebug.h"
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include<fstream>
namespace uammd{

  Poisson::Poisson(shared_ptr<ParticleData> pd,
		   shared_ptr<ParticleGroup> pg,
		   shared_ptr<System> sys,
		   Poisson::Parameters par):
    Interactor(pd, pg, sys, "IBM::Poisson"),
    epsilon(par.epsilon),
    box(par.box){

    double h;
    if(par.upsampling>0) h = par.upsampling;
    else h = (1.3 - std::min((-log10(par.tolerance))/10.0, 0.9))*par.gw;

    {
      int3 cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h+0.5));
      grid = Grid(par.box, cellDim);
      h = grid.cellSize.x;
    }
    int ncells = grid.getNumberCells();

    auto kernel = std::make_shared<Kernel>(par.tolerance, par.gw, h);

    ibm = std::make_shared<IBM<Kernel>>(sys, kernel);

    sys->log<System::MESSAGE>("[Poisson] tolerance: %g", par.tolerance);
    sys->log<System::MESSAGE>("[Poisson] support: %d", kernel->support);
    sys->log<System::MESSAGE>("[Poisson] epsilon: %g", epsilon);
    sys->log<System::MESSAGE>("[Poisson] cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
    sys->log<System::MESSAGE>("[Poisson] box size: %g %g %g", box.boxSize.x, box.boxSize.y, box.boxSize.z);

    CudaSafeCall(cudaStreamCreate(&st));
    CudaCheckError();
    initCuFFT();
  }

  Poisson::~Poisson(){
    CudaSafeCall(cudaStreamDestroy(st));
  }
  void Poisson::initCuFFT(){

    CufftSafeCall(cufftCreate(&cufft_plan_forward));
    CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));

    CufftSafeCall(cufftCreate(&cufft_plan_inverse));
    CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));

    //Required storage for the plans
    size_t cufftWorkSizef = 0, cufftWorkSizei;//f = 0, cufftWorkSizeie = 0;
    /*Set up cuFFT*/
    //This sizes have to be reversed according to the cufft docs
    int3 cdtmp = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x};
    int3 inembed = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x};

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
				    &inembed.x,
				    4, 1,
				    CUFFT_Complex2Real<real>::value, 4,
				    &cufftWorkSizei));

    /*Allocate cuFFT work area*/
    size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei)+10;
    size_t free_mem, total_mem;
    CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));

    sys->log<System::DEBUG>("[BDHI::Poisson] Necessary work space for cuFFT: %s, available: %s, total: %s",
			    printUtils::prettySize(cufftWorkSize).c_str(),
			    printUtils::prettySize(free_mem).c_str(),
			    printUtils::prettySize(total_mem).c_str());

    if(free_mem<cufftWorkSize){
      sys->log<System::CRITICAL>("[BDHI::Poisson] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
				 printUtils::prettySize(free_mem).c_str(),
				 printUtils::prettySize(cufftWorkSize).c_str());
    }

    cufftWorkArea.resize(cufftWorkSize);
    auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());

    CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
    CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
  }

  void Poisson::sumForce(cudaStream_t st){
    sys->log<System::DEBUG2>("[Poisson] Sum Force");

    try{
      gridCharges.resize(2*grid.cellDim.y*grid.cellDim.z*(grid.cellDim.x+1));
      gridForceEnergy.resize(grid.getNumberCells());
      thrust::fill(gridCharges.begin(), gridCharges.end(),
		   std::iterator_traits<decltype(gridCharges.begin())>::value_type());
    }
    catch(thrust::system_error &e){
      sys->log<System::CRITICAL>("[Poisson] Thrust could not reset grid data with error &s", e.what());
    }
    sys->log<System::DEBUG2>("[Poisson] Wave part");
    spreadCharges();
    forwardTransformCharge();
    convolveFourier();
    inverseTransform();
    interpolateFields();

  }

  real Poisson::sumEnergy(){
    sys->log<System::DEBUG2>("[Poisson] Sum Energy");
    sumForce(0);
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

    __global__ void chargeFourier2ForceAndEnergy(cufftComplex* gridCharges,
					cufftComplex4* gridForceEnergy,
				        real epsilon,
					Grid grid){
      int3 cell;
      cell.x= blockIdx.x*blockDim.x + threadIdx.x;
      cell.y= blockIdx.y*blockDim.y + threadIdx.y;
      cell.z= blockIdx.z*blockDim.z + threadIdx.z;

      if(cell.x>=grid.cellDim.x/2+1) return;
      if(cell.y>=grid.cellDim.y) return;
      if(cell.z>=grid.cellDim.z) return;

      const real3 k = cellToWaveNumber(cell, grid.cellDim, grid.box.boxSize);
      const real k2 = dot(k,k);

      if(k2 == 0){
	gridForceEnergy[0] = cufftComplex4();
	return;
      }

      const real B = 1.0/(k2*epsilon*grid.getNumberCells());

      const int i_icell = cell.x + (cell.y + cell.z*grid.cellDim.y)*(grid.cellDim.x/2+1);

      const cufftComplex fk = gridCharges[i_icell];
      cufftComplex4 force = cufftComplex4();

      force.x.x = k.x*fk.y*B; force.x.y = -k.x*fk.x*B;
      force.y.x = k.y*fk.y*B; force.y.y = -k.y*fk.x*B;
      force.z.x = k.z*fk.y*B; force.z.y = -k.z*fk.x*B;
      force.w = fk*B; //Energy

      const int o_icell = grid.getCellIndex(cell);
      gridForceEnergy[o_icell] = force;
    }

  }
  void Poisson::spreadCharges(){
    sys->log<System::DEBUG2>("[Poisson] Spreading charges");
    int numberParticles = pg->getNumberParticles();
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto charges = pd->getCharge(access::location::gpu, access::mode::read);

    real* d_gridCharges = (real*)thrust::raw_pointer_cast(gridCharges.data());

    ibm->spread(pos.begin(), charges.begin(), d_gridCharges, grid, numberParticles, st);
    CudaCheckError();
  }
  void Poisson::forwardTransformCharge(){
    CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
    auto d_gridCharges = thrust::raw_pointer_cast(gridCharges.data());
    auto d_gridChargesFourier = thrust::raw_pointer_cast(gridCharges.data())+grid.getNumberCells();
    sys->log<System::DEBUG2>("[Poisson] Taking grid to wave space");
    {
      auto cufftStatus =
	cufftExecReal2Complex<real>(cufft_plan_forward,
				    (cufftReal*)d_gridCharges,
				    (cufftComplex*)d_gridChargesFourier);
      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[Poisson] Error in forward CUFFT");
      }
    }
  }

  void Poisson::inverseTransform(){
    sys->log<System::DEBUG2>("[Poisson] Force to real space");
    CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
    auto d_gridForceEnergy = thrust::raw_pointer_cast(gridForceEnergy.data());
    auto d_gridForceEnergyFourier = thrust::raw_pointer_cast(gridForceEnergy.data());

    {
      auto cufftStatus =
	cufftExecComplex2Real<real>(cufft_plan_inverse,
				    (cufftComplex*)d_gridForceEnergyFourier,
				    (cufftReal*)d_gridForceEnergy);
      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[Poisson] Error in inverse CUFFT");
      }
    }
    CudaCheckError();
  }


  void Poisson::convolveFourier(){

    auto d_gridChargesFourier = thrust::raw_pointer_cast(gridCharges.data())+grid.getNumberCells();

    cufftComplex4* d_gridForceEnergyFourier = thrust::raw_pointer_cast(gridForceEnergy.data());

    sys->log<System::DEBUG2>("[Poisson] Wave space convolution");
    {
      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;
      {
	int ncellsx = grid.cellDim.x/2+1;
	NblocksCells.x= (ncellsx/NthreadsCells.x + ((ncellsx%NthreadsCells.x)?1:0));
	NblocksCells.y= grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
	NblocksCells.z= grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
      }

      Poisson_ns::chargeFourier2ForceAndEnergy<<<NblocksCells, NthreadsCells, 0, st>>>
	((cufftComplex*) d_gridChargesFourier,
	 d_gridForceEnergyFourier,
	 epsilon,
	 grid);
    }
    CudaCheckError();

  }

  namespace Poisson_ns{

    struct toReal4{
      __device__ real4 operator()(real3 a){
	return make_real4(a);
      }
    };

    struct Zip2Real4{

      real4* force;
      real* energy;
      int i;

      Zip2Real4(real4* f, real* e):force(f), energy(e), i(-1){}
      __device__ Zip2Real4 operator()(int ai){
	this->i = ai;
	return *this;
      }

      __device__ void operator += (real4 fande){
	force[i] += make_real4(fande.x, fande.y, fande.z, 0);
	energy[i] += fande.w;
      }
    };

  }
  void Poisson::interpolateFields(){
    sys->log<System::DEBUG2>("[Poisson] Interpolating forces and energies");

    int numberParticles = pg->getNumberParticles();
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto forces = pd->getForce(access::location::gpu, access::mode::readwrite);
    auto energies = pd->getEnergy(access::location::gpu, access::mode::readwrite);

    real4* d_gridForcesEnergies = (real4*)thrust::raw_pointer_cast(gridForceEnergy.data());
    //Transform real4 grid data into separated force and energy particle arrays
    auto f_tr = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
						Poisson_ns::Zip2Real4(forces.begin(),
								      energies.begin())
						);

    ibm->gather(pos.begin(),
   		f_tr,
   		d_gridForcesEnergies,
   		grid, numberParticles, st);


  }

}