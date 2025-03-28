/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver. Correction

 */
#include "misc/BoundaryValueProblem/BVPSchurComplementMatrices.cuh"
#include "misc/BoundaryValueProblem/BVPSolver.cuh"
#include"utils.cuh"
#include"misc/BoundaryValueProblem/MatrixUtils.h"
#include <iterator>
namespace uammd{
  namespace DPStokesSlab_ns{

    namespace correction_ns{
      __global__ void periodicExtension(complex* signal, int nz, int nbatch){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= nbatch*nz) return;
	const int ib = id%nbatch;
	const int iz  = id/(nbatch);
	if(iz>=nz-1 or iz == 0) return;
	const int zf = 2*nz-2-iz;
	const int zi = iz;
	signal[ib + zf*nbatch]  = signal[ib + zi*nbatch];
      }

      __global__ void scaleFFTToForwardChebyshevTransform(complex* signal, int nz, int nbatch){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= nbatch*nz) return;
	const int iz  = id/nbatch;
	const int ib = id%nbatch;
	const int zf = 2*nz-2-iz;
	const int zi = iz;
	if(iz>0 && iz<nz-1)
	  signal[ib + zi*nbatch] += signal[ib + zf*nbatch];
	signal[ib + zi*nbatch] *= real(1.0)/real(2*nz-2);
      }

      //For bottom wall x or y velocity have the BCs, We only need k=0 but I left the BCs generic for any k
      //u(k=0, z=0) = 0 and u'(k=0, z=H) = 0
      //For slit channel we have
      //u(k=0, z=0,H) = 0
      //In both cases z velocity is 0: w(k=0, z) = 0
      class TopBoundaryConditions{
	real k, H;
      public:
	TopBoundaryConditions(real k, real H):k(k),H(H){
	}

	real getFirstIntegralFactor() const{
	  return (k!=0)?H:1.0;
	}

	real getSecondIntegralFactor() const{
	  return k!=0?(H*H*k):(0.0);
	}
      };

      class BottomBoundaryConditions{
	real k, H;
      public:
	BottomBoundaryConditions(real k, real H):k(k),H(H){
	}

	real getFirstIntegralFactor() const{
	  return (k!=0)*H;
	}

	real getSecondIntegralFactor() const{
	  return k!=0?(-H*H*k):(-1.0);
	}
      };

      using SlitBoundaryConditions = BottomBoundaryConditions;

      //Solves the zero mode of the BVP y''= fn with Boundary conditions given by C and D
      //C and D are the Boundary condition matrices
      //an is temporal storage for the cheb coefficients of y''
      __global__ void correctZeroMode(FluidPointers<complex> correction,
				      DataXYZPtr<complex> gridForce,
				      complex* an, real4 D, real*C,
				      int nkx, int nky, int nz, real H, real mu){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id!=0) return;
	auto indexer = Index3D(nkx/2+1, nky, 2*nz-2);
	{//X VELOCITY
	  auto fxn = make_third_index_iterator(gridForce.x(), 0, 0, indexer);
	  complex b1 = complex();
	  complex b2 = complex();
	  //Compute C*f and store y''= fn
	  for(int i = 0; i<nz; i++){
	    b1 += -C[i]*fxn[i]/mu;
	    b2 += -C[nz+i]*fxn[i]/mu;
	    an[i] = -fxn[i]/mu;
	  }
	  auto c0d0 = BVP::solve2x2System(-D, thrust::make_pair(b1,b2));
	  BVP::SecondIntegralMatrix si(nz);
	  auto corr = make_third_index_iterator(correction.velocityX, 0, 0, indexer);
	  for(int i = 0; i<nz; i++){
	    corr[i] = si.computeSecondIntegralCoefficient(i, an, c0d0.first, c0d0.second)*H*H;
	  }
	}
	{//Y VELOCITY
	  auto fyn = make_third_index_iterator(gridForce.y(), 0, 0, indexer);
	  complex b1 = complex();
	  complex b2 = complex();
	  //Compute C*f and store y''= fn
	  for(int i = 0; i<nz; i++){
	    b1 += -C[i]*fyn[i]/mu;
	    b2 += -C[nz+i]*fyn[i]/mu;
	    an[i] = -fyn[i]/mu;
	  }
	  auto c0d0 = BVP::solve2x2System(-D, thrust::make_pair(b1,b2));
	  BVP::SecondIntegralMatrix si(nz);
	  auto vycorr = make_third_index_iterator(correction.velocityY, 0, 0, indexer);
	  auto vzcorr = make_third_index_iterator(correction.velocityZ, 0, 0, indexer);
	  auto pcorr = make_third_index_iterator(correction.pressure, 0, 0, indexer);
	  for(int i = 0; i<nz; i++){
	    vycorr[i] = si.computeSecondIntegralCoefficient(i, an, c0d0.first, c0d0.second)*H*H;
	    vzcorr[i] *=0; //Z velocity zero mode correction is zero in both bottom and slit modes
	    pcorr[i] *=0; //Do not correct pressure
	  }
	}
      }

      //Set the zero mode of the inside solution to zero. It is used to store the grid forces for the zero mode before the correction
      __global__ void cleanSolutionZeroMode(complex4* insideSolution, int nkx, int nky, int nz){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id!=0) return;
	auto fn = make_third_index_iterator(insideSolution, 0, 0, Index3D(nkx/2+1, nky, 2*nz-2));
	for(int i = 0; i<nz; i++){
	  fn[i] *= 0;
	}
      }

    }

    template<class Iterator>
    void sumRange(Iterator a, Iterator b, int size, cudaStream_t st){
      thrust::transform(thrust::cuda::par.on(st),
			a, a+size,
			b, a,
			thrust::plus<complex>());

    }

    void sumCorrectionToInsideSolution(FluidData<complex> &correction,
				       FluidData<complex> &insideSolution,
				       int size,
				       cudaStream_t  st){
      System::log<System::DEBUG>("Sum correction to solution");
      sumRange(insideSolution.pressure.begin(), correction.pressure.begin(), size, st);
      sumRange(insideSolution.velocity.x(), correction.velocity.x(), size, st);
      sumRange(insideSolution.velocity.y(), correction.velocity.y(), size, st);
      sumRange(insideSolution.velocity.z(), correction.velocity.z(), size, st);
    }

    class Correction{
      real H;
      real2 Lxy;
      int3 cells;
      real viscosity;
      WallMode mode = WallMode::none;

      gpu_container<complex> invA;
      gpu_container<real> zeroModeCMatrix;
      real4 zeroModeDMatrix;
      cufftHandle cufft_plan_forward;

      void initializeSlitCorrectionMatrix();
      void initializeCufft();
      void initializeZeroModeBVPSolver();
      FluidData<complex> computeAnalyticalCorrectionFourierSpace(const FluidData<complex> &insideSolution,
								 cudaStream_t st);
    public:
      struct Parameters{
	real H;
	real2 Lxy;
	int3 cells;
	real viscosity;
	WallMode mode = WallMode::none;
      };

      Correction(real H, real2 Lxy, int3 cells, real viscosity, WallMode mode):
      	H(H), Lxy(Lxy), cells(cells), viscosity(viscosity), mode(mode){
	initializeSlitCorrectionMatrix();
	initializeCufft();
	initializeZeroModeBVPSolver();
      }

      Correction(Parameters par): Correction(par.H, par.Lxy, par.cells, par.viscosity, par.mode){ }

      ~Correction(){
	cufftDestroy(cufft_plan_forward);
      }


      void correctSolution(FluidData<complex> &insideSolution,
			   DataXYZ<complex> &gridForces,
			   cudaStream_t st){
	System::log<System::DEBUG>("Computing correction");
	auto analyticalCorrection = computeAnalyticalCorrectionFourierSpace(insideSolution, st);
        takeAnalyticalCorrectionToChebyshevSpace(analyticalCorrection, st);
	const int3 n = cells;
	const int size = (n.x/2+1)*n.y*n.z;
	correctZeroMode(analyticalCorrection, gridForces, st);
        sumCorrectionToInsideSolution(analyticalCorrection,
				      insideSolution, size, st);
      }

    private:

      void chebyshevTransform(complex* d_data, cudaStream_t st){
	const int blockSize = 128;
	const int3 n = cells;
	int nblocks = (((n.x/2+1))*n.y*n.z)/blockSize+1;
	correction_ns::periodicExtension<<<nblocks, blockSize, 0, st>>>(d_data, n.z, (n.x/2+1)*n.y);
	CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
	CufftSafeCall(cufftExecComplex2Complex<real>(cufft_plan_forward, d_data, d_data, CUFFT_FORWARD));
	nblocks = ((n.x/2 + 1)*n.y*n.z)/blockSize + 1;
	correction_ns::scaleFFTToForwardChebyshevTransform<<<nblocks, blockSize, 0, st>>>(d_data, n.z, (n.x/2+1)*n.y);
	CudaCheckError();
      }

      void takeAnalyticalCorrectionToChebyshevSpace(FluidData<complex> &analyticalCorrection, cudaStream_t st){
	System::log<System::DEBUG>("Analytical Correction to Chebyshev");
	chebyshevTransform(analyticalCorrection.pressure.data().get(), st);
	chebyshevTransform(analyticalCorrection.velocity.x(), st);
	chebyshevTransform(analyticalCorrection.velocity.y(), st);
	chebyshevTransform(analyticalCorrection.velocity.z(), st);
      }

      void correctZeroMode(FluidData<complex> &correction, DataXYZ<complex> &gridForces,
			   cudaStream_t st){
	System::log<System::DEBUG>("Correcting zero mode");
	//Only the zero mode BVP is needed for the correction, in this case the second integral matrix is the identity
	//Additionally only the velocities in the plane have to be corrected
	//The equation to solve is y'' = f
	//Furthermore the BCs in this case are that either y(k=0, z=0,H) = 0 (slit channel) or  y'(k=0, z=H) =0 (bottom wall)
	//In both cases alpha and beta are zero, so the system:
	// (C*invA*B - D) (c0; d0) = C*invA*f - (alpha;beta) is simplified to:
	// -D*(c0;d0) = C*f
	//We just need to compute the BCs, solve the 2x2 system above to get c0 and d0 and then use them to integrate y'' twice.
	auto n = cells;
	cached_vector<complex> an(n.z);
	auto an_ptr = thrust::raw_pointer_cast(an.data());
	auto C_ptr  = thrust::raw_pointer_cast(zeroModeCMatrix.data());
	correction_ns::correctZeroMode<<<1,1,0,st>>>(correction.getPointers(),
						     DataXYZPtr<complex>(gridForces),
						     an_ptr, zeroModeDMatrix, C_ptr,
						     n.x, n.y, n.z, 0.5*H, viscosity);
	// correction_ns::cleanSolutionZeroMode<<<1,1,0,st>>>(DataXYZPtr<complex>(gridForces),
	// 						   n.x, n.y, n.z);
      }

    };

    namespace correction_ns{

      template<class Iterator>
      __device__ auto computeBottomSolution(Iterator insideSolution, int nz){
	typename std::iterator_traits<Iterator>::value_type bottomSolution{};
	int sign = 1;
	for(int i=0; i<nz; i++){
	  bottomSolution += insideSolution[i]*sign;
	  sign *= -1;
	}
	return bottomSolution;
      }

      template<class Iterator>
      __device__ auto computeTopSolution(Iterator insideSolution, int nz){
        typename std::iterator_traits<Iterator>::value_type topSolution{};
	for(int i=0; i<nz; i++){
	  topSolution += insideSolution[i];
	}
	return topSolution;
      }

      __global__ void fillRightHandSide(complex* rhs,
					const FluidPointers<complex> insideSolution,
					real H, real2 Lxy, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int2 ik = make_int2(id%(n.x/2+1), id/(n.x/2+1));
	if(id >= n.y*(n.x/2+1)){
	  return;
	}
	//zero mode is not corrected
	if(id==0){
	  return;
	}
	IndexToWaveNumber id2wn(n.x, n.y);
	WaveNumberToWaveVector wn2wv(Lxy);
	const real2 kvec = wn2wv(id2wn(id));
	const real k = sqrt(dot(kvec, kvec));
	auto indexer =  Index3D(n.x/2+1, n.y, n.z);
	auto vx = make_third_index_iterator(insideSolution.velocityX, ik.x, ik.y, indexer);
	auto vy = make_third_index_iterator(insideSolution.velocityY, ik.x, ik.y, indexer);
	auto vz = make_third_index_iterator(insideSolution.velocityZ, ik.x, ik.y, indexer);
	auto u0 = -computeBottomSolution(vx, n.z);
	auto uH = -computeTopSolution(vx, n.z);
	auto v0 = -computeBottomSolution(vy, n.z);
	auto vH = -computeTopSolution(vy, n.z);
	auto w0 = -computeBottomSolution(vz, n.z);
	auto wH = -computeTopSolution(vz, n.z);
	rhs[8*id] = {kvec.x*u0.y + kvec.y*v0.y,
	            -kvec.x*u0.x - kvec.y*v0.x};
	rhs[8*id+1] = {kvec.x*uH.y + kvec.y*vH.y,
	              -kvec.x*uH.x - kvec.y*vH.x};
	rhs[8*id+2] = u0;
	rhs[8*id+3] = v0;
	rhs[8*id+4] = w0;
	rhs[8*id+5] = uH;
	rhs[8*id+6] = vH;
	rhs[8*id+7] = wH;
      }

      __device__ complex complexDot(complex a, complex b ){
	complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
      }

      __device__ void complexmatvecprod(complex* A, complex* B, complex *C, int n){
	for(int i = 0; i<n; i++){
	  complex tmp{};
	  for(int j = 0; j<n; j++){
	    tmp += complexDot(A[n*i+j], B[j]);
	  }
	  C[i] = tmp;
	}
      }

      __global__ void computeAnalyticalCorrectionFourierSlit(FluidPointers<complex> correction,
							     real H, real2 Lxy, real viscosity,
							     complex* invA, complex* b, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int2 ik = make_int2(id%(n.x/2+1), id/(n.x/2+1));
	if(id >= n.y*(n.x/2+1)){
	  return;
	}
	auto indexer =  Index3D(n.x/2+1, n.y, n.z);
	auto pcorr = make_third_index_iterator(correction.pressure, ik.x, ik.y, indexer);
	auto vxcorr = make_third_index_iterator(correction.velocityX, ik.x, ik.y, indexer);
	auto vycorr = make_third_index_iterator(correction.velocityY, ik.x, ik.y, indexer);
	auto vzcorr = make_third_index_iterator(correction.velocityZ, ik.x, ik.y, indexer);
	//zero mode is not corrected, a BVP solve takes care of zero mode
	if(id==0){
	  for(int i = 0; i<n.z; i++){
	    pcorr[i] *= 0;
	    vxcorr[i] *= 0;
	    vycorr[i] *= 0;
	    vzcorr[i] *= 0;
	  }
	  return;
	}
	IndexToWaveNumber id2wn(n.x, n.y);
	WaveNumberToWaveVector wn2wv(Lxy);
	const real2 kvec = wn2wv(id2wn(id));
	const real k = sqrt(dot(kvec, kvec));
	complex C[8]; //Stores [C0, D0, C1, D1, ... C3, D3]
	complexmatvecprod(invA+8*8*id, b+8*id, C, 8);
	const real halfmu = real(0.5)/viscosity;
	for(int i = 0; i<n.z; i++){
	  const real z =  H-(real(-0.5)*H*cospi(i/real(n.z-1))+real(0.5)*H); //from H to 0
	  const real ekzmH = exp(k*(z-H));
	  const real ekz = exp(-k*z);
	  vxcorr[i] =
	    { C[0].y*kvec.x*halfmu/k*z*ekz - C[1].y*kvec.x*halfmu/k*z*ekzmH + C[2].x*ekz + C[3].x*ekzmH,
	     -C[0].x*kvec.x*halfmu/k*z*ekz + C[1].x*kvec.x*halfmu/k*z*ekzmH + C[2].y*ekz + C[3].y*ekzmH};
	  vycorr[i] =
	    { C[0].y*kvec.y*halfmu/k*z*ekz - C[1].y*kvec.y*halfmu/k*z*ekzmH + C[4].x*ekz + C[5].x*ekzmH,
	     -C[0].x*kvec.y*halfmu/k*z*ekz + C[1].x*kvec.y*halfmu/k*z*ekzmH + C[4].y*ekz + C[5].y*ekzmH};
	  vzcorr[i] = C[0]*halfmu*z*ekz + C[1]*halfmu*z*ekzmH + C[6]*ekz + C[7]*ekzmH;
	  pcorr[i] = C[0]*z*ekz + C[1]*ekzmH;
	}
      }

      __global__ void computeAnalyticalCorrectionFourierBottomWall(FluidPointers<complex> correction,
								   const FluidPointers<complex> insideSolution,
								   real H, real2 Lxy, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int2 ik = make_int2(id%(n.x/2+1), id/(n.x/2+1));
	if(id >= n.y*(n.x/2+1)){
	  return;
	}
	auto indexer =  Index3D(n.x/2+1, n.y, n.z);
	auto pcorr = make_third_index_iterator(correction.pressure, ik.x, ik.y, indexer);
	auto vxcorr = make_third_index_iterator(correction.velocityX, ik.x, ik.y, indexer);
	auto vycorr = make_third_index_iterator(correction.velocityY, ik.x, ik.y, indexer);
	auto vzcorr = make_third_index_iterator(correction.velocityZ, ik.x, ik.y, indexer);
	//zero mode is not corrected, a BVP solve takes care of zero mode
	if(id==0){
	  for(int i = 0; i<n.z; i++){
	    pcorr[i] *= 0;
	    vxcorr[i] *= 0;
	    vycorr[i] *= 0;
	    vzcorr[i] *= 0;
	  }
	  return;
	}
	IndexToWaveNumber id2wn(n.x, n.y);
	WaveNumberToWaveVector wn2wv(Lxy);
	const real2 kvec = wn2wv(id2wn(id));
	const real k = sqrt(dot(kvec, kvec));
	auto vx = make_third_index_iterator(insideSolution.velocityX, ik.x, ik.y, indexer);
	auto vy = make_third_index_iterator(insideSolution.velocityY, ik.x, ik.y, indexer);
	auto vz = make_third_index_iterator(insideSolution.velocityZ, ik.x, ik.y, indexer);
	auto u0 = -computeBottomSolution(vx, n.z);
	auto v0 = -computeBottomSolution(vy, n.z);
	auto w0 = -computeBottomSolution(vz, n.z);
	for(int i = 0; i<n.z; i++){
	  const real z = H-((real(-0.5)*H)*cospi(i/real(n.z-1))+real(0.5)*H); //from H to 0
	  const real ekz = exp(-k*z);
	  vxcorr[i] = { -kvec.x/k*(-k*w0.y + kvec.x*u0.x + kvec.y*v0.x)*z*ekz + u0.x*ekz,
			-kvec.x/k*(k*w0.x + kvec.x*u0.y + kvec.y*v0.y)*z*ekz + u0.y*ekz};
	  vycorr[i] = { -kvec.y/k*(-k*w0.y + kvec.x*u0.x + kvec.y*v0.x)*z*ekz + v0.x*ekz,
			-kvec.y/k*(k*w0.x + kvec.x*u0.y + kvec.y*v0.y)*z*ekz + v0.y*ekz};
	  vzcorr[i] = { (k*w0.x + kvec.x*u0.y + kvec.y*v0.y)*z*ekz + w0.x*ekz,
			(k*w0.y - kvec.x*u0.x - kvec.y*v0.x)*z*ekz + w0.y*ekz};
	  pcorr[i] = complex();
	}
      }

    }

    FluidData<complex> Correction::computeAnalyticalCorrectionFourierSpace(const FluidData<complex> &insideSolution,
									   cudaStream_t st){
      int3 n = cells;
      int nk = (n.x/2+1)*n.y;
      FluidData<complex> analyticalCorrection({n.x/2+1, n.y, (2*n.z-2)});
      int blockSize = 128;
      int nblocks = ((n.x/2+1)*n.y)/blockSize + 1;
      if(mode == WallMode::slit){
	cached_vector<complex> rhs(nk*8);
	auto d_rhs = thrust::raw_pointer_cast(rhs.data());
	correction_ns::fillRightHandSide<<<nblocks, blockSize, 0, st>>>(d_rhs,
									insideSolution.getPointers(),
									H, Lxy, n);
	auto d_invA = thrust::raw_pointer_cast(invA.data());
	correction_ns::computeAnalyticalCorrectionFourierSlit<<<nblocks, blockSize, 0, st>>>(
				 analyticalCorrection.getPointers(),
				 H, Lxy, viscosity, d_invA, d_rhs,  n);
      }
      else if(mode == WallMode::bottom){
        correction_ns::computeAnalyticalCorrectionFourierBottomWall<<<nblocks, blockSize, 0, st>>>(
            analyticalCorrection.getPointers(), insideSolution.getPointers(),
	    H, Lxy, n);
      }
      else{
	analyticalCorrection.velocity.fillWithZero();
	thrust::fill(analyticalCorrection.pressure.begin(),
		     analyticalCorrection.pressure.end(),
		     complex());
      }
      return analyticalCorrection;
    }

    std::vector<complex> evaluateSlitCorrectionMatrix(real2 kvec, real viscosity, real H){
      const complex o = {1,0};
      const complex i = {0,1};
      const complex z = {0,0};
      const real k = sqrt(dot(kvec, kvec));
      const real halfmu = 0.5/viscosity;
      const real ekH = exp(-k*H);
      const std::vector<complex> A
	{halfmu*o,                 ekH*halfmu*o,             z,     z,     z,     z,     -k*o,     k*ekH*o,
	 (1.0-k*H)*ekH*halfmu*o,   (1.0+k*H)*halfmu*o,       z,     z,     z,     z,     -k*ekH*o, k*o,
	 z,                        z,                        o,     ekH*o, z,     z,     z,        z,
	 z,                        z,                        z,     z,     o,     ekH*o, z,        z,
	 z,                        z,                        z,     z,     z,     z,     o,        ekH*o,
	 -i*kvec.x*H*ekH*halfmu/k, i*kvec.x*H*halfmu/k,      ekH*o, o,     z,     z,     z,        z,
	 -i*kvec.y*H*ekH*halfmu/k, i*kvec.y*H*halfmu/k,      z,     z,     ekH*o, o,     z,        z,
	 H*ekH*halfmu*o,           H*halfmu*o,               z,     z,     z,     z,     ekH*o,    o};
      return A;
    }

    void Correction::initializeSlitCorrectionMatrix(){
      //No need to solve a system in non-slit mode
      if(mode != WallMode::slit){
	return;
      }
      IndexToWaveVector i2wv(cells.x, cells.y, Lxy);
      int3 n = cells;
      int nk = (n.x/2+1)*n.y;
      std::vector<complex> h_invA(8*8*nk, complex());
      auto it = thrust::make_counting_iterator<int>(0);
      std::for_each(it+1, it + nk, //The zero mode is excluded
		      [&](int i){
			real2 kvec = i2wv(i);
			auto A = evaluateSlitCorrectionMatrix(kvec, viscosity, H);
			auto h_invA_i = BVP::invertSquareMatrix(A, 8);
			std::copy(h_invA_i.begin(), h_invA_i.end(), h_invA.begin()+8*8*i);
		      }
		      );
      this->invA = h_invA;
    }

    void Correction::initializeCufft(){
      const int3 n = cells;
      int size = 2*n.z-2;
      int stride = (n.x/2+1)*n.y;
      int dist = 1;
      int batch = (n.x/2+1)*n.y;
      CufftSafeCall(cufftPlanMany(&cufft_plan_forward, 1, &size, &size,
				  stride, dist, &size, stride,
				  dist, CUFFT_Complex2Complex<real>::value, batch));
    }

    auto computeZeroModeBoundaryConditions(int nz, real H, WallMode mode){
      BVP::SchurBoundaryConditionReal bcs(nz, H);
      if(mode == WallMode::bottom){
	correction_ns::TopBoundaryConditions top(0, H);
	correction_ns::BottomBoundaryConditions bot(0, H);
	return bcs.computeBoundaryConditionMatrix(top, bot);
      }
      else if(mode == WallMode::slit){
	correction_ns::SlitBoundaryConditions top(0, H);
	auto bot = top;
	return bcs.computeBoundaryConditionMatrix(top, bot);
      }
      else{
	System::log<System::ERROR>("Invalid wall mode");
	throw std::runtime_error("Invalid wall mode for correction");
      }
    }

    void Correction::initializeZeroModeBVPSolver(){
      if(mode != WallMode::none){
	auto n = cells;
	auto CandD = computeZeroModeBoundaryConditions(n.z, 0.5*H, mode);
	zeroModeCMatrix = gpu_container<real>(CandD.begin(), CandD.begin() + 2*n.z);
	zeroModeDMatrix = {CandD[2*n.z], CandD[2*n.z+1], CandD[2*n.z+2], CandD[2*n.z+3]};
	CudaCheckError();
      }
    }
  }
}
