/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver. Correction

 */
#include"utils.cuh"
#include"misc/BoundaryValueProblem/MatrixUtils.h"
namespace uammd{
  namespace DPStokesSlab_ns{

    namespace detail{
      __global__ void periodicExtension(cufftComplex4* signal, int nz, int nbatch){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= nbatch*nz) return;
	const int ib = id%nbatch;
	const int iz  = id/(nbatch);
	if(iz>=nz-1 or iz == 0) return;
	const int zf = 2*nz-2-iz;
	const int zi = iz;
	signal[ib + zf*nbatch]  = signal[ib + zi*nbatch];
      }

      __global__ void scaleFFTToForwardChebyshevTransform(cufftComplex4* signal, int nz, int nbatch){
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
    }

    template<class Container>
    void sumCorrectionToInsideSolution(Container &correction,
				       Container &insideSolution,
				       int size,
				       cudaStream_t  st){
      System::log<System::DEBUG>("Sum correction to solution");
      thrust::transform(thrust::cuda::par.on(st),
			insideSolution.begin(), insideSolution.begin()+size,
			correction.begin(), insideSolution.begin(),
			thrust::plus<cufftComplex4>());
    }

    class Correction{
      real H;
      real2 Lxy;
      int3 cells;
      real viscosity;
      gpu_container<cufftComplex> invA;
      cufftHandle cufft_plan_forward;

      void initializeCorrectionMatrix();
      void initializeCufft();
      cached_vector<cufftComplex4> computeAnalyticalCorrectionFourierSpace(const cufftComplex4 *insideSolution, cudaStream_t st);

    public:
      struct Parameters{
	real H;
	real2 Lxy;
	int3 cells;
	real viscosity;
      };

      Correction(real H, real2 Lxy, int3 cells, real viscosity):
      	H(H), Lxy(Lxy), cells(cells), viscosity(viscosity){
	initializeCorrectionMatrix();
	initializeCufft();
      }

      Correction(Parameters par): Correction(par.H, par.Lxy, par.cells, par.viscosity){ }

      ~Correction(){
	cufftDestroy(cufft_plan_forward);
      }

      template<class Container>
      void correctSolution(Container &insideSolution, cudaStream_t st){
	const cufftComplex4* d_insideSolution = thrust::raw_pointer_cast(insideSolution.data());
	auto analyticalCorrection = computeAnalyticalCorrectionFourierSpace(d_insideSolution, st);
        takeAnalyticalCorrectionToChebyshevSpace(analyticalCorrection, st);
	const int3 n = cells;
	const int size = (n.x/2+1)*n.y*n.z;
	sumCorrectionToInsideSolution(analyticalCorrection, insideSolution, size, st);
      }

    private:
      void takeAnalyticalCorrectionToChebyshevSpace(cached_vector<cufftComplex4> &analyticalCorrection, cudaStream_t st){
	System::log<System::DEBUG>("Analytical Correction to Chebyshev");
	cufftComplex* d_data = (cufftComplex*) thrust::raw_pointer_cast(analyticalCorrection.data());
	const int blockSize = 128;
	const int3 n = cells;
	int nblocks = (((n.x/2+1))*n.y*n.z)/blockSize+1;
	detail::periodicExtension<<<nblocks, blockSize, 0, st>>>((cufftComplex4*)d_data, n.z, (n.x/2+1)*n.y);
	CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
	CufftSafeCall(cufftExecComplex2Complex<real>(cufft_plan_forward, d_data, d_data, CUFFT_FORWARD));
	nblocks = ((n.x/2 + 1)*n.y*n.z)/blockSize + 1;
	detail::scaleFFTToForwardChebyshevTransform<<<nblocks, blockSize, 0, st>>>((cufftComplex4*)d_data, n.z, (n.x/2+1)*n.y);
	CudaCheckError();
      }

    };

    namespace detail{

      template<class Iterator>
      __device__ cufftComplex4 computeBottomSolution(Iterator insideSolution, int nz){
	cufftComplex4 bottomSolution{};
	for(int i=0; i<nz; i++){
	  bottomSolution += insideSolution[i]*pow(-1,i);
	}
	return bottomSolution;
      }

      template<class Iterator>
      __device__ cufftComplex4 computeTopSolution(Iterator insideSolution, int nz){
	cufftComplex4 topSolution{};
	for(int i=0; i<nz; i++){
	  topSolution += insideSolution[i];
	}
	return topSolution;
      }

      __global__ void fillRightHandSide(cufftComplex* rhs, const cufftComplex4* insideSolution, real H, real2 Lxy, int3 n){
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
	auto sol = make_third_index_iterator(insideSolution, ik.x, ik.y, Index3D(n.x/2+1, n.y, n.z));
	auto solutionBottom = computeBottomSolution(sol, n.z);
	auto solutionTop = computeTopSolution(sol, n.z);
	auto u0 = -solutionBottom.x;
	auto v0 = -solutionBottom.y;
	auto w0 = -solutionBottom.z;
	auto uH = -solutionTop.x;
	auto vH = -solutionTop.y;
	auto wH = -solutionTop.z;
	rhs[8*id] = {kvec.x*u0.y + kvec.y*v0.y, -kvec.x*u0.x - kvec.y*v0.x};
	rhs[8*id+1] = {kvec.x*uH.y + kvec.y*vH.y, -kvec.x*uH.x - kvec.y*vH.x};
	rhs[8*id+2] = u0;
	rhs[8*id+3] = v0;
	rhs[8*id+4] = w0;
	rhs[8*id+5] = uH;
	rhs[8*id+6] = vH;
	rhs[8*id+7] = wH;
      }

      __device__ cufftComplex complexDot(cufftComplex a, cufftComplex b ){
	cufftComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
      }

      __device__ void complexmatvecprod(cufftComplex* A, cufftComplex* B, cufftComplex *C, int n){
	for(int i = 0; i<n; i++){
	  cufftComplex tmp{};
	  for(int j = 0; j<n; j++){
	    tmp += complexDot(A[n*i+j], B[j]);
	  }
	  C[i] = tmp;
	}
      }

      __global__ void computeAnalyticalCorrectionFourierSpace(cufftComplex4* d_correction, const cufftComplex4* insideSolution,
							      real H, real2 Lxy, real viscosity, cufftComplex* invA, cufftComplex* b, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int2 ik = make_int2(id%(n.x/2+1), id/(n.x/2+1));
	if(id >= n.y*(n.x/2+1)){
	  return;
	}
	auto corr = make_third_index_iterator(d_correction, ik.x, ik.y, Index3D(n.x/2+1, n.y, n.z));
	//zero mode is not corrected, BVP solve takes care of zero mode
	if(id==0){
	  for(int i = 0; i<n.z; i++){
	    corr[i] = cufftComplex4();
	  }
	  return;
	}
	IndexToWaveNumber id2wn(n.x, n.y);
	WaveNumberToWaveVector wn2wv(Lxy);
	const real2 kvec = wn2wv(id2wn(id));
	const real k = sqrt(dot(kvec, kvec));
	cufftComplex C[8]; //Stores [C0, D0, C1, D1, ... C3, D3]
	complexmatvecprod(invA+8*8*id, b+8*id, C, 8);
	const real halfmu = real(0.5)/viscosity;
	for(int i = 0; i<n.z; i++){
	  const real z =  (real(0.5)*H)*cospi(i/real(n.z-1))+real(0.5)*H; //from 0 to H
	  const real ekzmH = exp(k*(z-H));
	  const real ekz = exp(-k*z);
	  const cufftComplex ucorr =
	    { C[0].y*kvec.x*halfmu/k*z*ekz - C[1].y*kvec.x*halfmu/k*z*ekzmH + C[2].x*ekz + C[3].x*ekzmH,
	     -C[0].x*kvec.x*halfmu/k*z*ekz + C[1].x*kvec.x*halfmu/k*z*ekzmH + C[2].y*ekz + C[3].y*ekzmH};
	  const cufftComplex vcorr =
	    { C[0].y*kvec.y*halfmu/k*z*ekz - C[1].y*kvec.y*halfmu/k*z*ekzmH + C[4].x*ekz + C[5].x*ekzmH,
	     -C[0].x*kvec.y*halfmu/k*z*ekz + C[1].x*kvec.y*halfmu/k*z*ekzmH + C[4].y*ekz + C[5].y*ekzmH};
	  const cufftComplex wcorr = C[0]*halfmu*z*ekz + C[1]*halfmu*z*ekzmH + C[6]*ekz + C[7]*ekzmH;
	  const cufftComplex pcorr = C[0]*z*ekz + C[1]*ekzmH;
	  corr[i] = {ucorr, vcorr, wcorr, pcorr};
	}
      }
    }

    cached_vector<cufftComplex4> Correction::computeAnalyticalCorrectionFourierSpace(const cufftComplex4 *insideSolution,
										     cudaStream_t st){
      IndexToWaveVector i2wv(cells.x, cells.y, Lxy);
      int3 n = cells;
      int nk = (n.x/2+1)*n.y;
      cached_vector<cufftComplex4> analyticalCorrection(nk*(2*n.z-2));
      int blockSize = 128;
      int nblocks = ((n.x/2+1)*n.y)/blockSize + 1;
      auto d_corr = thrust::raw_pointer_cast(analyticalCorrection.data());
      auto d_invA = thrust::raw_pointer_cast(invA.data());
      cached_vector<cufftComplex> rhs(nk*8);
      auto d_rhs = thrust::raw_pointer_cast(rhs.data());
      detail::fillRightHandSide<<<nblocks, blockSize, 0, st>>>(d_rhs, insideSolution, H, Lxy, n);
      detail::computeAnalyticalCorrectionFourierSpace<<<nblocks, blockSize, 0, st>>>(d_corr, insideSolution,
										     H, Lxy, viscosity, d_invA, d_rhs,  n);
      return analyticalCorrection;
    }

    std::vector<cufftComplex> evaluateCorrectionMatrix(real2 kvec, real viscosity, real H){
      const cufftComplex o = {1,0};
      const cufftComplex i = {0,1};
      const cufftComplex z = {0,0};
      const real k = sqrt(dot(kvec, kvec));
      const real halfmu = 0.5/viscosity;
      const real ekH = exp(-k*H);
      const std::vector<cufftComplex> A
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

    void Correction::initializeCorrectionMatrix(){
      IndexToWaveVector i2wv(cells.x, cells.y, Lxy);
      int3 n = cells;
      int nk = (n.x/2+1)*n.y;
      std::vector<cufftComplex> h_invA(8*8*nk, cufftComplex());
      auto it = thrust::make_counting_iterator<int>(0);
      std::for_each(it+1, it + nk, //The zero mode is excluded
		      [&](int i){
			real2 kvec = i2wv(i);
			auto A = evaluateCorrectionMatrix(kvec, viscosity, H);
			auto h_invA_i = BVP::invertSquareMatrix(A, 8);
			std::copy(h_invA_i.begin(), h_invA_i.end(), h_invA.begin()+8*8*i);
		      }
		      );
      this->invA = h_invA;
    }

    void Correction::initializeCufft(){
      const int3 n = cells;
      int size = 2*n.z-2;
      int stride = 4*(n.x/2+1)*n.y;
      int dist = 1;
      int batch = 4*(n.x/2+1)*n.y;
      CufftSafeCall(cufftPlanMany(&cufft_plan_forward, 1, &size, &size,
				  stride, dist, &size, stride,
				  dist, CUFFT_Complex2Complex<real>::value, batch));
    }
  }
}
