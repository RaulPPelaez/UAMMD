/*Raul P. Pelaez 2022. Utilities to compute structure factors using the GPU.

 */
#include"uammd.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include"utils/container.h"
namespace uammd{


  //A functor to compute a fourier correlation, computes a·b^*
  struct Convolution{
    real nomalization;
    Convolution(real normalization): nomalization(normalization){}

    template<class complex>
    __device__ complex operator()(complex a, complex b){
      return {(a.x*b.x + a.y*b.y)/nomalization, (a.y*b.x - a.x*b.y)/nomalization};
    }
  };

  //Computes static structure factors in Fourier space.
  class StaticStructureFactor3D{
    cufftHandle plan;
    int3 ncells;
  public:
    using complex = uammd::cufftComplex_t<real>;
    template<class T>
    using cached_container = uninitialized_cached_vector<T>;
    StaticStructureFactor3D(int3 ncells):ncells(ncells){
      cufftPlan3d(&plan, ncells.x, ncells.y, ncells.z, CUFFT_Real2Complex<real>::value);
    }

    //Computes S(k) = <a_k·b_k^*> given a(r) and b(r)
    template<class RealIterator>
    auto compute(RealIterator a, RealIterator b){
      int n = ncells.x*ncells.y*ncells.z;
      int nhat = (ncells.x/2+1)*ncells.y*ncells.z;
      cached_container<complex> a_hat(nhat);
      cached_container<complex> b_hat(nhat);
      cached_container<complex> structureFactor(nhat);
      cufftExecReal2Complex<real>(plan, a, thrust::raw_pointer_cast(a_hat.data()));
      cufftExecReal2Complex<real>(plan, b, thrust::raw_pointer_cast(b_hat.data()));
       auto cit = thrust::make_counting_iterator(0);
       thrust::transform(a_hat.begin(), a_hat.end(),
       			b_hat.begin(),
       			structureFactor.begin(),
       			Convolution(n));
      auto it = thrust::make_constant_iterator(complex{0,0});
      thrust::copy(it, it+1, structureFactor.begin());
      return structureFactor;
    }

  };


  //This functor is used to permute a matrix
  struct ColMajorToRowMajor{
    int ntimes, time;
    ColMajorToRowMajor(int nt, int t):ntimes(nt), time(t){}

    __device__ int operator()(int i){
      return ntimes*i+time;
    }

  };

  //Given a series of samples in Fourier space, computes their correlation.
  class DynamicStructureFactor{
  public:
    template<class T>
    using cached_container = uninitialized_cached_vector<T>;
    using complex = uammd::cufftComplex_t<real>;
  private:
    cufftHandle plan;
    int nsamples, ntimes;
    cached_container<complex> uqt, vqt;

  public:

    DynamicStructureFactor(int nsamples, int ntimes):nsamples(nsamples), ntimes(ntimes){
      uqt.resize(nsamples*ntimes);
      vqt.resize(nsamples*ntimes);
      int size = ntimes;
      int n[] = {size};
      int istride = 1;
      int ostride = 1;
      int idist = size;
      int odist = size;
      int inembed[] = {0};
      int onembed[] = {0};
      cufftPlanMany(&plan, 1,n,
		    inembed, istride, idist,
		    onembed, ostride, odist,
		    CUFFT_Complex2Complex<real>::value,
		    nsamples);
    }

    //Add samples in Fourier space. Simply stores the provided data.
    //Expects two vectors of size nsamples
    template<class ComplexIterator>
    void addSamplesFourier(ComplexIterator samples_u, ComplexIterator samples_v,  int time){
      auto cit = thrust::make_counting_iterator(0);
      auto indexit = thrust::make_transform_iterator(cit, ColMajorToRowMajor(ntimes, time));
      auto perm = thrust::make_permutation_iterator(uqt.begin(), indexit);
      thrust::copy(samples_u, samples_u+nsamples, perm);
      perm = thrust::make_permutation_iterator(vqt.begin(), indexit);
      thrust::copy(samples_v, samples_v+nsamples, perm);
    }

    //Compute the correlation between the stored samples.
    auto compute(){
      cached_container<complex> uqw(ntimes*nsamples);
      cached_container<complex> vqw(ntimes*nsamples);
      cufftExecComplex2Complex<real>(plan,
				     thrust::raw_pointer_cast(uqt.data()),
				     thrust::raw_pointer_cast(uqw.data()),
				     CUFFT_FORWARD);
      cufftExecComplex2Complex<real>(plan,
				     thrust::raw_pointer_cast(vqt.data()),
				     thrust::raw_pointer_cast(vqw.data()),
				     CUFFT_FORWARD);
      thrust::transform(uqw.begin(), uqw.end(),
			vqw.begin(),
			uqw.begin(),
			Convolution(ntimes));
      return uqw;

    }
  };

  //Computes real to complex 3D Fourier transforms.
  class FourierTransform3D{
    cufftHandle plan;
    int3 ncells;
  public:
    using complex = uammd::cufftComplex_t<real>;
    template<class T>
    using cached_container = uninitialized_cached_vector<T>;
    FourierTransform3D(int3 ncells):ncells(ncells){
      cufftPlan3d(&plan, ncells.x, ncells.y, ncells.z, CUFFT_Real2Complex<real>::value);
    }

    template<class RealIterator>
    auto transform(RealIterator a){
      int nhat = (ncells.x/2+1)*ncells.y*ncells.z;
      cached_container<complex> a_hat(nhat);
      cufftExecReal2Complex<real>(plan, a, thrust::raw_pointer_cast(a_hat.data()));
      {
	auto it = thrust::make_constant_iterator(complex{0,0});
	thrust::copy(it, it+1, a_hat.begin());
      }
      {
       	int ntot = ncells.x*ncells.y*ncells.z;
       	auto it = thrust::make_constant_iterator<complex>({real(ntot), real(ntot)});
       	thrust::transform(a_hat.begin(), a_hat.end(), it, a_hat.begin(), thrust::divides<complex>());
      }
      return a_hat;
    }

  };

  //Given a series of samples in Fourier space, computes their FFT.
  class FourierTransformComplex1D{
  public:
    template<class T>
    using cached_container = uninitialized_cached_vector<T>;
    using complex = uammd::cufftComplex_t<real>;
  private:
    cufftHandle plan;
    int nsamples, ntimes;
    cached_container<complex> uqt;

  public:

    FourierTransformComplex1D(int nsamples, int ntimes):nsamples(nsamples), ntimes(ntimes){
      uqt.resize(nsamples*ntimes);
      int size = ntimes;
      int n[] = {size};
      int istride = 1;
      int ostride = 1;
      int idist = size;
      int odist = size;
      int inembed[] = {0};
      int onembed[] = {0};
      cufftPlanMany(&plan, 1,n,
		    inembed, istride, idist,
		    onembed, ostride, odist,
		    CUFFT_Complex2Complex<real>::value,
		    nsamples);
    }

    //Add samples in Fourier space. Simply stores the provided data.
    //Expects two vectors of size nsamples
    template<class ComplexIterator>
    void addSamplesFourier(ComplexIterator samples_u, int time){
      auto cit = thrust::make_counting_iterator(0);
      auto indexit = thrust::make_transform_iterator(cit, ColMajorToRowMajor(ntimes, time));
      auto perm = thrust::make_permutation_iterator(uqt.begin(), indexit);
      thrust::copy(samples_u, samples_u+nsamples, perm);
    }

    //Compute the fft for the stored samples.
    auto compute(){
      cached_container<complex> uqw(ntimes*nsamples);
      cufftExecComplex2Complex<real>(plan,
				     thrust::raw_pointer_cast(uqt.data()),
				     thrust::raw_pointer_cast(uqw.data()),
				     CUFFT_FORWARD);
      auto it = thrust::make_constant_iterator<complex>({real(ntimes), real(ntimes)});
      thrust::transform(uqw.begin(), uqw.end(), it, uqw.begin(), thrust::divides<complex>());
      return uqw;
    }
  };
}
