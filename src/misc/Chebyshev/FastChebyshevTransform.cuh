/*Raul P. Pelaez 2022. Utility functions to transform signals between real/complex space and Chebyshev or Fourier-Chebyshev space
 */
#ifndef CHEBYSHEVUTILSFASTCHEBYSHEVTRANSFORM_CUH
#define CHEBYSHEVUTILSFASTCHEBYSHEVTRANSFORM_CUH
#include <uammd.cuh>
#include "utils/cufftDebug.h"
#include"utils/cufftPrecisionAgnostic.h"
#include "misc/ChevyshevUtils.cuh"
#include "utils/complex.cuh"

namespace uammd{
  namespace chebyshev{
    namespace detail{
      struct BatchedIteratorTransform{
	int id, offset;
	__host__ __device__ BatchedIteratorTransform(int id, int offset):id(id), offset(offset){}

	inline __host__ __device__ int operator()(int z) const{
	  return id + offset*z;
	}
      };

    }
    //Given an iterator to a segment storing a group of interleaved signals where
    // element k of signal id is stored at id+k*offset, returns a permutation iterator
    // in which elements k of signal id are contiguous.
    template<class RandomAccessIterator>
    inline __host__ __device__
    auto make_interleaved_iterator(RandomAccessIterator ptr,
				   int id,
				   int offset){
      auto tr = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
						detail::BatchedIteratorTransform(id, offset));
      return thrust::make_permutation_iterator(ptr, tr);
    }

    namespace detail{
      //A functor to scale the result of a FFT periodic-extended transformation to Chebyshev coefficients
      struct FFTToChebyshev{
	int3 n;
	real normalization;
	FFTToChebyshev(int3 n, real normalization):
	  n(n), normalization(normalization){}

	//Scales the element stored at id
	//Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
	template<class T>
	__device__ auto operator()(T v, int id){
	  int k = id/(n.x*n.y);
	  auto pm = (k==0 or k==(n.z-1))?real(1.0):real(2.0);
	  return v*(pm)/((2*n.z-real(2.0))*normalization);
	}

      };

      //Scales the result of a 3D FFT periodic-extended transformation to Chebyshev coefficients.
      //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
      //Returns a new vector, leaving the input unmodified.
      template<class Container>
      auto scaleFFTToChebyshev(Container v, int3 n, real normalization){
	auto cit = thrust::make_counting_iterator(0);
	thrust::transform(v.begin(), v.end(), cit, v.begin(), FFTToChebyshev(n, normalization));
	return v;
      }

      //A functor to scale the Chebyshev coefficients to the coefficients of a cosine transform prior to periodic extension.
      struct ChebyshevToiFFT{
	int3 n;
	ChebyshevToiFFT(int3 n):n(n){}

	//Scales the element stored at id
	//Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
	template<class T>
	__device__ auto operator()(T v, int id){
	  int k = id/(n.x*n.y);
	  auto pm = (k==0 or k==(n.z-1))?real(1.0):real(2.0);
	  return v/(pm);
	}

      };

      //Scales the Chebyshev coefficients to the coefficients of a cosine transform prior to periodic extension.
      //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
      //Returns a new vector, leaving the input unmodified.
      template<class Container>
      auto scaleChebyshevToiFFT(Container v, int3 n){
	auto cit = thrust::make_counting_iterator(0);
	thrust::transform(v.begin(), v.end(), cit, v.begin(), ChebyshevToiFFT(n));
	return v;
      }

      //Periodic extends a 3D signal in the z direction
      //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
      template<class Iterator>
      __global__ void periodicExtendD(Iterator v, int nk, int nz){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= nk) return;
	auto v_k = make_interleaved_iterator(v, id, nk);
	for(int z = nz; z<2*nz-2; z++){
	  v_k[z] = v_k[2*nz-2-z];
	}
      }

      //Periodic extends a 3D signal in the z direction
      //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
      //Returns a new vector, leaving the input unmodified.
      template<class Container>
      auto periodicExtend(Container v, int nk, int nz){
	v.resize(nk*(2*nz-2));
	int nthreads = 128;
	int nblocks = nk/nthreads + 1;
	periodicExtendD<<<nblocks, nthreads>>>(v.data().get(), nk, nz);
	return v;
      }

      //Creates a cufftPlan with rank 3 and the dimensions provided by the argument.
      //Assumes the output and input have sizes n.x*n.y*(2*(n.z-1))
      //Assumes the element (i,j,k) is located at element i+n.x*(j+n.y*k)
      auto createCufftPlanFourierChebyshev(int3 n){
	cufftHandle cufft_plan;
	int3 cdtmp = {2*(n.z-1), n.y, n.x};
	int3 inembed = {2*(n.z-1), n.y, n.x};
	int3 oembed = {2*(n.z-1), n.y, n.x};
	CufftSafeCall(cufftPlanMany(&cufft_plan,
				    3, &cdtmp.x,
				    &inembed.x,
				    1, 1,
				    &oembed.x,
				    1, 1,
				    uammd::CUFFT_Complex2Complex<real>::value, 1));
	return cufft_plan;
      }

      //Creates a cufftPlan with rank 1 of size 2*(n.z-1) and with n.x*n.y batches.
      //Assumes the output and input have sizes n.x*n.y*(2*(n.z-1)).
      //Assumes the element (i,j,k) is located at element i+n.x*(j+n.y*k).
      auto createCufftPlanChebyshev(int3 n){
	cufftHandle cufft_plan;
	int size = 2*n.z-2;
	int stride = n.x*n.y;
	int dist = 1;
	int batch = n.x*n.y;
	CufftSafeCall(cufftPlanMany(&cufft_plan, 1, &size, &size,
				    stride, dist, &size, stride,
				    dist, uammd::CUFFT_Complex2Complex<real>::value, batch));
	return cufft_plan;
      }

    }

    //From a 3D field of complex values with Z values evaluated at Chebyshev roots (z=cos(pi*k/(n.z-1))
    // returns the Chebyshev coefficients in Z for each wavenumber of the input in Fourier space in the plane directions.
    //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
    //Returns a new vector, leaving the input unmodified.
    template<class Container>
    auto fourierChebyshevTransform3DCufft(Container i_fx, int3 n){
      auto fx = detail::periodicExtend(i_fx, n.x*n.y, n.z);
      auto plan = detail::createCufftPlanFourierChebyshev(n);
      Container idata(fx);
      Container data(fx.size());
      auto d_idata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(idata.data()));
      auto d_odata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(data.data()));
      CufftSafeCall(uammd::cufftExecComplex2Complex<real>(plan, d_idata, d_odata, CUFFT_FORWARD));
      data = detail::scaleFFTToChebyshev(data, n, n.x*n.y);
      CufftSafeCall(cufftDestroy(plan));
      return data;
    }

    //From a 1D field of complex values evaluated at Chebyshev roots (z=cos(pi*k/(n.z-1))
    // returns their Chebyshev coefficients.
    //Returns a new vector, leaving the input unmodified.
    template<class Container>
    auto chebyshevTransform1DCufft(Container fx){
      int nz = fx.size();
      return fourierChebyshevTransform3DCufft(fx, {1,1,nz});
    }

    //From the complex Chebyshev coeffients of a series of signals (each signal assigned to a 2D wave number in Fourier space)
    // returns the (complex valued) inverse transform in the plane directions and the values of the signal in Z
    // evaluated at the Chebyshev roots (z=cos(pi*k/(n.z-1))
    //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k), being i,j wavenumbers and k Chebyshev coeffients
    //Returns a new vector, leaving the input unmodified.
    template<class Container>
    auto inverseFourierChebyshevTransform3DCufft(Container fn, int3 n){
      using uammd::System;
      fn = detail::scaleChebyshevToiFFT(fn, n);
      fn = detail::periodicExtend(fn, n.x*n.y, n.z);
      auto plan = detail::createCufftPlanFourierChebyshev(n);
      Container idata(fn);
      Container data(fn.size());
      auto d_idata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(idata.data()));
      auto d_odata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(data.data()));
      CufftSafeCall(uammd::cufftExecComplex2Complex<real>(plan, d_idata, d_odata, CUFFT_INVERSE));
      CufftSafeCall(cufftDestroy(plan));
      return data;
    }

    //From a group of complex Chebyshev coefficients returns the corresponding signal evaluated at the
    // Chebyshev roots (z=cos(pi*k/(n.z-1))
    //Returns a new vector, leaving the input unmodified.
    template<class Container>
    auto inverseChebyshevTransform1DCufft(Container fn, int nz){
      return inverseFourierChebyshevTransform3DCufft(fn, {1, 1, nz});
    }

    //From a group of n.x*n.y batched complex valued signals, each of them evaluated at the Chevyshev roots (z=cos(pi*k/(n.z-1)),
    // returns the Chebyshev coefficients for each signal.
    //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k), being i,j different signals and k z elements
    //Returns a new vector, leaving the input unmodified.
    template<class Container>
    auto chebyshevTransform3DCufft(Container fx, int3 n){
      fx = detail::periodicExtend(fx, n.x*n.y, n.z);
      auto plan = detail::createCufftPlanChebyshev(n);
      Container idata(fx);
      Container data(fx.size());
      auto d_idata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(idata.data()));
      auto d_odata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(data.data()));
      CufftSafeCall(uammd::cufftExecComplex2Complex<real>(plan, d_idata, d_odata, CUFFT_FORWARD));
      data = detail::scaleFFTToChebyshev(data, n, 1);
      CufftSafeCall(cufftDestroy(plan));
      return data;
    }

    //From a group of n.x*n.y batched signals containing complex Chebyshev coefficients, returns, for
    // each signal, the inverse Chebyshev transform (the values of the function evaluated at the Chebyshev roots)
    //Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k), being i,j different signals and k Chebyshev coefficients
    //Returns a new vector, leaving the input unmodified.
    template<class Container>
    auto inverseChebyshevTransform3DCufft(Container fn, int3 n){
      using uammd::System;
      fn = detail::scaleChebyshevToiFFT(fn, n);
      fn = detail::periodicExtend(fn, n.x*n.y, n.z);
      auto plan = detail::createCufftPlanChebyshev(n);
      Container idata(fn);
      Container data(fn.size());
      auto d_idata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(idata.data()));
      auto d_odata = (uammd::cufftComplex_t<real>*)(thrust::raw_pointer_cast(data.data()));
      CufftSafeCall(uammd::cufftExecComplex2Complex<real>(plan, d_idata, d_odata, CUFFT_INVERSE));
      CufftSafeCall(cufftDestroy(plan));
      return data;
    }

  }
}
#endif
