#ifndef DOUBLYPERIODIC_STOKESSLAB_UTILS_CUH
#define DOUBLYPERIODIC_STOKESSLAB_UTILS_CUH
#include "global/defines.h"
#include "third_party/managed_allocator.h"
#include "utils/utils.h"
#include "System/System.h"
#include <thrust/device_vector.h>
#include"utils/cufftDebug.h"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftComplex4.cuh"
#include "utils/cufftComplex2.cuh"
#include"utils/container.h"
namespace uammd{
  namespace DPStokesSlab_ns{  
#ifndef UAMMD_DEBUG
    template<class T> using gpu_container = thrust::device_vector<T>;
    template<class T>  using cached_vector = uninitialized_cached_vector<T>;
#else
    template<class T> using gpu_container = thrust::device_vector<T, managed_allocator<T>>;
    template<class T> using cached_vector = thrust::device_vector<T, managed_allocator<T>>;
#endif

    enum class WallMode{bottom, slit, none};

    class IndexToWaveNumber{
      const int nkx, nky;
    public:
      __device__ __host__ IndexToWaveNumber(int nkx, int nky):nkx(nkx), nky(nky){}

      __host__ __device__ int2 operator()(int i) const{
	int ikx = i%(nkx/2+1);
	int iky = i/(nkx/2+1);
	ikx -= nkx*(ikx >= (nkx/2+1));
	iky -= nky*(iky >= (nky/2+1));
	return make_int2(ikx, iky);
      }
    };

    class WaveNumberToWaveVector{
      const real2 waveNumberToWaveVector;
    public:
      __device__ __host__ WaveNumberToWaveVector(real2 L):
	waveNumberToWaveVector(real(2.0)*real(M_PI)/L){
      }

      __host__ __device__ real2 operator()(int2 ik) const{
	const real2 k = make_real2(ik)*waveNumberToWaveVector;
	return k;
      }
    };

    class IndexToWaveVector{
      IndexToWaveNumber i2wn;
      WaveNumberToWaveVector wn2wv;
    public:
      __device__ __host__ IndexToWaveVector(int nkx, int nky, real2 L):i2wn(nkx, nky), wn2wv(L){}

      __host__ __device__ real2 operator()(int i) const{
	return wn2wv(i2wn(i));
      }
    };

    class WaveNumberToWaveVectorModulus{
      const WaveNumberToWaveVector wn2wv;
    public:
      __device__ __host__ WaveNumberToWaveVectorModulus(real2 L): wn2wv(L){}

      __host__ __device__ real operator()(int2 ik) const{
	const real2 k = wn2wv(ik);
	const real kmod = sqrt(dot(k,k));
	return kmod;
      }

    };

    class IndexToWaveVectorModulus{
      const IndexToWaveNumber id2ik;
      const WaveNumberToWaveVectorModulus ik2k;
    public:
      __device__ __host__ IndexToWaveVectorModulus(IndexToWaveNumber id2ik, WaveNumberToWaveVectorModulus ik2k):
	id2ik(id2ik), ik2k(ik2k){}

      __host__ __device__ real operator()(int i) const{
	int2 waveNumber = id2ik(i);
	real k = ik2k(waveNumber);
	return k;
      }
    };

    using WaveVectorListIterator = thrust::transform_iterator<IndexToWaveVectorModulus,
      thrust::counting_iterator<int>>;

    WaveVectorListIterator make_wave_vector_modulus_iterator(int2 nk, real2 Lxy){
      IndexToWaveVectorModulus i2k(IndexToWaveNumber(nk.x, nk.y),
				   WaveNumberToWaveVectorModulus(Lxy));
      auto klist = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), i2k);
      return klist;
    }

    __device__ int2 computeWaveNumber(int id, int nkx, int nky){
      IndexToWaveNumber id2wn(nkx, nky);
      const auto waveNumber = id2wn(id);
      return waveNumber;
    }

    __device__ real2 computeWaveVector(int2 waveNumber, real2 Lxy){
      WaveNumberToWaveVector wn2wv(Lxy);
      const auto waveVector = wn2wv(waveNumber);
      return waveVector;
    }

    class Index3D{
      const int nx, ny, nz;
    public:
      __host__ __device__ Index3D(int nx, int ny, int nz): nx(nx), ny(ny), nz(nz){};
      inline __host__ __device__ int operator()(int x, int y, int z) const{
	return x + nx*y + z*nx*ny;
      }

    };

    struct ThirdIndexIteratorTransform{
      const Index3D index;
      const int x, y;
      __host__ __device__ ThirdIndexIteratorTransform(Index3D index, int x, int y): index(index), x(x), y(y){}

      inline __host__ __device__ int operator()(int z) const{
	return index(x, y, z);
      }
    };

    template<class RandomAccessIterator>
    using Iterator = thrust::permutation_iterator<RandomAccessIterator,
      thrust::transform_iterator<ThirdIndexIteratorTransform, thrust::counting_iterator<int>>>;

    template<class RandomAccessIterator>
    inline __host__ __device__
    Iterator<RandomAccessIterator> make_third_index_iterator(RandomAccessIterator ptr,
							     int ikx, int iky,
							     const Index3D &index){
      auto tr = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
					        ThirdIndexIteratorTransform(index, ikx, iky));
      return thrust::make_permutation_iterator(ptr, tr);
    }

    using cufftComplex4 = cufftComplex4_t<real>;
    using cufftComplex2 = cufftComplex2_t<real>;
    using cufftComplex = cufftComplex_t<real>;
    using cufftReal = cufftReal_t<real>;

  }
}

#endif
