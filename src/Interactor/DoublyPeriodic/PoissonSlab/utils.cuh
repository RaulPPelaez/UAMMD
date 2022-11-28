#ifndef DOUBLYPERIODIC_POISSONSLAB_UTILS_CUH
#define DOUBLYPERIODIC_POISSONSLAB_UTILS_CUH
#include "global/defines.h"
#include "utils/utils.h"
#include "System/System.h"
#include <thrust/device_vector.h>
#include"utils/cufftDebug.h"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftComplex4.cuh"
#include "utils/cufftComplex2.cuh"
#include <fstream>
#include <iomanip>
#include<numeric>
namespace uammd{
  namespace DPPoissonSlab_ns{
    struct Permitivity{
      real top, bottom, inside;
    };

    class SurfaceValueDispatch{

    public:

      virtual real top(real x, real y){
	return 0;
      }

      virtual real bottom(real x, real y){
	return 0;
      }

    };

    namespace detail{
      //This is a very barebones container. Its purpose is only to avoid the unnecessary unninitialized_fill kernel that thrust issues on device_vector creation. Thus it mascarades as a thrust::device_vector.
      template<class T, class Allocator = System::allocator<T>>
      class UninitializedCachedContainer{
	using Container = std::shared_ptr<T>;
	using Ptr = T*;
	Container m_data;
	size_t m_size, capacity;

	Container create(size_t s){
	  if(s>0){
	    try{
	      return Container(Allocator().allocate(s), [](T* ptr){Allocator().deallocate(ptr);});
	    }
	    catch(...){
	      System::log<System::EXCEPTION>("[UninitializedCachedContainer] Could not allocate buffer of size %zu", s);
	      throw;
	    }
	  }
	  else{
	    return Container();
	  }
	}

      public:
	UninitializedCachedContainer(size_t i_size = 0)
	  : m_size(0), capacity(0), m_data() {
	  this->resize(i_size);
	}

	UninitializedCachedContainer(const std::vector<T> &other):
	  UninitializedCachedContainer(other.size()){
	  thrust::copy(other.begin(), other.end(), thrust::device_ptr<T>(begin()));
	}

	Ptr begin() const{ return m_data.get(); }

	Ptr end() const{ return m_data.get() + m_size; }

	size_t size() const{
	  return m_size;
	}

	void resize(size_t newsize){
	  if(newsize > capacity){
	    auto data2 = create(newsize);
	    if(size()>0){
	      thrust::copy(thrust::cuda::par, begin(), end(), data2.get());
	    }
	    m_data.swap(data2);
	    capacity = newsize;
	    m_size = newsize;
	  }
	  else{
	    m_size = newsize;
	  }
	}

	thrust::device_ptr<T> data() const{
	  return thrust::device_ptr<T>(begin());
	}

      };
    }
#ifndef UAMMD_DEBUG
    template<class T> using gpu_container = thrust::device_vector<T>;
    //template<class T>  using cached_vector = thrust::device_vector<T, System::allocator_thrust<T>>;
    template<class T>  using cached_vector = detail::UninitializedCachedContainer<T>;
#else
    template<class T> using gpu_container = thrust::device_vector<T, managed_allocator<T>>;
    template<class T> using cached_vector = thrust::device_vector<T, managed_allocator<T>>;
#endif

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

    class Index3D{
      const int nx, ny, nz;
    public:
      __host__ __device__ Index3D(int nx, int ny, int nz): nx(nx), ny(ny), nz(nz){};
      inline __host__ __device__ int operator()(int x, int y, int z) const{
	return x + nx*y + z*nx*ny;
      }

      inline __host__ __device__ int operator()(int3 i) const{
	return this->operator()(i.x, i.y, i.z);
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

    template<class Container>
    void writeComplexField(Container & field, int3 dim, real H, std::string file){
      int nkx = dim.x;
      int nky = dim.y;
      int nz = dim.z;
      int ntot = (nkx/2+1)*nky*nz;
      std::ofstream out(file);
      std::vector<cufftComplex> h_field(field.size());
      auto ptr = field.data();
      thrust::copy(ptr, ptr + field.size(), h_field.begin());
      for (int id = 0; id<(nkx/2+1)*nky; id++){
	int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
	Index3D indexer(nkx/2+1, nky, nz);
	auto fn = make_third_index_iterator(h_field.data(), ik.x, ik.y, indexer);
	out<<std::setprecision(2*sizeof(real));
	fori(0, nz){
	  //real z = real(0.5)*H*cos(M_PI*i/(nz-1));
	  real z = i;
	  auto f = fn[i];
	  out<<ik.x<<" "<<ik.y<<" "<<z<<" "<<f<<"\n";
	}
      }
    }


    template<class Container>
    void writeComplex4Field(Container & field, int3 dim, real H, std::string file){
      int nkx = dim.x;
      int nky = dim.y;
      int nz = dim.z;
      int ntot = (nkx/2+1)*nky*nz;
      std::ofstream out(file);
      std::vector<cufftComplex4> h_field(field.size());
      auto ptr = field.data();
      thrust::copy(ptr, ptr + field.size(), h_field.begin());
      for (int id = 0; id<(nkx/2+1)*nky; id++){
	int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
	Index3D indexer(nkx/2+1, nky, nz);
	auto fn = make_third_index_iterator(h_field.data(), ik.x, ik.y, indexer);
	out<<std::setprecision(2*sizeof(real));
	fori(0, nz){
	  //real z = real(0.5)*H*cos(M_PI*i/(nz-1));
	  real z = i;
	  auto f = fn[i];
	  out<<ik.x<<" "<<ik.y<<" "<<z<<" "<<f.x<<" "<<f.y<<" "<<f.z<<" "<<f.w<<"\n";
	}
      }
    }

    template<class Container>
    void writeComplex2Field(Container & field, int3 dim, real H, std::string file){
      int nkx = dim.x;
      int nky = dim.y;
      int nz = dim.z;
      int ntot = (nkx/2+1)*nky*nz;
      std::ofstream out(file);
      out<<std::setprecision(2*sizeof(real));
      std::vector<cufftComplex2> h_field(field.size());
      auto ptr = field.data();
      thrust::copy(ptr, ptr + field.size(), h_field.begin());
      out<<std::setprecision(2*sizeof(real));
      for (int id = 0; id<(nkx/2+1)*nky; id++){
	int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
	Index3D indexer(nkx/2+1, nky, nz);
	auto fn = make_third_index_iterator(h_field.data(), ik.x, ik.y, indexer);
	fori(0,nz){
	  //real z = real(0.5)*H*cos(M_PI*i/(nz-1));
	  real z = i;
	  auto f = fn[i];
	  out<<ik.x<<" "<<ik.y<<" "<<z<<" "<<f.x<<" "<<f.y<<"\n";
	}
      }
    }

    template<class T, class Container>
    void writeRealField(Container & field, int3 dim, std::string file){
      int nx = dim.x;
      int ny = dim.y;
      int nz = dim.z;
      int ntot = nx*ny*nz;
      std::ofstream out(file);
      std::vector<T> h_field(field.size());
      auto ptr = field.data();
      thrust::copy(ptr, ptr + field.size(), h_field.begin());
      out<<std::setprecision(2*sizeof(real));
      for (int id = 0; id<ntot; id++){
        int3 ik = {id%nx, (id/nx)%ny, id/(nx*ny)};
	Index3D indexer(2*(nx/2+1), ny, nz);
	auto f = h_field[indexer(ik)];
	out<<ik.x<<" "<<ik.y<<" "<<ik.z<<" "<<f<<"\n";
      }
    }
  }
}

#endif
