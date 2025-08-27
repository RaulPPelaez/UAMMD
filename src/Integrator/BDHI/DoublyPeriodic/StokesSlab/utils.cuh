#ifndef DOUBLYPERIODIC_STOKESSLAB_UTILS_CUH
#define DOUBLYPERIODIC_STOKESSLAB_UTILS_CUH
#include "System/System.h"
#include "global/defines.h"
#include "third_party/managed_allocator.h"
#include "utils/container.h"
#include "utils/cufftComplex2.cuh"
#include "utils/cufftComplex3.cuh"
#include "utils/cufftComplex4.cuh"
#include "utils/cufftDebug.h"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/utils.h"
#include <thrust/device_vector.h>
namespace uammd {
namespace DPStokesSlab_ns {
#ifndef UAMMD_DEBUG
template <class T> using gpu_container = thrust::device_vector<T>;
template <class T> using cached_vector = uninitialized_cached_vector<T>;
#else
template <class T>
using gpu_container = thrust::device_vector<T, managed_allocator<T>>;
template <class T>
using cached_vector = thrust::device_vector<T, managed_allocator<T>>;
#endif
/**
 * @enum WallMode
 * @brief Specifies the mode of wall configuration in the simulation.
 */
enum class WallMode {
  bottom, ///< Only the bottom wall is present.
  slit,   ///< Both top and bottom walls are present, forming a slit.
  none,   ///< No walls are present.
};

class IndexToWaveNumber {
  const int nkx, nky;

public:
  __device__ __host__ IndexToWaveNumber(int nkx, int nky)
      : nkx(nkx), nky(nky) {}

  __host__ __device__ int2 operator()(int i) const {
    int ikx = i % (nkx / 2 + 1);
    int iky = i / (nkx / 2 + 1);
    ikx -= nkx * (ikx >= (nkx / 2 + 1));
    iky -= nky * (iky >= (nky / 2 + 1));
    return make_int2(ikx, iky);
  }
};

class WaveNumberToWaveVector {
  const real2 waveNumberToWaveVector;

public:
  __device__ __host__ WaveNumberToWaveVector(real2 L)
      : waveNumberToWaveVector(real(2.0) * real(M_PI) / L) {}

  __host__ __device__ real2 operator()(int2 ik) const {
    const real2 k = make_real2(ik) * waveNumberToWaveVector;
    return k;
  }
};

class IndexToWaveVector {
  IndexToWaveNumber i2wn;
  WaveNumberToWaveVector wn2wv;

public:
  __device__ __host__ IndexToWaveVector(int nkx, int nky, real2 L)
      : i2wn(nkx, nky), wn2wv(L) {}

  __host__ __device__ real2 operator()(int i) const { return wn2wv(i2wn(i)); }
};

class WaveNumberToWaveVectorModulus {
  const WaveNumberToWaveVector wn2wv;

public:
  __device__ __host__ WaveNumberToWaveVectorModulus(real2 L) : wn2wv(L) {}

  __host__ __device__ real operator()(int2 ik) const {
    const real2 k = wn2wv(ik);
    const real kmod = sqrt(dot(k, k));
    return kmod;
  }
};

class IndexToWaveVectorModulus {
  const IndexToWaveNumber id2ik;
  const WaveNumberToWaveVectorModulus ik2k;

public:
  __device__ __host__ IndexToWaveVectorModulus(
      IndexToWaveNumber id2ik, WaveNumberToWaveVectorModulus ik2k)
      : id2ik(id2ik), ik2k(ik2k) {}

  __host__ __device__ real operator()(int i) const {
    int2 waveNumber = id2ik(i);
    real k = ik2k(waveNumber);
    return k;
  }
};

using WaveVectorListIterator =
    thrust::transform_iterator<IndexToWaveVectorModulus,
                               thrust::counting_iterator<int>>;

WaveVectorListIterator make_wave_vector_modulus_iterator(int2 nk, real2 Lxy) {
  IndexToWaveVectorModulus i2k(IndexToWaveNumber(nk.x, nk.y),
                               WaveNumberToWaveVectorModulus(Lxy));
  auto klist = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0), i2k);
  return klist;
}

__device__ int2 computeWaveNumber(int id, int nkx, int nky) {
  IndexToWaveNumber id2wn(nkx, nky);
  const auto waveNumber = id2wn(id);
  return waveNumber;
}

__device__ real2 computeWaveVector(int2 waveNumber, real2 Lxy) {
  WaveNumberToWaveVector wn2wv(Lxy);
  const auto waveVector = wn2wv(waveNumber);
  return waveVector;
}

class Index3D {
  const int nx, ny, nz;

public:
  __host__ __device__ Index3D(int nx, int ny, int nz)
      : nx(nx), ny(ny), nz(nz) {};
  inline __host__ __device__ int operator()(int x, int y, int z) const {
    return x + nx * y + z * nx * ny;
  }
};

struct ThirdIndexIteratorTransform {
  const Index3D index;
  const int x, y;
  __host__ __device__ ThirdIndexIteratorTransform(Index3D index, int x, int y)
      : index(index), x(x), y(y) {}

  inline __host__ __device__ int operator()(int z) const {
    return index(x, y, z);
  }
};

template <class RandomAccessIterator>
inline __host__ __device__ auto
make_third_index_iterator(RandomAccessIterator ptr, int ikx, int iky,
                          const Index3D &index) {
  auto tr = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      ThirdIndexIteratorTransform(index, ikx, iky));
  return thrust::make_permutation_iterator(ptr, tr);
}

using complex4 = cufftComplex4_t<real>;
using complex3 = cufftComplex3_t<real>;
using complex2 = cufftComplex2_t<real>;
using complex = cufftComplex_t<real>;
template <class T> struct AoSToSoAVec3 {
  template <class VecType>
  __device__ thrust::tuple<T, T, T> operator()(VecType v) {
    return thrust::make_tuple(v.x, v.y, v.z);
  }
};

struct SoAToAoSVec3 {
  __device__ real3 operator()(thrust::tuple<real, real, real> v) {
    return {thrust::get<0>(v), thrust::get<1>(v), thrust::get<2>(v)};
  }
  __device__ complex3 operator()(thrust::tuple<complex, complex, complex> v) {
    return {thrust::get<0>(v), thrust::get<1>(v), thrust::get<2>(v)};
  }
  __device__ complex4
  operator()(thrust::tuple<complex, complex, complex, complex> v) {
    return {thrust::get<0>(v), thrust::get<1>(v), thrust::get<2>(v),
            thrust::get<3>(v)};
  }
};

struct ToReal3 {
  template <class T> __device__ real3 operator()(T v) { return make_real3(v); }
};
__device__ complex3 toComplex3(thrust::tuple<complex, complex, complex> v) {
  return {thrust::get<0>(v), thrust::get<1>(v), thrust::get<2>(v)};
}

__device__ auto toTuple(complex3 v) {
  return thrust::make_tuple(v.x, v.y, v.z);
}

struct ToReal4 {
  template <class T> __device__ real4 operator()(T v) { return make_real4(v); }
};

template <class T> struct DataXYZ {
  cached_vector<T> m_x, m_y, m_z;

  DataXYZ() : DataXYZ(0) {}

  template <class VectorTypeIterator>
  DataXYZ(VectorTypeIterator &input, int size) : DataXYZ(size) {
    auto zip = thrust::make_zip_iterator(
        thrust::make_tuple(m_x.begin(), m_y.begin(), m_z.begin()));
    thrust::transform(thrust::cuda::par, input, input + size, zip,
                      AoSToSoAVec3<T>());
  }

  DataXYZ(int size) { resize(size); }

  void resize(int newSize) {
    m_x.resize(newSize);
    m_y.resize(newSize);
    m_z.resize(newSize);
  }

  void fillWithZero() {
    thrust::fill(m_x.begin(), m_x.end(), T());
    thrust::fill(m_y.begin(), m_y.end(), T());
    thrust::fill(m_z.begin(), m_z.end(), T());
  }

  using Iterator = T *;
  Iterator x() const { return (Iterator)m_x.data().get(); }
  Iterator y() const { return (Iterator)m_y.data().get(); }
  Iterator z() const { return (Iterator)m_z.data().get(); }

  auto xyz() const {
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(x(), y(), z()));
    const auto tr = thrust::make_transform_iterator(zip, SoAToAoSVec3());
    return tr;
  }

  void swap(DataXYZ &another) {
    m_x.swap(another.m_x);
    m_y.swap(another.m_y);
    m_z.swap(another.m_z);
  }

  void clear() {
    m_x.clear();
    m_y.clear();
    m_z.clear();
  }

  auto size() const { return this->m_x.size(); }

  DataXYZ(const DataXYZ<T> &other)
      : // copy constructor
        m_x(other.m_x), m_y(other.m_y), m_z(other.m_z) {}

  DataXYZ(DataXYZ<T> &&other) noexcept
      : // move constructor
        m_x(std::exchange(other.m_x, cached_vector<T>())),
        m_y(std::exchange(other.m_y, cached_vector<T>())),
        m_z(std::exchange(other.m_z, cached_vector<T>())) {}

  auto operator=(const DataXYZ<T> &other) { // copy assignment
    return *this = other;
  }

  DataXYZ<T> &operator=(DataXYZ<T> &&other) { // move assignment
    std::swap(m_x, other.m_x);
    std::swap(m_y, other.m_y);
    std::swap(m_z, other.m_z);
    return *this;
  }
};

template <class T> class DataXYZPtr {
  typename DataXYZ<T>::Iterator m_x, m_y, m_z;

public:
  DataXYZPtr(const DataXYZ<T> &data)
      : m_x(data.x()), m_y(data.y()), m_z(data.z()) {}

  using Iterator = T *;
  __host__ __device__ Iterator x() const { return m_x; }
  __host__ __device__ Iterator y() const { return m_y; }
  __host__ __device__ Iterator z() const { return m_z; }

  __host__ __device__ auto xyz() const {
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(x(), y(), z()));
    const auto tr = thrust::make_transform_iterator(zip, SoAToAoSVec3());
    return tr;
  }
};

auto toReal3Vector(DataXYZ<real> &data) {
  cached_vector<real3> out(data.size());
  thrust::copy(thrust::cuda::par, data.xyz(), data.xyz() + data.size(),
               out.begin());
  return out;
}

template <class T> struct FluidPointers {
  FluidPointers() {}
  template <class Container>
  FluidPointers(const Container &pressure, const DataXYZ<T> &vel)
      : pressure((T *)(pressure.data().get())), velocityX(vel.x()),
        velocityY(vel.y()), velocityZ(vel.z()) {}
  T *pressure;
  typename DataXYZ<T>::Iterator velocityX, velocityY, velocityZ;
};

template <class T> struct FluidData {
  FluidData(int3 n)
      : m_size(n), velocity(n.x * n.y * n.z), pressure(n.x * n.y * n.z) {}

  FluidData() : FluidData({0, 0, 0}) {}

  DataXYZ<T> velocity;
  cached_vector<T> pressure;
  int3 m_size;

  FluidPointers<T> getPointers() const {
    return FluidPointers<T>(pressure, velocity);
  }

  void resize(int3 n) {
    this->m_size = n;
    int newSize = n.x * n.y * n.z;
    velocity.resize(newSize);
    pressure.resize(newSize);
  }

  void clear() {
    velocity.clear();
    pressure.clear();
  }

  int3 size() const { return m_size; }

  auto operator=(const FluidData<T> &other) { // copy assignment
    return *this = FluidData(other);
  }

  FluidData(const FluidData<T> &other)
      : // copy constructor
        velocity(other.velocity), pressure(other.pressure),
        m_size(other.m_size) {}

  FluidData(FluidData<T> &&other) noexcept
      : // move constructor
        velocity(std::exchange(other.velocity, DataXYZ<T>())),
        pressure(std::exchange(other.pressure, cached_vector<T>())),
        m_size(std::exchange(other.m_size, int3())) {}

  FluidData<T> &operator=(FluidData<T> &&other) noexcept { // move assignment
    std::swap(velocity, other.velocity);
    std::swap(pressure, other.pressure);
    std::swap(m_size, other.m_size);
    return *this;
  }
};

} // namespace DPStokesSlab_ns
} // namespace uammd

#endif
