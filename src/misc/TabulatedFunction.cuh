/*Raul P. Pelaez 2018. Tabulated Function

  A tabulated function allows to precompute a function on a certain range and
  access to it later on the GPU using interpolation.

  The default interpolation mechanism is linear (like a cuda texture).

  USAGE:
  A table to store reals:
  int ntablePoints = 4096;
    thrust::device_vector<real> tableData(ntablePoints);
    real * d_table = thrust::raw_pointer_cast(tableData.data());
    real rmin = 0;
    real rmax = 1;
    TabulatedFunction<real> tableFunction(d_table, ntablePoints, rmin, rmax,
  foo);

    //After this you will be able to call tableFunction(r) as if it were foo(r)
  inside the interval rmin, rmax
    //Below rmin, tableFunction(r) = tableFunction(rmin). Above rmax,
  tableFunction(r) = 0.

    //Alternatively you can omit the first argument to the constructor and
  TabulatedFunction will maintain an inner buffer.


 */
#ifndef TABULATEDFUNCTION_CUH
#define TABULATEDFUNCTION_CUH

#include "global/defines.h"
#include "third_party/uammd_cub.cuh"
#include "utils/debugTools.h"
#include <vector>

namespace uammd {

template <typename T, typename T2>
__host__ __device__ inline T lerp(T v0, T v1, T2 t) {
#ifdef SINGLE_PRECISION
  return ::fmaf(t, v1, ::fmaf(-t, v0, v0));
#else
  return ::fma(t, v1, ::fma(-t, v0, v0));
#endif
}

template <typename T2>
__host__ __device__ inline real2 lerp(real2 v0, real2 v1, T2 t) {
  return make_real2(lerp(v0.x, v1.x, t), lerp(v0.y, v1.y, t));
}

template <typename T2>
__host__ __device__ inline real3 lerp(real3 v0, real3 v1, T2 t) {
  return make_real3(lerp(v0.x, v1.x, t), lerp(v0.y, v1.y, t),
                    lerp(v0.z, v1.z, t));
}
template <typename T2>
__host__ __device__ inline real4 lerp(real4 v0, real4 v1, T2 t) {
  return make_real4(lerp(v0.x, v1.x, t), lerp(v0.y, v1.y, t),
                    lerp(v0.z, v1.z, t), lerp(v0.w, v1.w, t));
}

struct LinearInterpolation {
  template <class iterator,
            class T = typename std::iterator_traits<iterator>::value_type>
  inline __device__ T operator()(const iterator &table, int Ntable, real dr,
                                 real r) const {
    const int i = r * Ntable;
    const real r0 = i * dr;
    const T v0 = table[i];
    const T v1 = table[i + 1];
    const real t = (r - r0) * (real)Ntable;
    return lerp(v0, v1, t);
  }
};

template <class T, class Interpolation = LinearInterpolation>
struct TabulatedFunction {
  int Ntable;
  real rmin, rmax, interval;
  real dr;
  T *table;
  Interpolation interp;
  bool freeTable = false;
  bool isCopy;

  TabulatedFunction() {}

private:
  T *myCudaMalloc(int N) {
    T *ptr;
    CudaSafeCall(cudaMalloc((void **)&ptr, N * sizeof(T)));
    return ptr;
  }

public:
  template <class Functor>
  TabulatedFunction(int N, real rmin, real rmax, Functor foo)
      : TabulatedFunction(myCudaMalloc(N), N, rmin, rmax, foo) {
    this->freeTable = true;
  }

  template <class Functor>
  TabulatedFunction(T *table, int N, real rmin, real rmax, Functor foo)
      : Ntable(N - 1), rmin(rmin), rmax(rmax), interval(1.0 / (rmax - rmin)),
        dr(1.0 / real(Ntable)), table(table), interp(), freeTable(false),
        isCopy(false) {
    std::vector<T> tableCPU(Ntable + 1);
    for (int i = 0; i <= Ntable; i++) {
      double x = (i / (double)(Ntable)) * (rmax - rmin) + rmin;
      tableCPU[i] = foo(x);
    }
    CudaSafeCall(cudaMemcpy(table, tableCPU.data(), (Ntable + 1) * sizeof(T),
                            cudaMemcpyHostToDevice));
  }

  // This copy constructor prevents cuda from calling the destructor after a
  // kernel call
  TabulatedFunction(const TabulatedFunction &_orig)
      : Ntable(_orig.Ntable), rmin(_orig.rmin), rmax(_orig.rmax),
        interval(_orig.interval), dr(_orig.dr), table(_orig.table),
        interp(_orig.interp), freeTable(false), isCopy(true) {}

  void operator=(TabulatedFunction &&_orig) {
    this->Ntable = _orig.Ntable;
    this->rmin = _orig.rmin;
    this->rmax = _orig.rmax;
    this->interval = _orig.interval;
    this->dr = _orig.dr;
    this->table = _orig.table;
    this->interp = _orig.interp;
    this->freeTable = _orig.freeTable;
    this->isCopy = _orig.isCopy;
    _orig.freeTable = false;
    _orig.isCopy = true;
  }

  ~TabulatedFunction() {
    if (freeTable and !isCopy)
      CudaSafeCall(cudaFree(table));
  }

  template <cub::CacheLoadModifier modifier = cub::LOAD_DEFAULT>
  inline __device__ T get(real rs) const {
    return this->operator()<modifier>(rs);
  }

  template <cub::CacheLoadModifier modifier = cub::LOAD_DEFAULT>
  inline __device__ T operator()(real rs) const {
    real r = (rs - rmin) * interval;
    if (rs >= rmax)
      return T();
    if (r <= real(0.0))
      return table[0];
    cub::CacheModifiedInputIterator<modifier, T> table_itr(table);
    return interp(table_itr, Ntable, dr, r);
  }
};

} // namespace uammd

#endif
