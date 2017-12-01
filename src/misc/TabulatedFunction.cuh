/*Raul P. Pelaez. Tabulated Function
  
  A tabulated function allows to precompute a function on a certain range and access to it later on the GPU using
  interpolation.
  
  The default interpolation mechanism is linear (like a cuda texture).

  USAGE:
  A table to store reals:
  int ntablePoints = 4096;
    thrust::device_vector<real> tableData(ntablePoints);
    real * d_table = thrust::raw_pointer_cast(tableData.data());
    real rmin = 0;
    real rmax = 1;
    TabulatedFunction<real> tableFunction(d_table, ntablePoints, rmin, rmax, foo);

    //After this you will be able to call tableFunction(r) as if it were foo(r) inside the interval rmin, rmax   
    //Below rmin, tableFunction(r) = tableFunction(rmin). Above rmax, tableFunction(r) = 0.
    
    

 */
#ifndef TABULATEDFUNCTION_CUH
#define TABULATEDFUNCTION_CUH

#include"global/defines.h"

#include<thrust/device_vector.h>

namespace uammd{

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
    return make_real3(lerp(v0.x, v1.x, t),
		      lerp(v0.y, v1.y, t),
		      lerp(v0.z, v1.z, t));
  }
  template <typename T2>
  __host__ __device__ inline real4 lerp(real4 v0, real4 v1, T2 t) {
    return make_real4(lerp(v0.x, v1.x, t),
		      lerp(v0.y, v1.y, t),
		      lerp(v0.z, v1.z, t),
		      lerp(v0.w, v1.w, t));
  }

  struct LinearInterpolation{
    template<class T>
    inline __host__  __device__ T operator()(T *table, int Ntable, real dr, real r){
      int i = r*Ntable;
      real r0 = i*dr;    
      T v0 = table[i];
      T v1 = table[i+1];
    
      real t = (r - r0)*(real)Ntable;

      return lerp(v0, v1, t);    
    }
  };

 
  template<class T, class Interpolation = LinearInterpolation>
  struct TabulatedFunction{
    int Ntable;
    real rmin, rmax, invInterval;
    real drNormalized;
    T *table;
    Interpolation interp;
    template<class Functor>
    TabulatedFunction(T* table, int N, real rmin, real rmax, Functor foo):
      Ntable(N-1),
      rmin(rmin),
      rmax(rmax),
      table(table),
      drNormalized(1.0/(real)Ntable),
      invInterval(1.0/(rmax-rmin)),
      interp()    
    {

      thrust::host_vector<T> tableCPU(Ntable+1);

      for(int i = 0; i<=Ntable; i++){
	double x = (i/(double)(Ntable))*(rmax-rmin) + rmin;
	tableCPU[i] = foo(x);
      }    
      cudaMemcpy(table,
		 thrust::raw_pointer_cast(tableCPU.data()),
		 (Ntable+1)*sizeof(T),
		 cudaMemcpyHostToDevice);
    
    }
    ~TabulatedFunction(){ }
    __host__ __device__ T operator()(real rs){
      real r = (rs-rmin)*invInterval;
      if(rs >= rmax) return T();
      if(r < real(0.0)) return table[0];
      return interp(table, Ntable, drNormalized, r);
    }
 
  };

  /*
    struct LJ{
    __device__ __host__ real2 operator()(real x){
    
    return {(real)(pow(x, -13) - pow(x,-7)), real(1.0/(x*x))};
    }
    };

  template<class Table, class Functor>
  __global__ void sample(Table table, Functor foo, real rmin, real rmax, int N){
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    real x = (real(id)/real(N))*(rmax-rmin)+rmin;
    printf("%d %.9f %.9f %.9f\n", id, x, foo(x).y, table(x).y);


  }
  

  int main(){

    int Ntable = 64000;
    LJ lj;
    real rmax = 2.5;
    real rmin = 0.1;
    Table<real2, LinearInterpolation> table(Ntable, rmin, rmax, lj);

    cerr<<rmax/Ntable<<endl;

  

    int N = 2*Ntable;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1e8*sizeof(char));

    sample<<<N, 1>>>(table, lj, rmin, rmax, N);
  
    cudaDeviceSynchronize();
  
    return 0;
  }
  */
}

#endif