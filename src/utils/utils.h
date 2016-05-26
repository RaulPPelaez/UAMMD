/*Raul P. Pelaez 2016. Some utils

-A host vector class that holds GPU and CPU versions of the data
-A timer class
-A Xorshift 128+ random number generator

 */
#ifndef UTILS_H
#define UTILS_H
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<algorithm>
#include<cuda_runtime.h>
#include"helper_gpu.cuh"
#include"helper_math.h"
#include<iostream>
#include<sys/time.h>
using namespace std;

#define fori(x,y) for(int i=x; i<y; i++)
#define forj(x,y) for(int j=x; j<y; j++)


template<class T>
class Vector{
public:
  T *data; //The data itself, stored aligned in memory
  T *d_m; //device pointer
  int n; //size of the matrix
  bool pinned; //Flag to use pinned memory
  Vector(){}
  Vector(int n, bool pinned = false){
    this->n = n;
    this->pinned = pinned;
    //Pined memory is allocated by cuda
    if(pinned){ gpuErrchk(cudaMallocHost((void **)&data, sizeof(T)*n));}
    else       data = (T *)malloc(sizeof(T)*n); //C style memory management
    //Allocate device memory
    gpuErrchk(cudaMalloc(&d_m, n*sizeof(T)));
  }
  void fill_with(T x){std::fill(data, data+n, x); }

  //Upload/Download from the GPU, ultra fast if is pinned memory
  inline void upload(){   gpuErrchk(cudaMemcpy(d_m, data, n*sizeof(T), cudaMemcpyHostToDevice)); }
  inline void download(){ gpuErrchk(cudaMemcpy(data, d_m, n*sizeof(T), cudaMemcpyDeviceToHost)); }

  //Free the CPU version of the Vector
  void freeCPU(){
    if(pinned) cudaFreeHost(data);
    else free(data);
  }
  //Free the device memory
  void freeGPU(){ cudaFree(d_m);}  

  void print(){
    download();
    for(int i=0; i<n; i++)
      cout<<data[i]<<" ";
    cout<<endl;
    
  }
  //Access data with bracket operator
  T& operator [](const int &i){return data[i];}
  //Cast to float* returns the device pointer!
  operator T *&() {return d_m;}
  operator T *() const{return d_m;}
};

/*A timer class to measure time, just use 
  t.tic to start and t.toc to get elapsed seconds*/
class Timer{
  struct timeval start, end;
public:
  Timer(){}
  void tic(){ gettimeofday(&start, NULL); }
  float toc(){
    gettimeofday(&end, NULL);
    return ((end.tv_sec  - start.tv_sec) * 1000000u + 
	    end.tv_usec - start.tv_usec) / 1.e6;
  }
};

#ifndef RANDOM_MAX
 #define RANDOM_MAX 18446744073709551615u
#endif
/* Pseudorandom number generation */
class Xorshift128plus{
  uint64_t s[2]; /* PRNG state */
public:
  /* The PRNG state must be seeded so that it is not everywhere zero. */
  Xorshift128plus(uint64_t s0, uint64_t s1){
  s[0] = s0;  s[1] = s1;  
  }
  Xorshift128plus(){
    /* The PRNG state must be seeded so that it is not everywhere zero. */
    s[0] = 12679825035178159220u;
    s[1] = 15438657923749336752u;
  }
  /* 64-bit (pseudo)random integer */
  uint64_t xorshift128plus(void){
    uint64_t x = s[0];
    uint64_t const y = s[1];
    s[0] = y;
    x ^= x << 23; // a
    x ^= x >> 17; // b
    x ^= y ^ (y >> 26); // c
    s[1] = x;
    return x + y;
  }
  /* Random number from a uniform distribution */
  float uniform(float min, float max){
    return min + (xorshift128plus()/((float) RANDOM_MAX))*(max - min);
  }
};


//This function treats pos a s a float4 and puts the particles in a cubic lattice
void cubicLattice(float4 *pos, float L, uint N);
#endif
