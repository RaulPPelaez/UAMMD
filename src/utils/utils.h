/*Raul P. Pelaez 2016. Some utils

-A host vector class that holds GPU and CPU versions of the data
-A host Matrix class that holds GPU and CPU versions of the data (derived from Vector, allows for [][] access)
-A timer class
-A Xorshift 128+ random number generator
-Functions to create initial positions of particles


-Several typedefs, defines and cast overloads to make things easier to read

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
#include<memory>

using namespace std;
typedef uint32_t uint;

#define fori(x,y) for(int i=x; i<y; i++)
#define forj(x,y) for(int j=x; j<y; j++)

#include "Vector.h"


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

//2^64-1
#define RANDOM_MAX 18446744073709551615ULL
/* Pseudorandom number generation */
class Xorshift128plus{
  uint64_t s[2]; /* PRNG state */
public:
  /* The PRNG state must be seeded so that it is not everywhere zero. */
  Xorshift128plus(uint64_t s0, uint64_t s1){
  s[0] = s0;  s[1] = s1;  
  }
  Xorshift128plus(uint64_t s0){
  s[0] = s0;  s[1] = s0+15438657923749336752ULL;  
  }

  Xorshift128plus(){
    /* The PRNG state must be seeded so that it is not everywhere zero. */
    s[0] = 12679825035178159220ULL;
    s[1] = 15438657923749336752ULL;
  }
  /* 64-bit (pseudo)random integer */
  uint64_t next(void){
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
    return min + (next()/((float) RANDOM_MAX))*(max - min);
  }
};


//This function treats pos a s a float4 and puts the particles in a cubic lattice
void cubicLattice(float4 *pos, float L, uint N);
void cubicLattice2D(float4 *pos, float L, uint N);

void readFile(Vector4 &pos, const char * fileName);

bool randInitial(float4 *pos, float L, uint N);

#endif
