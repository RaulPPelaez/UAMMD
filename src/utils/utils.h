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
#include"globals/defines.h"
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<algorithm>
#include<cuda_runtime.h>
#include"helper_gpu.cuh"
#include"helper_math.h"
#include<iostream>
#include<sys/time.h>
#include<memory>
#include"third_party/bravais/bravais.h"
#include<iomanip>

using std::shared_ptr;
using std::string;
using std::cerr;
using std::endl;
using std::cout;
using std::make_shared;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::flush;
using std::setprecision;

std::ostream& operator<<(std::ostream& out, const real3 &f);
std::ostream& operator<<(std::ostream& out, const real4 &f);



/*Returns the input argument number of a flag, -1 if it doesnt exist*/
int checkFlag(int argc, char *argv[], const char *flag);

std::vector<std::string> stringSplit(string s);


#include "Texture.h"
#include "Vector.h"


typedef Vector<real4> Vector4;
typedef GPUVector<real4> GPUVector4;
typedef Vector<real3> Vector3;
typedef GPUVector<real3> GPUVector3;
#define Vector4Ptr shared_ptr<Vector4> 
#define Vector3Ptr shared_ptr<Vector3> 
typedef Matrix<real> Matrixf;


//Returns an identity matrix of size n
Matrixf eye(uint n);



/*A timer class to measure time, just use 
  t.tic to start and t.toc to get elapsed seconds*/
class Timer{
  struct timeval start, end;
public:
  Timer():
    start((struct timeval){0,0}),
    end((struct timeval){0,0}){}
  void tic(){ gettimeofday(&start, NULL); }
  float toc(){
    gettimeofday(&end, NULL);
    return ((end.tv_sec  - start.tv_sec) * 1000000u + 
	    end.tv_usec - start.tv_usec) / 1.e6;
  }
};

//2^64-1
#define RANDOM_MAX 0xFFffFFffFFffFFffULL
/* Pseudorandom number generation */
class Xorshift128plus{
  uint64_t s[2]; /* PRNG state */
public:
  /* The PRNG state must be seeded so that it is not everywhere zero. */
  Xorshift128plus(uint64_t s0, uint64_t s1){
  s[0] = s0;  s[1] = s1;  
  }
  explicit Xorshift128plus(uint64_t s0){
    s[0] = s0;  s[1] = (s0+15438657923749336752ULL)%RANDOM_MAX;  
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
  double uniform(double min, double max){
    return min + (next()/((double) RANDOM_MAX))*(max - min);
  }
  double3 uniform3(double min, double max){
    return {uniform(min, max), uniform(min, max), uniform(min, max)};
  }
  
  double gaussian(double mean, double std){

    const double pi2 = 2.0*M_PI;

    static double z0, z1;
    static bool generate = false;
    generate = !generate;

    if (!generate)
      return z1 * std + mean;

    double u1, u2;
    do{
      u1 = uniform(0,1);
      u2 = uniform(0,1);
    }while ( u1 <= std::numeric_limits<double>::min() );

    z0 = sqrt(-2.0 * log(u1)) * cos(pi2 * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(pi2 * u2);
    return z0 * std + mean;

  }

  
};


//This function treats pos a s a float4 and puts the particles in a cubic lattice
Vector4 initLattice(real3 L, uint N, BRAVAISLAT lat = sc);

//Vector4 cubicLattice2D(float L, uint N);

Vector4 readFile(const char * fileName);

bool randInitial(real4 *pos, real L, uint N);



namespace printUtils{
  /*Get a pretty output of a size in bytes*/
  std::string prettySize(size_t size);
}

#endif
