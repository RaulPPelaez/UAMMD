/*Raul P. Pelaez 2017-2022. Some general utilities. Mainly:

  -A Timer to keep track of time
  -A Xorshift128+ CPU rng


References:
[1] https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
 */
#ifndef UAMMD_UTILS_H
#define UAMMD_UTILS_H
#include<cstdint>
#include<limits>
#include<sys/time.h>
#include <vector>
#include"utils/ForceEnergyVirial.cuh"
#include"printOverloads.h"
namespace uammd{
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

// Pseudorandom number generation, see [1]
class Xorshift128plus{
  uint64_t s[2]; /* PRNG state */
public:
  /* The PRNG state must be seeded so that it is not everywhere zero. */
  Xorshift128plus(uint64_t s0, uint64_t s1){
    setSeed(s0, s1);
  }
  explicit Xorshift128plus(uint64_t s0){
    setSeed(s0);
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
  /* 32-bit (pseudo)random integer */
  uint32_t next32(void){
    return next()%std::numeric_limits<uint32_t>::max();
  }

  /* Random number from a uniform distribution */
  double uniform(double min, double max){
    return min + (next()/((double) RANDOM_MAX))*(max - min);
  }
  double3 uniform3(double min, double max){
    return {uniform(min, max), uniform(min, max), uniform(min, max)};
  }
  double2 uniform2(double min, double max){
    return {uniform(min, max), uniform(min, max)};
  }

  double gaussian(double mean, double std){
    constexpr double pi2 = 2.0*M_PI;
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

  double3 gaussian3(double mean, double std){
    return make_double3(gaussian(mean, std),
			gaussian(mean, std),
			gaussian(mean, std));
  }

  double2 gaussian2(double mean, double std){
    return make_double2(gaussian(mean, std),
			gaussian(mean, std));
  }

  void setSeed(uint64_t s0, uint64_t s1){
    s[0] = s0;  s[1] = s1;
  }

  void setSeed(uint64_t s0){
    s[0] = s0;  s[1] = (s0+15438657923749336752ULL)%RANDOM_MAX;
  }

};

}
#endif
