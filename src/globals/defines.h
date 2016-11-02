#include<cuda_runtime.h>
#include<stdint.h>

#define UAMMD_VERSION "0.01(alpha)"
#define SINGLE_PRECISION

typedef uint32_t uint;
typedef unsigned long long int ullint;


#if defined SINGLE_PRECISION
typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;

#else
typedef double real;
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
#endif



