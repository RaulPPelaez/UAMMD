#include "cuda_runtime.h"
#include "globals/defines.h"
#include <math.h>
/*Raul P. Pelaez 2016. vector overloads*/


/////////////////REAL4////////////////////////////////
inline __host__ __device__ real4 make_real4(real x, real y, real z, real w)
{
 #ifdef SINGLE_PRECISION 
  return make_float4(x,y,z,w);
  #else
  return make_double4(x,y,z,w);
  #endif
}

inline __host__ __device__ real4 make_real4(real s)
{
    return make_real4(s, s, s, s);
}
inline __host__ __device__ real4 make_real4(real3 a)
{
    return make_real4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ real4 make_real4(real3 a, real w)
{
    return make_real4(a.x, a.y, a.z, w);
}
inline __host__ __device__ real4 make_real4(int4 a)
{
    return make_real4(real(a.x), real(a.y), real(a.z), real(a.w));
}
inline __host__ __device__ real4 make_real4(uint4 a)
{
    return make_real4(real(a.x), real(a.y), real(a.z), real(a.w));
}
//////////////////REAL3///////////////////////////


inline __host__ __device__ real3 make_real3(real x, real y, real z)
{
 #ifdef SINGLE_PRECISION 
  return make_float3(x,y,z);
  #else
  return make_double3(x,y,z);
  #endif
}

inline __host__ __device__ real3 make_real3(real s)
{
    return make_real3(s, s, s);
}
inline __host__ __device__ real3 make_real3(real3 a)
{
  return make_real3(a.x, a.y, a.z);
}
inline __host__ __device__ real3 make_real3(real4 a)
{
  return make_real3(a.x, a.y, a.z);
}
inline __host__ __device__ real3 make_real3(real2 a, real z)
{
    return make_real3(a.x, a.y, z);
}
inline __host__ __device__ real3 make_real3(int3 a)
{
    return make_real3(real(a.x), real(a.y), real(a.z));
}
inline __host__ __device__ real3 make_real3(uint3 a)
{
    return make_real3(real(a.x), real(a.y), real(a.z));
}

//////////////////REAL2///////////////////////////


inline __host__ __device__ real2 make_real2(real x, real y)
{
 #ifdef SINGLE_PRECISION 
  return make_float2(x,y);
  #else
  return make_double2(x,y);
  #endif
}

inline __host__ __device__ real2 make_real2(real s)
{
    return make_real2(s, s);
}
inline __host__ __device__ real2 make_real2(real2 a)
{
  return make_real2(a.x, a.y);
}
inline __host__ __device__ real2 make_real2(real4 a)
{
  return make_real2(a.x, a.y);
}
inline __host__ __device__ real2 make_real2(int3 a)
{
    return make_real2(real(a.x), real(a.y));
}
inline __host__ __device__ real2 make_real2(uint3 a)
{
    return make_real2(real(a.x), real(a.y));
}


////////////////DOUBLE PRECISION//////////////////////

inline __host__ __device__ float4 make_float4(double4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ double4 make_double4(double s)
{
    return make_double4(s, s, s, s);
}
inline __host__ __device__ double4 make_double4(double3 a)
{
    return make_double4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ double4 make_double4(double3 a, double w)
{
    return make_double4(a.x, a.y, a.z, w);
}
inline __host__ __device__ double4 make_double4(int4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}
inline __host__ __device__ double4 make_double4(uint4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}
inline __host__ __device__ double4 make_double4(float4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}

//////DOUBLE4///////////////
inline __host__ __device__  double4 operator +(const double4 &a, const double4 &b){
  return make_double4(
		      a.x + b.x,
		      a.y + b.y,
		      a.z + b.z,
		      a.w + b.w
		      );
}
inline __host__ __device__  void operator +=(double4 &a, const double4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline __host__ __device__  double4 operator +(const double4 &a, const double &b){
  return make_double4(
		      a.x + b,
		      a.y + b,
		      a.z + b,
		      a.w + b
		      );
}
inline __host__ __device__  double4 operator +(const double &b, const double4 &a){
  return make_double4(
		      a.x + b,
		      a.y + b,
		      a.z + b,
		      a.w + b
		      );
}
inline __host__ __device__  void operator +=(double4 &a, const double &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

inline __host__ __device__  double4 operator -(const double4 &a, const double4 &b){
  return make_double4(
		      a.x - b.x,
		      a.y - b.y,
		      a.z - b.z,
		      a.w - b.w
		      );
}
inline __host__ __device__  void operator -=(double4 &a, const double4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
inline __host__ __device__  double4 operator -(const double4 &a, const double &b){
  return make_double4(
		      a.x - b,
		      a.y - b,
		      a.z - b,
		      a.w - b
		      );
}
inline __host__ __device__  double4 operator -(const double &b, const double4 &a){
  return make_double4(
		      a.x - b,
		      a.y - b,
		      a.z - b,
		      a.w - b
		      );
}
inline __host__ __device__  void operator -=(double4 &a, const double &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}
inline __host__ __device__  double4 operator *(const double4 &a, const double4 &b){
  return make_double4(
		      a.x * b.x,
		      a.y * b.y,
		      a.z * b.z,
		      a.w * b.w
		      );
}
inline __host__ __device__  void operator *=(double4 &a, const double4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
inline __host__ __device__  double4 operator *(const double4 &a, const double &b){
  return make_double4(
		      a.x * b,
		      a.y * b,
		      a.z * b,
		      a.w * b
		      );
}
inline __host__ __device__  double4 operator *(const double &b, const double4 &a){
  return make_double4(
		      a.x * b,
		      a.y * b,
		      a.z * b,
		      a.w * b
		      );
}
inline __host__ __device__  void operator *=(double4 &a, const double &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}
inline __host__ __device__  double4 operator /(const double4 &a, const double4 &b){
  return make_double4(
		      a.x / b.x,
		      a.y / b.y,
		      a.z / b.z,
		      a.w / b.w
		      );
}
inline __host__ __device__  void operator /=(double4 &a, const double4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
}
inline __host__ __device__  double4 operator /(const double4 &a, const double &b){
  return make_double4(
		      a.x / b,
		      a.y / b,
		      a.z / b,
		      a.w / b
		      );
}
inline __host__ __device__  double4 operator /(const double &b, const double4 &a){
  return make_double4(
		      b / a.x,
		      b / a.y,
		      b / a.z,
		      b / a.w 
		      );
}
inline __host__ __device__  void operator /=(double4 &a, const double &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
}

inline __host__ __device__ float dot(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ float length(double4 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double4 normalize(double4 v)
{
  double invLen = 1.0/sqrt(dot(v, v));
  return v * invLen;
}
inline __host__ __device__ double4 floorf(double4 v)
{
    return make_double4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

/////////////////////DOUBLE3///////////////////////////////

inline __host__ __device__ int3 make_int3(double3 a){
  return make_int3((int)a.x, (int)a.y, (int)a.z);
}


inline __host__ __device__  double3 operator +(const double3 &a, const double3 &b){
  return make_double3(
		      a.x + b.x,
		      a.y + b.y,
		      a.z + b.z
		      );
}
inline __host__ __device__  void operator +=(double3 &a, const double3 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__  double3 operator +(const double3 &a, const double &b){
  return make_double3(
		      a.x + b,
		      a.y + b,
		      a.z + b
		      );
}
inline __host__ __device__  double3 operator +(const double &b, const double3 &a){
  return make_double3(
		      a.x + b,
		      a.y + b,
		      a.z + b
		      );
}
inline __host__ __device__  void operator +=(double3 &a, const double &b){
  a.x += b;
  a.y += b;
  a.z += b;
}

inline __host__ __device__  double3 operator -(const double3 &a, const double3 &b){
  return make_double3(
		      a.x - b.x,
		      a.y - b.y,
		      a.z - b.z
		      );
}
inline __host__ __device__  void operator -=(double3 &a, const double3 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
inline __host__ __device__  double3 operator -(const double3 &a, const double &b){
  return make_double3(
		      a.x - b,
		      a.y - b,
		      a.z - b
		      );
}
inline __host__ __device__  double3 operator -(const double &b, const double3 &a){
  return make_double3(
		      a.x - b,
		      a.y - b,
		      a.z - b
		      );
}
inline __host__ __device__  void operator -=(double3 &a, const double &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
}
inline __host__ __device__  double3 operator *(const double3 &a, const double3 &b){
  return make_double3(
		      a.x * b.x,
		      a.y * b.y,
		      a.z * b.z
		      );
}
inline __host__ __device__  void operator *=(double3 &a, const double3 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
inline __host__ __device__  double3 operator *(const double3 &a, const double &b){
  return make_double3(
		      a.x * b,
		      a.y * b,
		      a.z * b
		      );
}
inline __host__ __device__  double3 operator *(const double &b, const double3 &a){
  return make_double3(
		      a.x * b,
		      a.y * b,
		      a.z * b
		      );
}
inline __host__ __device__  void operator *=(double3 &a, const double &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
}
inline __host__ __device__  double3 operator /(const double3 &a, const double3 &b){
  return make_double3(
		      a.x / b.x,
		      a.y / b.y,
		      a.z / b.z
		      );
}
inline __host__ __device__  void operator /=(double3 &a, const double3 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}
inline __host__ __device__  double3 operator /(const double3 &a, const double &b){
  return make_double3(
		      a.x / b,
		      a.y / b,
		      a.z / b
		      );
}
inline __host__ __device__  double3 operator /(const double &b, const double3 &a){
  return make_double3(
		      b / a.x,
		      b / a.y,
		      b / a.z
		      );
}
inline __host__ __device__  void operator /=(double3 &a, const double &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
}


inline __host__ __device__ double3 floorf(double3 v)
{
    return make_double3(floor(v.x), floor(v.y), floor(v.z));
}
inline __host__ __device__ float dot(const double3 &a, const double3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float length(double3 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double3 normalize(double3 v)
{
  double invLen = 1.0/sqrt(dot(v, v));
  return v * invLen;
}

inline __host__ __device__ double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}


//////////////////////////////////////////////////////////
/****************************************************************************************/
