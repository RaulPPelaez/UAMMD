#include "cuda_runtime.h"
#include "globals/defines.h"
#include <math.h>
/*Raul P. Pelaez 2016. vector overloads not defined in helper_math.h*/

#define VECATTR inline __host__ __device__
/////////////////REAL4////////////////////////////////
VECATTR real4 make_real4(real x, real y, real z, real w){
 #ifdef SINGLE_PRECISION 
  return make_float4(x,y,z,w);
  #else
  return make_double4(x,y,z,w);
  #endif
}

VECATTR real4 make_real4(real s){return make_real4(s, s, s, s);}
VECATTR real4 make_real4(real3 a){ return make_real4(a.x, a.y, a.z, 0.0f);}
VECATTR real4 make_real4(real3 a, real w){ return make_real4(a.x, a.y, a.z, w);}

#ifdef SINGLE_PRECISION
VECATTR real4 make_real4(double3 a, real w){return make_real4(a.x, a.y, a.z, w);}
#else
VECATTR real4 make_real4(float3 a, real w){ return make_real4(a.x, a.y, a.z, w);}
#endif

VECATTR real4 make_real4(int4 a){ return make_real4(real(a.x), real(a.y), real(a.z), real(a.w));}
VECATTR real4 make_real4(uint4 a){return make_real4(real(a.x), real(a.y), real(a.z), real(a.w));}
//////////////////REAL3///////////////////////////


VECATTR real3 make_real3(real x, real y, real z){
 #ifdef SINGLE_PRECISION 
  return make_float3(x,y,z);
  #else
  return make_double3(x,y,z);
  #endif
}

VECATTR real3 make_real3(real s){ return make_real3(s, s, s);}
VECATTR real3 make_real3(real3 a){return make_real3(a.x, a.y, a.z);}

#ifdef SINGLE_PRECISION
VECATTR real3 make_real3(double3 a){return make_real3(a.x, a.y, a.z);}
VECATTR real3 make_real3(double4 a){return make_real3(a.x, a.y, a.z);}
#else
VECATTR real3 make_real3(float3 a){return make_real3(a.x, a.y, a.z);}
VECATTR real3 make_real3(float4 a){return make_real3(a.x, a.y, a.z);}
#endif
VECATTR real3 make_real3(real4 a){ return make_real3(a.x, a.y, a.z);}

VECATTR real3 make_real3(real2 a, real z){return make_real3(a.x, a.y, z);}
VECATTR real3 make_real3(int3 a){ return make_real3(real(a.x), real(a.y), real(a.z));}
VECATTR real3 make_real3(uint3 a){return make_real3(real(a.x), real(a.y), real(a.z));}

//////////////////REAL2///////////////////////////


VECATTR real2 make_real2(real x, real y){
 #ifdef SINGLE_PRECISION 
  return make_float2(x,y);
  #else
  return make_double2(x,y);
  #endif
}

VECATTR real2 make_real2(real s){ return make_real2(s, s);}
VECATTR real2 make_real2(real2 a){return make_real2(a.x, a.y);}
VECATTR real2 make_real2(real4 a){return make_real2(a.x, a.y);}
VECATTR real2 make_real2(int3 a){ return make_real2(real(a.x), real(a.y));}
VECATTR real2 make_real2(uint3 a){return make_real2(real(a.x), real(a.y));}


////////////////DOUBLE PRECISION//////////////////////
#ifndef SINGLE_PRECISION
VECATTR double3 make_double3(real3 a){return make_double3(a.x, a.y, a.z);}
#endif
VECATTR float4 make_float4(double4 a){return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));}

VECATTR double4 make_double4(double s){ return make_double4(s, s, s, s);}
VECATTR double4 make_double4(double3 a){return make_double4(a.x, a.y, a.z, 0.0f);}
VECATTR double4 make_double4(double3 a, double w){return make_double4(a.x, a.y, a.z, w);}
VECATTR double4 make_double4(int4 a){return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));}
VECATTR double4 make_double4(uint4 a){return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));}
VECATTR double4 make_double4(float4 a){return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));}

//////DOUBLE4///////////////
VECATTR  double4 operator +(const double4 &a, const double4 &b){
  return make_double4(a.x + b.x,
		      a.y + b.y,
		      a.z + b.z,
		      a.w + b.w
		      );
}
VECATTR  void operator +=(double4 &a, const double4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
VECATTR  double4 operator +(const double4 &a, const double &b){
  return make_double4(
		      a.x + b,
		      a.y + b,
		      a.z + b,
		      a.w + b
		      );
}
VECATTR  double4 operator +(const double &b, const double4 &a){
  return make_double4(
		      a.x + b,
		      a.y + b,
		      a.z + b,
		      a.w + b
		      );
}
VECATTR  void operator +=(double4 &a, const double &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

VECATTR  double4 operator -(const double4 &a, const double4 &b){
  return make_double4(
		      a.x - b.x,
		      a.y - b.y,
		      a.z - b.z,
		      a.w - b.w
		      );
}
VECATTR  void operator -=(double4 &a, const double4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
VECATTR  double4 operator -(const double4 &a, const double &b){
  return make_double4(
		      a.x - b,
		      a.y - b,
		      a.z - b,
		      a.w - b
		      );
}
VECATTR  double4 operator -(const double &b, const double4 &a){
  return make_double4(
		      a.x - b,
		      a.y - b,
		      a.z - b,
		      a.w - b
		      );
}
VECATTR  void operator -=(double4 &a, const double &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}
VECATTR  double4 operator *(const double4 &a, const double4 &b){
  return make_double4(
		      a.x * b.x,
		      a.y * b.y,
		      a.z * b.z,
		      a.w * b.w
		      );
}
VECATTR  void operator *=(double4 &a, const double4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
VECATTR  double4 operator *(const double4 &a, const double &b){
  return make_double4(
		      a.x * b,
		      a.y * b,
		      a.z * b,
		      a.w * b
		      );
}
VECATTR  double4 operator *(const double &b, const double4 &a){
  return make_double4(
		      a.x * b,
		      a.y * b,
		      a.z * b,
		      a.w * b
		      );
}
VECATTR  void operator *=(double4 &a, const double &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}
VECATTR  double4 operator /(const double4 &a, const double4 &b){
  return make_double4(
		      a.x / b.x,
		      a.y / b.y,
		      a.z / b.z,
		      a.w / b.w
		      );
}
VECATTR  void operator /=(double4 &a, const double4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
}
VECATTR  double4 operator /(const double4 &a, const double &b){
  return make_double4(
		      a.x / b,
		      a.y / b,
		      a.z / b,
		      a.w / b
		      );
}
VECATTR  double4 operator /(const double &b, const double4 &a){
  return make_double4(
		      b / a.x,
		      b / a.y,
		      b / a.z,
		      b / a.w 
		      );
}
VECATTR  void operator /=(double4 &a, const double &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
}

VECATTR float dot(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
VECATTR float length(double4 v)
{
    return sqrt(dot(v, v));
}
VECATTR double4 normalize(double4 v)
{
  double invLen = 1.0/sqrt(dot(v, v));
  return v * invLen;
}
VECATTR double4 floorf(double4 v)
{
    return make_double4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

/////////////////////DOUBLE3///////////////////////////////

VECATTR int3 make_int3(double3 a){
  return make_int3((int)a.x, (int)a.y, (int)a.z);
}
VECATTR double3 make_double3(double a){
  return make_double3(a, a, a);
}

VECATTR double3 make_double3(int3 a){
  return make_double3(a.x, a.y, a.z);
}
VECATTR double3 make_double3(float3 a){
  return make_double3(a.x, a.y, a.z);
}

VECATTR  double3 operator +(const double3 &a, const double3 &b){
  return make_double3(
		      a.x + b.x,
		      a.y + b.y,
		      a.z + b.z
		      );
}
VECATTR  void operator +=(double3 &a, const double3 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
VECATTR  double3 operator +(const double3 &a, const double &b){
  return make_double3(
		      a.x + b,
		      a.y + b,
		      a.z + b
		      );
}
VECATTR  double3 operator +(const double &b, const double3 &a){
  return make_double3(
		      a.x + b,
		      a.y + b,
		      a.z + b
		      );
}
VECATTR  void operator +=(double3 &a, const double &b){
  a.x += b;
  a.y += b;
  a.z += b;
}

VECATTR  double3 operator -(const double3 &a, const double3 &b){
  return make_double3(
		      a.x - b.x,
		      a.y - b.y,
		      a.z - b.z
		      );
}
VECATTR  void operator -=(double3 &a, const double3 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
VECATTR  double3 operator -(const double3 &a, const double &b){
  return make_double3(
		      a.x - b,
		      a.y - b,
		      a.z - b
		      );
}
VECATTR  double3 operator -(const double &b, const double3 &a){
  return make_double3(
		      a.x - b,
		      a.y - b,
		      a.z - b
		      );
}
VECATTR  void operator -=(double3 &a, const double &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
}
VECATTR  double3 operator *(const double3 &a, const double3 &b){
  return make_double3(
		      a.x * b.x,
		      a.y * b.y,
		      a.z * b.z
		      );
}
VECATTR  void operator *=(double3 &a, const double3 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
VECATTR  double3 operator *(const double3 &a, const double &b){
  return make_double3(
		      a.x * b,
		      a.y * b,
		      a.z * b
		      );
}
VECATTR  double3 operator *(const double &b, const double3 &a){
  return make_double3(
		      a.x * b,
		      a.y * b,
		      a.z * b
		      );
}
VECATTR  void operator *=(double3 &a, const double &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
}
VECATTR  double3 operator /(const double3 &a, const double3 &b){
  return make_double3(
		      a.x / b.x,
		      a.y / b.y,
		      a.z / b.z
		      );
}
VECATTR  void operator /=(double3 &a, const double3 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}
VECATTR  double3 operator /(const double3 &a, const double &b){
  return make_double3(
		      a.x / b,
		      a.y / b,
		      a.z / b
		      );
}
VECATTR  double3 operator /(const double &b, const double3 &a){
  return make_double3(
		      b / a.x,
		      b / a.y,
		      b / a.z
		      );
}
VECATTR  void operator /=(double3 &a, const double &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

//DOUBLE2


VECATTR  double2 operator -(const double2 &a, const double2 &b){
  return make_double2(
		      a.x - b.x,
		      a.y - b.y
		      );
}
VECATTR  void operator -=(double2 &a, const double2 &b){
  a.x -= b.x;
  a.y -= b.y;
}
VECATTR  double2 operator -(const double2 &a, const double &b){
  return make_double2(
		      a.x - b,
		      a.y - b
		      );
}
VECATTR  double2 operator -(const double &b, const double2 &a){
  return make_double2(
		      a.x - b,
		      a.y - b
		      );
}
VECATTR  void operator -=(double2 &a, const double &b){a.x -= b; a.y -= b;}

VECATTR  double2 operator *(const double2 &a, const double2 &b){return make_double2(a.x * b.x, a.y * b.y);}
VECATTR  void operator *=(double2 &a, const double2 &b){ a.x *= b.x; a.y *= b.y;}
VECATTR  double2 operator *(const double2 &a, const double &b){ return make_double2(a.x * b, a.y * b);}
VECATTR  double2 operator *(const double &b, const double2 &a){return make_double2(a.x * b,a.y * b);}
VECATTR  void operator *=(double2 &a, const double &b){a.x *= b; a.y *= b;}


////////////////////////////

VECATTR double3 floorf(double3 v){return make_double3(floor(v.x), floor(v.y), floor(v.z));}
VECATTR float dot(const double3 &a, const double3 &b){return a.x * b.x + a.y * b.y + a.z * b.z;}
VECATTR float length(double3 v){return sqrt(dot(v, v));}
VECATTR double3 normalize(double3 v)
{
  double invLen = 1.0/sqrt(dot(v, v));
  return v * invLen;
}

VECATTR double3 cross(double3 a, double3 b){
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}


//////////////////////////////////////////////////////////


/****************************************************************************************/


///////////INT3/////////////////
VECATTR int3 operator /(int3 a, int3 b){
  return make_int3( a.x/b.x, a.y/b.y, a.z/b.z);
}
