/* Raul P. Pelaez 2017.
   Some basic defines and aliases for uammd
 */
#ifndef UAMMD_DEFINES_H
#define UAMMD_DEFINES_H

#define UAMMD_VERSION "0.5.alpha"

/*Comment to compile for double precision*/
#define SINGLE_PRECISION


#define fori(x,y) for(int i=x; i<int(y); i++)
#define forj(x,y) for(int j=x; j<int(y); j++)
namespace uammd{

#if defined SINGLE_PRECISION
using  real  = float;
using  real2 = float2;
using  real3 = float3;
using  real4 = float4;
#else
using  real  = double;
using  real2 = double2;
using  real3 = double3;
using  real4 = double4;

#endif
}
#endif
