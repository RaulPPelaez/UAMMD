/* Raul P. Pelaez. 2022
   Definition of the complex type and related functions.

 */
#ifndef UAMMD_COMPLEX_CUH
#define UAMMD_COMPLEX_CUH
#include "global/defines.h"
#include <thrust/complex.h>
namespace uammd {
template <class T> using complex_t = thrust::complex<T>;

// Currently the complex type is just an alias to the one provided by thrust,
// which is compatible with
//  the std one but works in device functions
using complex = thrust::complex<real>;
} // namespace uammd

#endif
