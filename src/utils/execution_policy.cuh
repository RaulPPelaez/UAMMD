/* Raul P. Pelaez 2022. Custom execution policy for UAMMD.

   Inherits thrust::cuda::par_t, allowing to specify a cuda stream.

   When used on a thrust algorithm this policy will make thrust leverage UAMMD's cached allocator
    meachanism.
   This is useful, for instance, when calling algorithms that require memory allocation such as thrust::sort

 */
#ifndef UAMMDEXECUTIONPOLICY_CUH
#define UAMMDEXECUTIONPOLICY_CUH

#include<System/System.h>
#include <memory>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory.h>

namespace uammd{

  namespace detail{
    static const auto cached_device_execution_policy = thrust::device(System::allocator_thrust<char>());
  }

  using detail::cached_device_execution_policy;

}
#endif
