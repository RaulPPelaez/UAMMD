#ifndef MISC_IBM_UTILS_CUH
#define MISC_IBM_UTILS_CUH

#include"utils/utils.h"
namespace uammd{
  namespace IBM_ns{
    namespace detail{
      SFINAE_DEFINE_HAS_MEMBER(getSupport)
      template<class Kernel, bool def = has_getSupport<Kernel>::value> struct GetSupport;
      template<class Kernel> struct GetSupport<Kernel, true>{
	static __host__  __device__ int3 get(Kernel &kernel, real3 pos, int3 cell){return kernel.getSupport(pos, cell);}
      };
      template<class Kernel> struct GetSupport<Kernel, false>{
	static __host__  __device__ int3 get(Kernel &kernel, real3 pos, int3 cell){return make_int3(kernel.support);}
      };

      SFINAE_DEFINE_HAS_MEMBER(getMaxSupport)
      template<class Kernel, bool def = has_getMaxSupport<Kernel>::value> struct GetMaxSupport;
      template<class Kernel> struct GetMaxSupport<Kernel, true>{
	static __host__  __device__ int3 get(Kernel &kernel){return kernel.getMaxSupport();}
      };
      template<class Kernel> struct GetMaxSupport<Kernel, false>{
	static __host__  __device__ int3 get(Kernel &kernel){return make_int3(kernel.support);}
      };

      SFINAE_DEFINE_HAS_MEMBER(phiX)
      SFINAE_DEFINE_HAS_MEMBER(phiY)
      SFINAE_DEFINE_HAS_MEMBER(phiZ)
#define ENABLE_PHI_IF_HAS(foo) template<class Kernel> __device__ inline SFINAE::enable_if_t<has_phi##foo<Kernel>::value, real>
#define ENABLE_PHI_IF_NOT_HAS(foo) template<class Kernel> __device__ inline SFINAE::enable_if_t<not has_phi##foo<Kernel>::value, real>
      ENABLE_PHI_IF_HAS(X) phiX(Kernel &kern, real r, real3 pos){return kern.phiX(r, pos);}
      ENABLE_PHI_IF_HAS(Y) phiY(Kernel &kern, real r, real3 pos){return kern.phiY(r, pos);}
      ENABLE_PHI_IF_HAS(Z) phiZ(Kernel &kern, real r, real3 pos){return kern.phiZ(r, pos);}
      ENABLE_PHI_IF_NOT_HAS(X) phiX(Kernel &kern, real r, real3 pos){return kern.phi(r, pos);}
      ENABLE_PHI_IF_NOT_HAS(Y) phiY(Kernel &kern, real r, real3 pos){return kern.phi(r, pos);}
      ENABLE_PHI_IF_NOT_HAS(Z) phiZ(Kernel &kern, real r, real3 pos){return kern.phi(r, pos);}
  }
  }
}
    #endif
