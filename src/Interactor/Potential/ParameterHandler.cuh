#ifndef PARAMETERHANDLER_CUH
#define PARAMETERHANDLER_CUH
#include"global/defines.h"
#include<thrust/device_vector.h>
namespace uammd{
  namespace Potential{


    template<class Functor>
    class BasicParameterHandler{
      using PairParameters = typename Functor::PairParameters;
      thrust::device_vector<PairParameters> pairParameters;
      int ntypes;
      real cutOff;
    public:
      BasicParameterHandler():ntypes(1), cutOff(0){}

      ~BasicParameterHandler(){}

      void add(int ti, int tj, typename Functor::InputPairParameters p){
	this->cutOff = std::max(p.cutOff, this->cutOff);
	int new_ntypes = ntypes;
	if(ti >= ntypes) new_ntypes = ti+1;
	if(tj >= ntypes) new_ntypes = tj+1;
	pairParameters.resize(new_ntypes*new_ntypes);
	if(new_ntypes != ntypes){
	  auto tmp = pairParameters;
	  fori(0,ntypes)
	    forj(0,ntypes){
	    pairParameters[i+new_ntypes*j] = tmp[i+ntypes*j];
	  }
	  ntypes = new_ntypes;
	}
	pairParameters[ti+ntypes*tj] = Functor::processPairParameters(p);
	if(ti != tj)
	  pairParameters[tj+ntypes*ti] = Functor::processPairParameters(p);
      }

      real getCutOff(){
	return this->cutOff;
      }

      struct Iterator{
	PairParameters * globalMem;
	int ntypes;

	Iterator(PairParameters * globalMem, int ntypes): globalMem(globalMem), ntypes(ntypes){}

	inline __device__ PairParameters operator()(int ti, int tj){
	  if(ntypes==1) return this->globalMem[0];
	  cub::CacheModifiedInputIterator<cub::LOAD_CA, PairParameters> itr(globalMem);
	  if(ti>tj) thrust::swap(ti,tj);
	  int typeIndex = ti+this->ntypes*tj;
	  if(ti >= ntypes || tj >= ntypes) typeIndex = 0;
	  return itr[typeIndex];
	}

      };

      Iterator getIterator(){
	auto tp = thrust::raw_pointer_cast(pairParameters.data());
	return Iterator(tp, ntypes);
      }

    };
  }
}
#endif
