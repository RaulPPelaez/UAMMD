#ifndef PARAMETERHANDLER_CUH
#define PARAMETERHANDLER_CUH

#include<thrust/device_vector.h>
namespace uammd{
  namespace Potential{


    template<class Functor>
    class BasicParameterHandler{
      using PairParameters = typename Functor::PairParameters;
      thrust::device_vector<PairParameters> pairParameters;
      //cub::TexObjInputIterator<PairParameters> itr;
      int ntypes;
      real cutOff;
      //cub::TexRefInputIterator<PairParameters, 1> itr;

    public:
      BasicParameterHandler():ntypes(1), cutOff(0){}
      ~BasicParameterHandler(){
       	//itr.UnbindTexture();
      }
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

	// itr.UnbindTexture();
	// itr.BindTexture(thrust::raw_pointer_cast(pairParameters.data()), sizeof(PairParameters)*ntypes);
      }

      real getCutOff(){
	return this->cutOff;
      }

      struct Iterator{
	//PairParameters * shMem;
	PairParameters * globalMem;
	//cub::CacheModifiedInputIterator<cub::LOAD_LDG, PairParameters> itr;
	//cub::TexObjInputIterator<PairParameters> itr;
	//cub::TexRefInputIterator<PairParameters, 1> itr;
	int ntypes;
	Iterator(PairParameters * globalMem, int ntypes): globalMem(globalMem), ntypes(ntypes){}
	//Iterator(cub::TexRefInputIterator<PairParameters, 1> itr, int ntypes): itr(itr), ntypes(ntypes){}
	//Iterator(PairParameters * globalMem, int ntypes): itr(globalMem), ntypes(ntypes){}
	//Iterator(cub::TexObjInputIterator<PairParameters> itr, int ntypes): itr(itr), ntypes(ntypes){}
	size_t getSharedMemorySize(){
	  return 0;// sizeof(PairParameters)*ntypes*ntypes;
	}

	inline __device__ void zero(){
	  // extern __shared__  PairParameters aux[];
	  //  if(ntypes==1)
	  //    aux[0] = globalMem[0];
	  //  else
	  //    for(int i=threadIdx.x; i<ntypes*ntypes; i+= blockDim.x)
	  //      aux[i] = globalMem[i];
	  // shMem = aux;
	  //  __syncthreads();
	  //shMem = globalMem;
	}
      inline __device__ PairParameters operator()(int ti, int tj){
	if(ntypes==1) return this->globalMem[0];


	#if CUB_PTX_ARCH < 300
        constexpr auto cubModifier = cub::LOAD_DEFAULT;
	#else
	constexpr auto cubModifier = cub::LOAD_CA;
	#endif

	cub::CacheModifiedInputIterator<cubModifier, PairParameters> itr(globalMem);

	  //if(ntypes==1) return itr[0];

	  if(ti>tj) thrust::swap(ti,tj);

	  int typeIndex = ti+this->ntypes*tj;
	  if(ti >= ntypes || tj >= ntypes) typeIndex = 0;

	   // real4 tmp = __ldg((real4*)this->globalMem+typeIndex);

	   // PairParameters tmp2;
	   // tmp2.cutOff2 = tmp.x;
	   // tmp2.sigma2 = tmp.y;
	   // tmp2.epsilonDivSigma2= tmp.z;
	   // tmp2.shift = tmp.w;

	   // return tmp2;

	  //return this->globalMem[typeIndex];
	  return itr[typeIndex];
	}

      };

      Iterator getIterator(){
	auto tp = thrust::raw_pointer_cast(pairParameters.data());
	return Iterator(tp, ntypes);
	//return Iterator(itr, ntypes);
      }

    };
  }
}
#endif
