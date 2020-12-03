/*Raul P. Pelaez 2019. Boundary Value Problem Memory layout and access handler
 */
#ifndef BVPMEMORY_CUH
#define BVPMEMORY_CUH

#include<thrust/iterator/permutation_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/counting_iterator.h>

namespace uammd{
  namespace BVP{

    template<class T>
    struct StorageHandle{
      using value_type = T;
      using pointer = T*;
      size_t offset;
      int numberElements;
    };

    namespace detail{
      size_t computeExtraAlignment(size_t addressOffset, size_t alignment){
	size_t extraAlignment = 0;
	bool isMissAligned = addressOffset%alignment != 0;
	if(isMissAligned)
	  extraAlignment += alignment - addressOffset%alignment;
	return extraAlignment;
      }
    }

    class StorageRegistration{
      size_t currentOffset = 0;
      size_t allocationSize = 0;
      int numberCopies;

    public:

      StorageRegistration(int numberCopies):numberCopies(numberCopies){}

      template<class T>
      StorageHandle<T> registerStorageRequirement(int numberElements){
	const size_t extraAlignment = detail::computeExtraAlignment(currentOffset, sizeof(T));
	StorageHandle<T> stor;
	stor.offset = currentOffset + extraAlignment;
	stor.numberElements = numberElements;
	const size_t requestedAllocationSize = (numberCopies*numberElements)*sizeof(T);
	allocationSize += requestedAllocationSize + extraAlignment;
	currentOffset += requestedAllocationSize + extraAlignment;
	return stor;
      }

      size_t getRequestedStorageBytes(){
	return allocationSize;
      }

    };

    namespace layout{
      class Contigous{
	int first;
      public:
	__host__ __device__ Contigous(int numberElements, int numberSystems, int instance){
	  first = instance*numberElements;
	}

	__device__ __host__ int operator()(int i){
	  return first+i;
	};

      };

      class Strided{
	int stride;
	int first;
      public:
	__host__ __device__ Strided(int numberElements, int numberSystems, int instance){
	  first = instance;
	  stride = numberSystems;
	}

	__device__ __host__ int operator()(int i){
	  return first+stride*i;
	};

      };
    }

    using MemoryLayout = layout::Strided;

    using CountingIterator = thrust::counting_iterator<int>;
    using MemoryAccessTransformIterator = thrust::transform_iterator<MemoryLayout, CountingIterator>;

    template<class T>
    using Iterator = thrust::permutation_iterator<typename StorageHandle<T>::pointer, MemoryAccessTransformIterator>;

    template<class T>
    __device__ __host__ Iterator<T> make_iterator(char* raw, const StorageHandle<T> &stor,
						  int numberSystems, int id){
      auto ptr = reinterpret_cast<typename StorageHandle<T>::pointer>(raw + stor.offset);
      auto layoutTransform = MemoryLayout(stor.numberElements, numberSystems, id);
      auto indexTransform = MemoryAccessTransformIterator(CountingIterator(0), layoutTransform);
      return thrust::make_permutation_iterator(ptr, indexTransform);
    }

    class StorageRetriever{
      char* raw = nullptr;
      int numberCopies;
      int instance;
    public:

      __host__ __device__ StorageRetriever(int numberCopies, int instance, char* ptr):
	raw(ptr), numberCopies(numberCopies), instance(instance){}

      template<class T>
      __device__ __host__ Iterator<T> retrieveStorage(const StorageHandle<T> &stor){
	return make_iterator<T>(raw, stor, numberCopies, instance);
      }

    };

  }
}
#endif