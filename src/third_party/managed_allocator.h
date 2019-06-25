// Raul P. Pelaez 2019. Thrust Managed  memory allocator, adapted from devtalk nvidia:
// https://devtalk.nvidia.com/default/topic/987577/-thrust-is-there-a-managed_vector-with-unified-memory-do-we-still-only-have-device_vector-cuda-thrust-managed-vectors-/
#include <thrust/device_vector.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/device_malloc_allocator.h>
namespace uammd{
  template<class T>
  class managed_allocator : public thrust::device_malloc_allocator<T>{
  public:
    using value_type = T;
    typedef thrust::device_ptr<T>  pointer;
    inline pointer allocate(size_t n){
      value_type* result = nullptr;      
      cudaError_t error = cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal);      
      if(error != cudaSuccess)
	throw thrust::system_error(error, thrust::cuda_category(),
				   "managed_allocator::allocate(): cudaMallocManaged");           
      return thrust::device_pointer_cast(result);
    }
    
    inline void deallocate(pointer ptr, size_t){
      cudaError_t error = cudaFree(thrust::raw_pointer_cast(ptr));
      if(error != cudaSuccess)
	throw thrust::system_error(error, thrust::cuda_category(),
				   "managed_allocator::deallocate(): cudaFree");	
    }
  };

  template<class T>
  using managed_vector = thrust::device_vector<T, managed_allocator<T>>;
}
