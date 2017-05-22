#include "Sorter.cuh"
#include<cub/cub.cuh>

namespace Sorter{

  void sortByKey(uint *&index, uint *&index_alt, uint *&hash, uint *&hash_alt, int N, cudaStream_t st){
    //This uses the CUB API to perform a radix sort
    //CUB orders by key an array pair and copies them onto another pair
    //This function switches the arrays with _alt each time its called
    static bool init = false;
    static void *d_temp_storage = NULL;
    static size_t temp_storage_bytes = 0; //Additional storage needed by cub

    /**Initialize CUB at first call**/
    if(!init){
      /*Allocate temporal value/key pair*/
      gpuErrchk(cudaMalloc(&hash_alt,  N*sizeof(uint)));
      gpuErrchk(cudaMalloc(&index_alt, N*sizeof(uint)));    
      /*On first call, this function only computes the size of the required temporal storage*/
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				      hash, hash_alt,
				      index, index_alt,
				      N,
				      0, sizeof(uint)*8,st);
      /*Allocate temporary storage*/
      gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      init = true;
    }

    /**Perform the Radix sort on the index/hash pair**/
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				    hash, hash_alt,
				    index, index_alt,
				    N,
				    0, sizeof(uint)*8,st);

    /**Swap the references**/
    std::swap(hash, hash_alt);
    std::swap(index, index_alt);

    // thrust::stable_sort_by_key(device_ptr<uint>(hash),
    // 			device_ptr<uint>(hash+N),
    // 			device_ptr<uint>(index));
    //cudaCheckErrors("Sort hash");					  


  }


}