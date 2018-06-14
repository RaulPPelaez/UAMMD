/* Raul P. Pelaez 2017. ParticleSorter.
   
   A helper class to sort particles according to their positions following a certain rule.
   This rule can be a morton hash, so the particle positions are sorted to follow a Z-order curve, a cell hash, particle ID...


   USAGE:
   This class is meant to be used by ParticleData, but can be used to sort from others.

   //Create an instance of ParticleSorter:
   ParticleSorter ps;
   //Update/create a sorted index list from the positions using a certain hash
   ps.updateOrderByHash<Sorter::MortonHash>(pos, numberParticles, BoxSize, cudaStream);
   //Apply new order to some array
   ps.applyCurrentOrder(array_in, array_out, numberParticles, cudaStream);

   //Sort a key/value pair list with cub
   ps.sortByKey(key_dobleBuffer, value_dobleBuffer, numberParticles, cudaStream);

   //Order the id array (without changing it) as keys and a 0..numberparticles array as values,
   //return this device array. It does not affect the current order.
   ps.getIndexArrayById(id, numberParticles, cudaStream);
   TODO:
   100-More hashes

REFERENCES:
[1] https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
*/
#ifndef PARTICLESORTER_CUH
#define PARTICLESORTER_CUH

#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"System/System.h"
#include"utils/debugTools.cuh"
#include<third_party/cub/cub.cuh>

namespace uammd{
  
  namespace Sorter{
    
    struct MortonHash{
      //Interleave a 10 bit number in 32 bits, fill one bit and leave the other 2 as zeros. See [1]
      static inline __host__ __device__ uint encodeMorton(const uint &i){
	uint x = i;
	x &= 0x3ff;
	x = (x | x << 16) & 0x30000ff;
	x = (x | x << 8)  & 0x300f00f;
	x = (x | x << 4)  & 0x30c30c3;
	x = (x | x << 2)  & 0x9249249;
	return x;
      }
      /*Fuse three 10 bit numbers in 32 bits, producing a Z order Morton hash*/
      static inline __host__ __device__ uint hash(const int3 &cell, const Grid &grid){
	return encodeMorton(cell.x) | (encodeMorton(cell.y) << 1) | (encodeMorton(cell.z) << 2);
      }      
    };
    //The hash is the cell 1D index, this pattern is better than random for neighbour transverse, but worse than Morton
    struct CellHash{
      static inline __device__ __host__ uint hash(const int3 &cell, const Grid &grid){
	return cell.x + cell.y*grid.cellDim.x + cell.z*grid.cellDim.x*grid.cellDim.z;
      }      
    };
  
    /*Assign a hash to each particle*/
    template<class HashComputer = MortonHash, class InputIterator>
    __global__ void computeHash(InputIterator pos,
				int* __restrict__ index,
				uint* __restrict__ hash , int N,
				Grid grid){
      const int i = blockIdx.x*blockDim.x + threadIdx.x;  
      if(i>=N) return;
      const real3 p = make_real3(pos[i]);
    
      const int3 cell = grid.getCell(p);
      /*The particleIndex array will be sorted by the hashes, any order will work*/
      const uint ihash = HashComputer::hash(cell, grid);
      /*Before ordering by hash the index in the array is the index itself*/
      index[i] = i;
      hash[i]  = ihash;
    }


    
    /*In case old position is a texture*/
    template<class InputIterator, class OutputIterator>
    __global__ void reorderArray(const InputIterator old,
				 OutputIterator sorted,
				 int* __restrict__ pindex, int N){
      int i = blockIdx.x*blockDim.x + threadIdx.x;   
      if(i>=N) return;
      sorted[i] = old[pindex[i]];
    }

  }

  class ParticleSorter{
    bool init = false;
    bool originalOrderNeedsUpdate = true;
    void *d_temp_storage = nullptr;
    int temp_storage_num_elements = 0;
    size_t temp_storage_bytes = 0; //Additional storage needed by cub
    thrust::device_vector<int>  original_index;
    thrust::device_vector<int>  index, index_alt;
    thrust::device_vector<uint> hash, hash_alt; 
    /*Radix sort by key using cub, puts sorted versions of index,hash in index_alt, hash_alt*/
  public: 
    ParticleSorter() = default;
    ~ParticleSorter(){
      CudaSafeCall(cudaFree(d_temp_storage));
    }
    template<class hashType>
    void sortByKey(cub::DoubleBuffer<int> &index,
		   cub::DoubleBuffer<hashType> &hash,
		   int N, cudaStream_t st = 0, int end_bit = sizeof(uint)*8){


      //This uses the CUB API to perform a radix sort
      //CUB orders by key an array pair and copies them onto another pair
    
      /**Initialize CUB if more temp storage is needed**/
      if(N > temp_storage_num_elements){
	temp_storage_num_elements = N;
	CudaSafeCall(cudaFree(d_temp_storage));
	temp_storage_bytes = 0;
	d_temp_storage = nullptr;
	/*On first call, this function only computes the size of the required temporal storage*/
	CudaSafeCall(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
						     hash,
						     index,
						     N,
						     0, end_bit,
						     st));
			
	/*Allocate temporary storage*/
	CudaSafeCall(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      }

      /**Perform the Radix sort on the index/hash pair**/
      CudaSafeCall(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
						   hash, 
						   index,
						   N,
						   0, end_bit,
						   st));

      

    }
    //Return the most significant bit of an unsigned integral type
    template <typename T> inline int msb(T n){
      static_assert(std::is_integral<T>::value && !std::is_signed<T>::value,
		    "msb<T>(): T must be an unsigned integral type.");

      for (T i = std::numeric_limits<T>::digits - 1, mask = 1 << i;
	   i >= 0;
	   --i, mask >>= 1){
	if((n & mask) != 0) return i;
	}
      return 0;
    }
    
    template<class HashType = Sorter::MortonHash, class InputIterator>
    void updateOrderByCellHash(InputIterator pos, uint N, Box box, int3 cellDim, cudaStream_t st = 0){
      init = true;
      if(hash.size() != N){hash.resize(N); hash_alt.resize(N);}
      if(index.size()!= N){index.resize(N); index_alt.resize(N);}

      
      int Nthreads=128;
      int Nblocks=N/Nthreads + ((N%Nthreads)?1:0);
      Grid grid(box, cellDim);
      Sorter::computeHash<HashType><<<Nblocks, Nthreads, 0, st>>>(pos,
								  thrust::raw_pointer_cast(index.data()),
								  thrust::raw_pointer_cast(hash.data()),
								  N,
								  grid);
      auto db_index = cub::DoubleBuffer<int>(
					     thrust::raw_pointer_cast(index.data()),
					     thrust::raw_pointer_cast(index_alt.data()));
      auto db_hash  = cub::DoubleBuffer<uint>(
					      thrust::raw_pointer_cast(hash.data()),
					      thrust::raw_pointer_cast(hash_alt.data()));

      uint maxHash = HashType::hash(cellDim, grid);

      //Cub just needs this endbit at least
      int maxbit = int(std::log2(maxHash)+0.5)+1;
      maxbit = std::min(maxbit, 32);
      
      this->sortByKey(db_index,
		      db_hash,
		      N,
		      st, maxbit);
      //Sometimes CUB will not swap the references in the DoubleBuffer
      if(db_index.selector)
	index.swap(index_alt);
      if(db_hash.selector)
	hash.swap(hash_alt);
      
      originalOrderNeedsUpdate = true;
    }

    //Reorder with a custom key, store in orginal_index
    void updateOrderById(int *id, int N, cudaStream_t st = 0){
      int lastN = original_index.size();
      if(lastN != N){
	original_index.resize(N);	
      }
      cub::CountingInputIterator<int> ci(0);
      try{
	thrust::copy(ci, ci+N, original_index.begin());       
      }
      catch(thrust::system_error &e){
	fprintf(stderr,"[ParticleSorter] Thrust could not copy ID vector. Error: %s", e.what());
	exit(1);
      }
	
      
      auto db_index = cub::DoubleBuffer<int>(
					     thrust::raw_pointer_cast(original_index.data()),
					     thrust::raw_pointer_cast(index_alt.data()));
      //store current index in hash

      //thrust::copy will assume cpu copy if the first argument is a raw pointer
      
      int* d_hash = (int*)thrust::raw_pointer_cast(hash.data());
      CudaSafeCall(cudaMemcpy(d_hash, id, N*sizeof(int), cudaMemcpyDeviceToDevice));

      auto db_hash  = cub::DoubleBuffer<int>(
					     d_hash,
					     (int*)thrust::raw_pointer_cast(hash_alt.data())); 
      this->sortByKey(db_index,
		      db_hash,
		      N,
		      st);

      original_index.swap(index_alt);      
    }
    //WARNING: _unsorted and _sorted cannot be aliased!
    template<class InputIterator, class OutputIterator>
    void applyCurrentOrder(InputIterator d_property_unsorted,
			   OutputIterator d_property_sorted,
			   int N, cudaStream_t st = 0){
      int Nthreads=128;
      int Nblocks=N/Nthreads + ((N%Nthreads)?1:0);

      Sorter::reorderArray<<<Nblocks, Nthreads, 0, st>>>(d_property_unsorted,
							 d_property_sorted,
							 thrust::raw_pointer_cast(index.data()),
							 N);
    }

    //Get current order keys
    int * getSortedIndexArray(int N){
      int lastN = index.size();
      
      if(lastN != N){
	cub::CountingInputIterator<int> ci(lastN);
        index.resize(N);
	try{
	  thrust::copy(ci, ci+(N-lastN), index.begin()+lastN);
	}
	catch(thrust::system_error &e){
	  fprintf(stderr, "[ParticleSorter] Thrust failed in %s(%d). Error: %s", __FILE__, __LINE__, e.what());
	  exit(1);
	}

      }

      return thrust::raw_pointer_cast(index.data());
    }

    //Update and return reorder with a custom key
    int * getIndexArrayById(int * id, int N, cudaStream_t st = 0){
      if(!init) return nullptr;
      if(originalOrderNeedsUpdate){
	this->updateOrderById(id, N, st);
	originalOrderNeedsUpdate = false;
      }
      
      int lastN = original_index.size();
      
      if(lastN != N){
	original_index.resize(N);
	try{
	  thrust::copy(id, id+(N-lastN), original_index.begin()+lastN);
	}
	catch(thrust::system_error &e){
	  fprintf(stderr, "[ParticleSorter] Thrust failed in %s(%d). Error: %s", __FILE__, __LINE__, e.what());
	  exit(1);
	}
      }
      return thrust::raw_pointer_cast(original_index.data());
    }
  };
}
#endif
