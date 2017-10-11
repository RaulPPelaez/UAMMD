//This file tests a bug in cub
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/extrema.h>
#include<thrust/sort.h>
#include<cub/cub.cuh>


#include<iostream>

#include<cmath>


#include<limits>

using namespace std;


int main(){

  //Number of elements to sort
  int N = 1<<12;
  //Keys will go from max_key to zero
  uint max_key = 1<<26;
  //Most significant bit
  //msb must be that of the max between N and max_key!! WHY??
  int msb = int(std::log2(std::max((uint)N, max_key)+0.5));
  //Number of extra bits to sum to end_bit
  int cub_extra_bits = 1;
  int end_bit = (msb+cub_extra_bits);
  end_bit = std::min(end_bit, 32);

  
  cerr<<"Number of elements: "<<N<<endl;
  thrust::device_vector<uint> key(N), key_alt(N);
  thrust::device_vector<int> value(N), value_alt(N);



  auto db_value = cub::DoubleBuffer<int>(thrust::raw_pointer_cast(value.data()),
					 thrust::raw_pointer_cast(value_alt.data()));
  auto db_key  = cub::DoubleBuffer<uint>(thrust::raw_pointer_cast(key.data()),
					  thrust::raw_pointer_cast(key_alt.data()));

  //Fill keys with max_key...0 and values with 0...N-1, so that after sorting values are N-1...0
  {
    thrust::host_vector<int> valueCPU(N);
    thrust::host_vector<int> keyCPU(N);
  
    for(int i = 0; i<N; i++){
      valueCPU[i]  = i;

      keyCPU[i] = ((N-i-1)/(double)N)*(max_key);
    }
    value = valueCPU;
    key = keyCPU;
  }
  //Correct sort with thrust
  auto value_thrust = value;
  auto key_thrust = key;
  thrust::stable_sort_by_key(key_thrust.begin(), key_thrust.end(), value_thrust.begin());  


  //Try to sort with cub, select end_bit as the msb of the largest hash + cub_extra_bits  
  int min_key = *(thrust::min_element(key.begin(), key.end()));

  
  cerr<<"Most significant bit in max key value ("<<max_key<<"): "<<msb<<endl;
  cerr<<"Min key: "<<min_key<<endl;

  cerr<<"end_bit to cub::SortPairs: "<<end_bit<<endl;

  
  size_t temp_storage_bytes = 0;
  void * d_temp_storage = nullptr;

  /*On first call, this function only computes the size of the required temporal storage*/
  auto err = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				  db_key,
				  db_value,
				  N,
				  0, end_bit);
			
  /*Allocate temporary storage*/
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cerr<<temp_storage_bytes<<" bytes Allocated for cub"<<endl;
  /**Perform the Radix sort on the value/key pair**/
  err = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				  db_key, 
				  db_value,
				  N,
				  0, end_bit);

  key.swap(key_alt);
  value.swap(value_alt);
  
  cudaFree(d_temp_storage);

  {
    thrust::host_vector<int> valueCPU = key;
    thrust::host_vector<int> value_thrustCPU = key_thrust;
    //Check result
    for(int i = 0; i<20; i++){
      cerr<<valueCPU[i]<<" ";
    }
    cerr<<endl;
    for(int i = 0; i<20; i++){
      cerr<<value_thrustCPU[i]<<" ";
    }
    cerr<<endl;
    
    for(int i = 0; i<N; i++){
      if(valueCPU[i] != value_thrustCPU[i]){
	cerr<<endl<<" ERROR in "<<i<<"th element!!"<<endl;
	cerr<<"Result is: "<<valueCPU[i]<<endl;
	cerr<<"Should be:"<<value_thrustCPU[i]<<endl;	
	exit(1);
      }
    }
  } 
  cerr<<"SUCCESS!!"<<endl;
  
  return 0;
}