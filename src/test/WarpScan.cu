
//This file tests a bug in cub

#include"../third_party/cub/cub/cub.cuh"



constexpr int tpp = 3;

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 123456
#endif

using WarpScan = cub::WarpScan<int, tpp, 520>;




template<int tpp>
__global__ void test(int N){
  int id = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ typename WarpScan::TempStorage temp_storage[];
  if(id>=N) return;
     
  int my_value  = 1;//threadIdx.x;//(threadIdx.x==1)?1:2;
  
  int my_local_sum = 0;
  int warp_aggregate = 0;
    
    int warp_id = threadIdx.x/tpp;
    __syncthreads();
    WarpScan(temp_storage[warp_id]).InclusiveSum(my_value, my_local_sum, warp_aggregate);
    __syncthreads();
    
#if __CUDA_ARCH__ > 210
     int warpid = threadIdx.x%tpp;
     int delta = tpp-warpid-1;

     warp_aggregate = __shfl_down(my_local_sum, delta, 32);
 #endif
    printf("arch: %d, id: %d - delta: %d - my_value = %d - my_local_sum = %d - warp_aggregate = %d\n",
	   __CUDA_ARCH__, threadIdx.x, delta, my_value, my_local_sum, warp_aggregate);


};




int main(){

  int N = 6;
  test<tpp><<<1, N, N*tpp*sizeof(WarpScan)>>>(N);
  cudaDeviceSynchronize();

};