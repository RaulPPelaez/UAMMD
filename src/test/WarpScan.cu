
//This file tests a bug in cub

#include"../third_party/cub/cub.cuh"



constexpr int tpp = 2;

template<int tpp>
__global__ void test(int N){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  using WarpScan = cub::WarpScan<int, tpp>;
  
  __shared__ typename WarpScan::TempStorage temp_storage[tpp];
  if(id>=N) return;
     
  int my_value  = threadIdx.x;//(threadIdx.x==1)?1:2;
  
  int my_local_sum = 0;
  int warp_aggregate = 0;
    
    int warp_id = threadIdx.x/tpp;
    __syncthreads();
    WarpScan(temp_storage[warp_id]).InclusiveSum(my_value, my_local_sum, warp_aggregate);
    __syncthreads();
    
    // #if __CUDA_ARCH__ > 210
    int warpid = threadIdx.x%tpp;
    int delta = tpp-warpid-1;

    warp_aggregate = __shfl_down(my_local_sum, delta, 32);
//  #endif
    #ifdef __CUDA_ARCH__
    printf("arch: %d, id: %d - delta: %d - my_value = %d - my_local_sum = %d - warp_aggregate = %d\n",
	   __CUDA_ARCH__, threadIdx.x, delta, my_value, my_local_sum, warp_aggregate);
    #endif

};




int main(){

  int N = 6;
  test<tpp><<<1, N>>>(N);
  cudaDeviceSynchronize();

};