/*Raul P. Pealez 2022.
  Usage example of the GPU execution policy exposed by UAMMD.

  It mimics thrust::cuda::par, but leveraging the UAMMD's cached allocator.

  We will use it to demonstrate how to call thrust::sort without having to worry about the overhead
   incurred by its required temporal memory allocation
 */

#include"utils/execution_policy.cuh"
#include<thrust/device_vector.h>
#include<thrust/sort.h>
#include<thrust/sequence.h>
#include <sstream>
#include <cuda_profiler_api.h>
//Lets fill a vector with a decrasing sequence of number
auto get_unsorted_vector(int n){
  thrust::device_vector<int> vec(n);
  thrust::sequence(vec.rbegin(), vec.rend(), 0);
  return vec;
}

template<class Vector>
void print_first_elements(Vector &vec, int n = 3){
   thrust::copy_n(vec.begin(), 3, std::ostream_iterator<int>(std::cout," "));
   std::cout<<std::endl;
}

int main(){

  int n = 16384;
  //Lets fill a vector with a decrasing sequence of numbers
  auto vec = get_unsorted_vector(n);
  print_first_elements(vec);
  //We can use thrust::sort to sort it in increasing order
  thrust::sort(vec.begin(), vec.end());
  print_first_elements(vec);
  //However, sort requires allocating some temporary memory as part of the computation.
  //If this sorting is required a lot of times memory allocation will become a non-neglegible part of the time.
  //Moreover, the fact that memory allocation happens in the default stream would impose a synchronization barrier.
  //We can use uammd's execution policy to mitigate this:
  vec = get_unsorted_vector(n); //Unsorted vector again
  cudaStream_t st; cudaStreamCreate(&st);
  for(int i = 0;i<10;i++){
    if(i==1) cudaProfilerStart();
    thrust::sort(uammd::cached_device_execution_policy.on(st), vec.begin(), vec.end());
  }
  cudaProfilerStop();
  cudaStreamDestroy(st);
  print_first_elements(vec);
  //If you inspect the execution of this second batch of calls to sort you will see that calls to cudaMalloc/cudaFree
  // only happen during the first call (in fact cudaFree will never be called).
  //Moreover, all kernels will run in the stream "st", without synchronization barriers.
  //You can check this by using nsys profile, for instance:
  //If you run $ nsys profile -c cudaProfilerApi --stats=true ./execution_policy
  //Inspecting the part of the output about cuda api calls you will see something similar to:
  // Time (%)  Total Time (ns)  Num Calls  Avg (ns)          Name
  //  --------  ---------------  ---------  --------  ---------------------
  //      51.2          399,468        144   2,774.1  cudaLaunchKernel
  //      48.8          380,612         18  21,145.1  cudaStreamSynchronize
  //However, if you remove uammds execution policy and run again:
  //  Time (%)  Total Time (ns)  Num Calls  Avg (ns)  StdDev (ns)          Name
  // --------  ---------------  ---------  --------  -----------  ---------------------
  //     54.7          499,330         20  24,966.5     25,204.9  cudaStreamSynchronize
  //     40.6          370,021        160   2,312.6      1,957.3  cudaLaunchKernel
  //      3.1           27,919         10   2,791.9      4,715.2  cudaMalloc
  //      1.7           15,071         10   1,507.1      1,046.1  cudaFree
  //Granted, not a miraculous improvement in this case, take it as a proof of concept.
  return 0;
}
