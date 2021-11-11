/* Raul P. Pelaez 2021
   Numeric types in UAMMD.
   In this example we will work with uammd vector types used throughout the code.  
   GPU's are really fast with single precision arithmetics and really take a hit when dealing with double precision.
   However, sometimes double precision is needed. 
   UAMMD offers a generic type called "real" that can work as a double or a float via a compiler flag.
   By default UAMMD makes real = float. Try uncommenting the line with DOUBLE_PRECISION in the Makefile, which will make UAMMD define real as double.
   Additionally uammd offers a series of convenient vector types packing several numbers and overloading all arithmetic operations for them.
    
   By using these vector types we can increase chances of SIMD compiler vectorization in the CPU.
   Alas, GPUs are SIMT machines and do not take advantage of vectorized operations per se, but they are still advantageous for memory loading/storing purposes.
   CUDA exposes a few ones, like float2, double2 or float4, which are just small structs defined like:
   struct float4{float x,y,z,w;};
   The main thing to take into account is that CUDA can load/store a float4 or the price of a simple float.
   In particle simulations we can take advantage of this by defining highly used properties like particle positions or forces inside a float4 (even if that means discarding the .w element).
   UAMMD supercharges these native CUDA types and extends them by defining almost every arithmetic operator on them as you may need.
   Additionally, the types real2, real3 and real4 are exposed grouping float/double numbers.
 */

//uammd.cuh is the basic uammd include containing, among other things, the System struct.
#include<uammd.cuh>
#include"utils/vector.cuh" //Contains all vector type arithmetics, it is included by every other uammd include, but lets leave it here so you can know where to look for them
#include<thrust/device_vector.h>
using namespace uammd;

int main(int argc, char* argv[]){
  //Initialize System
  auto sys = std::make_shared<System>(argc, argv);

  //Lets declare a simple floating point value
  //Writing either 1.0f or 1.0 results in code with data type conversions depending on the compilation precision mode
  //This might not be a problem in the CPU, but the GPU really takes a hit from these conversions between double and float.
  //Wrapping values in the "real" type compiles to code without conversions, in single precision real(1.0) will be 1.0f.
  real value = real(1.0);
  //Lets now create 4 packed values
  real4 fourValues = real4(); //Holds {0,0,0,0} because of C++ default initialization
  //There are several ways to initialize these objects
  //These two are equivalent
  fourValues = {1,2,3,4};
  fourValues = make_real4(1,2,3,4);
  auto otherFourValue = fourValues; //{1,2,3,4}
  //For example lets now create a real3:
  real3 threeValues = {1,2,3};
  otherFourValue = make_real4(threeValues); //{1,2,3,0}
  otherFourValue = make_real4(threeValues, 4); //{1,2,3,4}
  //You will find this kind of behavior when working with these types:
  threeValues = make_real3(make_real4(1,2,3,4)); //{1,2,3} <- The last value is cropped
  //You can access individual elements with .x .y .z .w:
  sys->log<System::MESSAGE>("Contents of fourValues: %g %g %g %g", fourValues.x, fourValues.y, fourValues.z, fourValues.w);
  //It is also posisble to do arithmetic operations on them:
  const real4 fourOnes = {1,1,1,1};
  auto fourTwos = fourOnes + fourOnes; //{2,2,2,2}
  //Or:
  fourTwos = 2*fourOnes;
  //Or:
  fourTwos = make_real4(0,0,0,0) + 2;
  //Or:
  fourTwos = {0,0,0,0};
  fourTwos += 2;
  //You get the idea, there is also int2, int3, int4, uint2, double4, etc etc.
  //We will see later that UAMMD defines particle properties using these types.
  //Theres also the possibility of reading/writing these types. For example:
  std::cout<<"Contents of fourValues:"<<fourValues<<std::endl;
  //Now that we are here, lets create a gpu vector of one of these types and fill it with something:
  int numberElements = 10000;
  thrust::device_vector<real4> positions(numberElements);  
  thrust::fill(positions.begin(), positions.end(), make_real4(1,2,3,4));
  //Now positions is a vector of 10000 elements with {1,2,3,4} in all its elements;
  //UAMMD makes extensive use of the CUDA library thrust and thus the std C++ library.
  //If these last lines do not make sense I highly suggest going though a basic thrust tutorial now:
  //https://docs.nvidia.com/cuda/thrust/index.html
  //If you are familiar with the standard C++ library you will find thrust to be clonical to it.
  //The equivalent CUDA/C code to what you see above is quite uglier and unsafe, in case you do not trust me:
  // std::vector<real4> host_positions(numberElements);
  // std::fill(host_positions.begin(), host_positions.end(), make_real4(1,2,3,4));
  // real4* positions = nullptr;
  // cudaMalloc(&positions, numberElements*sizeof(real4));
  // cudaMemcpy(positions, host_positions.data(), numberElements*sizeof(real4), cudaMemcpyHostToDevice);
  //Dont forget to manually free the allocated memory once you are done:
  // //cudaFree(positions);
  //Destroy the UAMMD environment and exit
  sys->finish();
  return 0;
}
