Random number generation with Saru
======================================

:ref:`System` offers a handy serial CPU RNG that is good for things like seeding other generators, and in general as away for all UAMMD modules to share a common RNG. However, many times we find ourselves in needs of GPU random number generation.
     
UAMMD exposes Saru [1]_, a powerful massively parallel random number generator wrote by Steve Worley. Saru can be instanced in both GPU and CPU code and it is blazing fast. Additionally, it passes every RNG battery test imaginable (such as DieHard and the like).

.. cpp:class:: Saru


   .. cpp:function:: Saru(uint seed1, uint seed2, uint seed3)

      Creates a Saru instance using three seeds.
      
   .. cpp:function:: Saru(uint seed1, uint seed2)

      Creates a Saru instance using two seeds.
      
   .. cpp:function:: Saru(uint seed1)

      Creates a Saru instance using one seed.
      
   .. cpp:function:: float f()

      Generates a floating point number in the :math:`[0-1)` range.

   .. cpp:function:: double d()
			   
      Generates a double precision point number in the :math:`[0-1)` range.

   .. cpp:function:: float2 gf(float mean, float std)

      Generates two Gaussian float numbers with a given mean and standard deviation using the Box Muller algorithm.

   .. cpp:function:: double2 gd(double mean, double std)

      Generates two Gaussian double numbers with a given mean and standard deviation using the Box Muller algorithm.
      
   .. cpp:function:: uint u32()

      Outputs a random 32 bit unsigned integer


Saru exposes a bunch of other functions allowing to advance/rewind the state of the generator. See :code:`third_party/saruprng.cuh` for more information.

Take into account that the Saru state is small (consisting on only two 32 bit integers) and initializing it involves a small number of integer instructions. Contrary to most generators, Saru is really good handling seeds even when the offer low "bit chaos" (for instance, 0, or 0b1111111...) and it does not require any "warm up" (the first numbers provided by some generators will sometimes be highly correlated). My advise is to simply create and seed a new Saru instance whenever you need.


Example
---------

.. code:: cpp

  #include"uammd.cuh"
  #include "third_party/saruprng.cuh"
  
  using namespace uammd;
  
  int main(){
    //Let us fill some particle positions randomly using Saru in the GPU.
    int numberParticles = 1e5;
    auto pd = std::make_shared<ParticleData>(numberParticles);
    auto pos = pd->getPos(access::gpu, access::write);
    //Saru requires three seeds.
    //Typically, we will leverage this by using:
    // 1- A random number that we choose at the start of the simulation
    // 2- The current time step (i.e any counter that changes every time the generator is issued)
    // 3- A per-particle seed (i.e just the particle index)
    // One of the strenghts of Saru is the ability to accept "bad" seeds, such as 0.
    uint seed1 = pd->getSystem()->rng().next32();
    uint seed2 = 0;
    auto cit = thrust::make_counting_iterator(0);
    real3 lbox = {32,32,32}; //Some arbitrary domain size
    //Let us generate the positions inside a cubic box of size lbox
    thrust::transform(cit, cit + pos.size(),
  		    pos.begin(),
  		    [=]__device__(int i){
  		      //The third seed allows to generate different numbers for each particle concurrently
  		      Saru rng(seed1, seed2, i);
  		      //Three random numbers in the [0-1) range
  		      real3 z = {rng.f(), rng.f(),rng.f()};
  		      real4 p_i = make_real4((z-0.5)*lbox, 0);
  		      return p_i;
  		    });
  
    return 0;
  }
     




.. rubric:: References:

.. [1] Exploiting seeding of random number generators for efficient domain decomposition parallelization of dissipative particle dynamics. Y. Afshar et. al. 2013.
