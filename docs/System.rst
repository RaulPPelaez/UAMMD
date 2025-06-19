System
=======

.. doxygenfile:: System/System.h
   :project: uammd
   :sections: briefdescription detaileddescription


.. doxygenclass:: uammd::System
   :project: uammd
   :members:

.. doxygenstruct:: uammd::SystemParameters
   :project: uammd
   :members:


.. hint:: System (or :ref:`ParticleData`) should be the first thing to create in a :ref:`UAMMD simulation code <Simulation file>`, see any of the examples in examples folder.

**Example: creating a System instance**

.. code:: cpp
	  
  ...
  int main(int argc, char *argv[]){
    auto sys = std::make_shared<System>(argc, argv);
    //You may seed the rng algorithm here. The default seed is obtained from time(NULL).
    sys->rng().setSeed(time(NULL));
  ...
    //If we initialize System, ParticleData has to be provided with the same instance
    int N = 10000; //An arbitrary number of particles
    auto pd = std::make_shared<ParticleData>(N, sys);
    //This will ensure the termination of any UAMMD related operation.    
    sys->finish();
    return 0;
  }

Utilities provided by System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Random numbers
**************

You can access a Xorshift128plus random generator from System. This generator should be used for things like seeding other generators, maybe creating initial configurations.  

.. cpp:function:: Xorshift128plus& System::rng();

	Returns a reference to the System's random number generator.


.. code:: cpp
	  
  ...
  //Set the seed of the rng
  sys->rng().setSeed(123124);
  //Get a random integer between 0 and 2^64-1
  uint64_t Z64 = sys->rng().next();
  //Get a random integer between 0 and 2^32-1
  uint32_t Z32 = sys->rng().next32();
  //Get a uniform number between a and b
  double a=0, b=1;
  double Zf = sys->rng().uniform(a,b);
  //Get a gaussian number with mean a and std b
  double Zg = sys->rng().gaussian(a,b);
  ...

.. _logging-doc:

Logging
********

System provides a logging engine with several levels of relevance for the message via the :cpp:any:`System::log` method. The log level is set at compile time via the preprocessor macro :cpp:expr:`MAXLOGLEVEL`, which can be set to any value between 0 and 13. The higher the value, the more verbose the logging will be. See :ref:`Compiling UAMMD`. The levels are:

.. doxygenenum:: uammd::System::LogLevel
   :project: uammd

**Example:**

.. code:: cpp
	  
  System::log<System::CRITICAL>("The program will be terminated after printing this message!");
  System::log<System::ERROR>("There was an error, but I might be able to circumvent it!");
  System::log<System::WARNING>("Something happen that might be problematic, watch out! (I will keep running OK though)");
  System::log<System::MESSAGE>("Here you have some useful information I want you to know!");
  System::log<System::STDERR>("This goes straight to stderr");
  System::log<System::STDOUT>("This goes straight to stdout");
  System::log<System::DEBUG>("A debug message!");
  System::log<System::DEBUG1>("A more internal debug message!");
  ...
  System::log<System::DEBUG7>("There are 7 levels of debug!");


.. _memory-management-doc:
   
Cached memory allocation
**************************

UAMMD exposes a cached allocator of GPU memory via System under the names :cpp:any:`System::allocator\<T>` and :cpp:any:`System::allocator_thrust\<T>`. Both types comply with the `C++ standard library's Allocator concept <https://en.cppreference.com/w/cpp/named_req/Allocator>`_ and can thus be used with the standard library's containers, such as :cpp:type:`std::vector`.


.. cpp:class:: template<class T> System::allocator

      An std-compatible polymorphic pool allocator that provides GPU memory (allocated via cudaMalloc).
      Allocations via this type will be cached. If a chunk of memory is allocated and deallocated, the next time a similar chunk is requested will not incur in a cudaMalloc call.

.. cpp:class:: template<class T> System::thrust_allocator

      Thurst containers require an allocator with a pointer type :cpp:any:`thrust::device_ptr\<T>` (instead of the plain :cpp:expr:`T*` provided by :cpp:type:`System::allocator\<T>`). This type behaves identical to :cpp:any:`System::allocator\<T>` (and shares its memory pool) but can be used with thrust containers.

.. hint::

   System's allocators normally work with GPU global memory, but turn to managed memory if UAMMD_DEBUG is defined.
      

Usage example:
%%%%%%%%%%%%%%%%%%%

.. code:: cpp
	  
  #include"uammd.cuh"
  #include<thrust/device_vector.h>
  #include<memory>
  #include<vector>
  
  using namespace uammd;
  
  int main(){
    //Note that System's allocator functionality are provided via static calls, an actual instance is not needed. But a System being initialized somewhere will help.
    auto sys = std::make_shared<System>();
    //Only the first iteration incurs a cudaMalloc, and cudaFree is called only when System::finish() is called.
    for(int i = 0; i<10; i++){
      thrust::device_vector<char, System::allocator_thrust<char>> vec;
      vec.resize(10000);
    }
    //You can interchange with a thrust vector using the default allocator.
    {
      thrust::device_vector<char, System::allocator_thrust<char>> vec;
      vec.resize(10000);
      thrust::device_vector<char> device_copy_with_default_allocator(vec);
    }
    {
      //Using the allocator with a shared_ptr. 
       std::shared_ptr<int> temp;
       //Note that this is a static method, a System instance is not actually needed.
       //A default global instance of the allocator is returned.
       auto alloc = sys->getTemporaryDeviceAllocator<int>();
       temp = std::shared_ptr<int>(alloc.allocate(1000),
                                   [=](int* ptr){alloc.deallocate(ptr);});
    }  
    sys->finish();
    return 0;
  }
 

System options
~~~~~~~~~~~~~~~

Here is a list of flags accepted by System

    * --device X : UAMMD will run in the GPU with number X.
    * --increase_print_limit X : CUDA's printf limit will be increased to X.
