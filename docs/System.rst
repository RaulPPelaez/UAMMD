System
=======

System is the fundamental UAMMD module, any other module or submodule needs a reference to a System instance (although this references is not often needed to be provided explicitly).

System contains information about the machine UAMMD is running on and offers a series of utilities like a CPU random number generator and a logging infrastructure. Furthermore, System handles initializing and deleting the CUDA  environment (i.e selecting a GPU to run on).  


Creation
~~~~~~~~~


.. cpp:class:: System

	   .. cpp:function:: System::System()

			     Default Constructor.
			     
	   .. cpp:function:: System::System(int argc, char* argv[])

			     System can take in the command line arguments and understands :ref:`some options <System options>` for them.
			     

Most of the time you do not need to handle the creation of System. :ref:`ParticleData` will auto initialize it for you if you do not provide one.

You can request ParticleData for a reference to System like this:

.. code:: cpp

  auto sys = pd->getSystem();	  

System should be created explicitly if the user wants to provide System with the command line arguments (see :ref:`below <System options>` on why you would want this).

System (or :ref:`ParticleData`) should be the first thing to create in a :ref:`UAMMD simulation code <Simulation file>`, see any of the examples in examples folder.

A System instance can be initialized like this:

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
  
Logging
********

System provides a logging engine with several levels of relevance for the message. The available levels are in the System::level enum in System.h. You can see the following in this example:

.. cpp:function:: template<class LogLevel> void System::log<LogLevel>(const char * format, ...);

	Log messages with a printf-like interface. Requires a log level (see below).
		  
             

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


The maximum level of logging that will be processed will be the compile constant maxLogLevel in Log.h. Anything below this level will not even be compiled, so do not be worried about performance when writing debug logs. The highest level of logging that will print DEBUG messages is maxLogLevel = 6. maxLoglevel=13 will print up to DEBUG7, while maxLogLevel=0 will only print CRITICAL errors. However, the max log level should be set via a preprocessor macro at compilation, see :ref:`Compiling UAMMD`

Cached memory allocation
**************************

UAMMD exposes a cached allocator of GPU memory via System under the names :cpp:any:`System::allocator\<T>` and :cpp:any:`System::allocator_thrust\<T>`. Both types comply with the `C++ standard library's Allocator concept <https://en.cppreference.com/w/cpp/named_req/Allocator>`_ and can thus be used with the standard library's containers, such as :cpp:type:`std::vector`.


.. cpp:class:: template<class T> System::allocator<T>

      An std-compatible polymorphic pool allocator that provides GPU memory (allocated via cudaMalloc).
      Allocations via this type will be cached. If a chunk of memory is allocated and deallocated, the next time a similar chunk is requested will not incur in a cudaMalloc call.

.. cpp:class:: template<class T> System::thrust_allocator<T>

      Thurst containers require an allocator with a pointer type :cpp:any:`thrust::device_ptr\<T>` (instead of the plain :cpp:expr:`T*` provided by :cpp:type:`System::allocator\<T>`). This type behaves identical to :cpp:any:`System::allocator\<T>` (and shares its memory pool) but can be used with thrust containers.

            
      

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
  


Other methods
~~~~~~~~~~~~~~


 .. cpp:function:: int System::getargc();

    Returns the number of arguments provided at creation.

 .. cpp:function:: char** System::getargv();

    Returns the list of arguments provided at creation.
		   		   		  
 .. cpp:function:: void System::finish();

    Finishes all UAMMD-related operations and frees any memory allocated by UAMMD. After a call to :code:`finish()` all UAMMD modules are left in an invalid state. This function should be called after every other UAMMD object has been destroyed.
		   		   		  



System options
~~~~~~~~~~~~~~~

Here is a list of flags accepted by System

    * --device X : UAMMD will run in the GPU with number X.
    * --increase_print_limit X : CUDA's printf limit will be increased to X.
