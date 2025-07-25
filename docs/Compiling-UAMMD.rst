Compiling UAMMD
================

UAMMD is header-only, so a module is compiled only when its header is included. CMake integration is provided and is the recommended way to compile UAMMD-related sources, see below.

**In order to compile a file that includes an UAMMD module you will need the following external dependencies**:  

  #. CUDA 8.0+  (https://developer.nvidia.com/cuda-downloads )  
  #. A C++ compiler with C++14 support (g++ 5+ will probably do)  
  #. LAPACKE/CBLAS or Intel MKL (For some modules only)
  #. (Optional) Eigen3 

**These dependencies can be installed using conda with the provided environment.yml file.**

Additionally, UAMMD makes use of the following external libraries, **which are already included in the repo under third_party**. You can either compile using these or place symlinks to your preferred versions (I have seen recent versions of cub not compiling on some platforms).  

  * Boost Preprocessor (http://www.boost.org/ ) (Extracted using bcp, just a few headers)
  * nod (https://github.com/fr00b0/nod ) (A lightweight C++11 signal/slot library)
  * SaruPRNG (http://dx.doi.org/10.1016/j.cpc.2012.12.003 ) (A parallel friendly RNG)
    

**To compile and link all of the existing modules at once you would need the following flags in nvcc:**

.. code:: bash
	  
  nvcc -std=c++14 --expt-relaxed-constexpr -I$(UAMMD_ROOT)/src  -I$(UAMMD_ROOT)/src/third_party -lcufft -lcurand -lcublas -lcusolver -llapacke -lcblas

Lapacke and Cblas can be switched by intel's MKL by adding the define -DUSE_MKL and linking accordingly.

DEBUG COMPILATION
-------------------

You may encounter issues when trying to use the -G flag as some old versions of thrust appear to conflict with it.

https://github.com/thrust/thrust/wiki/Debugging  

Instead you may define THRUST_DEBUG.

Defining UAMMD_DEBUG will enable a lot of synchronizing cuda error checks to help pinpointing where an error occurs.  

AVAILABLE MACROS 
-----------------

You can define the following preprocessor macros to change compile options:

  * **-DDOUBLE_PRECISION** Will compile UAMMD in double precision mode (single precision by default)
  * **-DUSE_EIGEN** Will include Eigen3, can improve performance in some modules. Defaults to OFF.
  * **-DUSE_OPENMP** Will enable OpenMP parallelization in some modules. Defaults to ON.
  * **-DMAXLOGLEVEL=X** Will set the maximum log level to X, 0 will print only critical errors, 6 will print some debug messages, 14 is the maximum verbosity and 5 will only print up to information messages (recommended).  
  * **-DUAMMD_DEBUG** Will enable CudaCheckError() calls. These calls imply a synchronization barrier and will halt the execution if an error is found with information about what went wrong. You can see these lines scattered all around the code base.  
  * **-DUSE_NVTX** Will enable nvtx ranges. If undefined (default) PUSH/POP_RANGE calls will be ignored. See utils/NVTXTools.h.
  * **-DUSE_MKL** Will include mkl.h instead of lapacke.h and cblas.h when using lapack and blas functions. Note that UAMMD's CMake integration will automatically define this macro if it finds Intel MKL installed in your system as a BLAS provider.
  * **-DEXTRA_COMPUTABLES** Enables new :cpp:any:`Computables` in :ref:`Interactor`.
  * **-DEXTRA_PARTICLE_PROPERTIES** Enables new particle properties in :ref:`ParticleData`.
  * **-DEXTRA_UPDATABLE_PARAMETERS** Enables new parameters in :ref:`ParameterUpdatable`.


Compiling tests and examples 
..............................

The examples and many tests can be compiled with CMake, an example that should work most of the time is in the examples folder.

When a CMakeLists.txt file is present you can compile its target by running:

.. code:: bash

	  $ mkdir -p build && cd build
	  $ cmake ..
	  $ make

The top-level CMakeLists.txt can be used to install UAMMDs headers to your system:

.. code:: bash
	  
	  $ cd /path/to/uammd
	  $ mkdir -p build && cd build
	  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
	  $ make install

Integration with CMake's find_package
........................................

UAMMD can be integrated with CMake's find_package mechanism. To do this, you need to install UAMMD in your system and call find_package(UAMMD) in your CMakeLists.txt file.

After successful finding of UAMMD, the :code:`UAMMD_INCLUDE_DIRECTORIES` variable will be available, alongside with the :code:`uammd_setup_target` function, which you can use to configure your targets to use UAMMD. For example:

.. code:: cmake

   find_package(UAMMD REQUIRED)
   add_library(my_target my_target.cu)
   uammd_setup_target(my_target)

Integration with CMake's FetchContent
........................................

If you don't want to install UAMMD in your system, you can use CMake's FetchContent mechanism to download UAMMD and compile it as part of your project build process:

.. code:: cmake

   FetchContent_Declare(
	  uammd
	  GIT_REPOSITORY https://github.com/RaulPPelaez/uammd
	  GIT_TAG        v2.9.0 # or any other tag/branch you want to use
   )
   FetchContent_MakeAvailable(uammd)
   add_library(my_target my_target.cu)
   uammd_setup_target(my_target)


Additional CMake options
.........................
You can pass additional options to CMake when configuring your project. For example, if you want to enable double precision and use Eigen3 but not OpenMP, you can do:

.. code:: bash

   $ cmake -DDOUBLE_PRECISION=ON -DUSE_EIGEN=ON -DUSE_OPENMP=OFF ..
