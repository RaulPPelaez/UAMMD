Compiling UAMMD
================

UAMMD is header-only, so a module is compiled only when its header is included. See example/Makefile for a tutorial.  

**In order to compile a file that includes an UAMMD module you will need the following external dependencies**:  

  #. CUDA 8.0+  (https://developer.nvidia.com/cuda-downloads )  
  #. A C++ compiler with C++14 support (g++ 5+ will probably do)  
  #. LAPACKE/CBLAS or Intel MKL (For some modules only)  

The newest compiler versions I have tested are g++-12.x and clang++-13.0 with cuda-12 in Fedora 37.  You can even compile a source containing UAMMD code with clang++-7+ alone, without nvcc.

Additionally, UAMMD makes use of the following external libraries, **which are already included in the repo under third_party**. You can either compile using these or place symlinks to your preferred versions (I have seen recent versions of cub not compiling on some platforms).  

  * Boost Preprocessor (http://www.boost.org/ ) (Extracted using bcp, just a few headers)
  * Thrust (https://github.com/thrust/thrust )
  * CUB 1.5.2+ (https://github.com/NVlabs/cub ) (Currently 1.8.0 in third_party/cub_bak*)  
  * nod (https://github.com/fr00b0/nod ) (A lightweight C++11 signal/slot library)
  * SaruPRNG (http://dx.doi.org/10.1016/j.cpc.2012.12.003 ) (A parallel friendly RNG)
    
\*cub 1.8.0 might give compilation errors for GPU arquitecture < 30. If you need to compile for these arquitectures please replace cub for a previous version. cub is included with CUDA starting with CUDA 11, if you are using an older CUDA installation and get some related compilation problem, check third_party/uammd_cub.cuh.

Each module might need a different set of libraries and/or compiler flags, specified in the header of each module.

**To compile and link all of the existing modules at once you would need the following flags in nvcc:**

.. code:: bash
	  
  nvcc -std=c++14 --expt-relaxed-constexpr -I$(UAMMD_ROOT)/src  -I$(UAMMD_ROOT)/src/third_party -lcufft -lcurand -lcublas -lcusolver -llapacke -lcblas

Lapacke and Cblas can be switched by intel's MKL by adding the define -DUSE_MKL and linking accordingly.

CUB comes bundled with newer CUDA toolkits (11+) and will probably cause troubles if intermixed with the version stored under third_party. In that case simply remove or rename third_party/cub.  

DEBUG COMPILATION
-------------------

You may encounter issues when trying to use the -G flag as some old versions of thrust appear to conflict with it.

https://github.com/thrust/thrust/wiki/Debugging  

Instead you may define THRUST_DEBUG.  

Defining UAMMD_DEBUG will enable a lot of synchronizing cuda error checks to help pinpointing where an error occurs.  

AVAILABLE MACROS 
-----------------

You can define the following preprocessor macros to change compile options:

  * **-DUSE_MKL** Will include mkl.h instead of lapacke.h and cblas.h when using lapack and blas functions.  
  * **-DDOUBLE_PRECISION** Will compile UAMMD in double precision mode (single precision by default)  
  * **-DMAXLOGLEVEL=X** Will set the maximum log level to X, 0 will print only critical errors, 6 will print some debug messages, 14 is the maximum verbosity and 5 will only print up to information messages (recommended).  
  * **-DUAMMD_DEBUG** Will enable CudaCheckError() calls. These calls imply a synchronization barrier and will halt the execution if an error is found with information about what went wrong. You can see these lines scattered all around the code base.  
  * **-DUSE_NVTX** Will enable nvtx ranges. If undefined (default) PUSH/POP_RANGE calls will be ignored. See utils/NVTXTools.h.  
  * **-DEXTRA_COMPUTABLES** Enables new :cpp:any:`Computables` in :ref:`Interactor`.
  * **-DEXTRA_PARTICLE_PROPERTIES** Enables new particle properties in :ref:`ParticleData`.
  * **-DEXTRA_UPDATABLE_PARAMETERS** Enables new parameters in :ref:`ParameterUpdatable`.

COMMON ERRORS
---------------

CUB and CUDA versions
.....................

You might get an error similar to this one:

.. code:: bash

	  
	  thrust/system/cuda/config.h:79:2: error: #error The version of CUB in your include path is not compatible with this release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.

This error might arise if the mechanism UAMMD uses to include cub in older CUDA versions fails and :code:`third_party/cub_bak` is included instead (in which case the version provided by the CUDA installation should be used). In order to fix this, start looking in the file :code:`third_party/cub_bak`.

Bug in GCC >11.2.1 with CUDA 11+
......................................

You might get an error containing something like:

.. code:: bash

    /usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’:
    435 |         function(_Functor&& __f)
        |                                                                                                                                                 ^
  /usr/include/c++/11/bits/std_function.h:435:145: note:         ‘_ArgTypes’
  /usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
    530 |         operator=(_Functor&& __f)
        |                                                                                                                                                  ^
  /usr/include/c++/11/bits/std_function.h:530:146: note:         ‘_ArgTypes’


This is a bug in GCC that prevents from compiling CUDA code. Related discussion: https://github.com/pytorch/pytorch/issues/71518

If you encounter this, downgrade GCC to 11.2.1 or use Clang 12 instead. Check in the CUDA documentation that you have valid versions of the different compilers: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements


Compiling with MKL
.....................

Some UAMMD headers require LAPACK and/or BLAS functions. Many systems lack the LAPACKE and/or CBLAS libraries and instead provide Intel's MKL. UAMMD allows to use MKL if you define the USE_MKL macro (by passing -DUSE_MKL when compiling an code including some UAMMD header).

Then, instead of linking with lapacke/cblas (for instance with -llapacke -lcblas) you will need to set up a compilation line using intel's mkl link line advisor:


https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html


Which will provide you with a set of flags, for instance: :code:`-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl`

Compiling with CMake
.........................

The examples and many tests can be compiled with CMake, an example that should work most of the time is in the examples folder.

When a CMakeLists.txt file is present you can compile its target by running:

.. code:: bash

	  $ mkdir -p build && cd build
	  $ cmake ..
	  $ make
