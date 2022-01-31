Compiling UAMMD
================

UAMMD is header-only, so a module is compiled only when its header is included. See example/Makefile for a tutorial.  

**In order to compile a file that includes an UAMMD module you will need the following external dependencies**:  

  #. CUDA 7.5+  (https://developer.nvidia.com/cuda-downloads )  
  #. A C++ compiler with C++14 support (g++ 5+ will probably do)  
  #. Thrust (The version shipped with the cuda release will do)  
  #. LAPACKE/CBLAS or Intel MKL (For some modules only)  

The newest compiler versions I have tested are g++-9.x and clang++-9.0 with cuda-11.1 in CentOS 8.  You can even compile a source containing UAMMD code with clang++-7+ alone, without nvcc.

Additionally, UAMMD makes use of the following external libraries, **which are already included in the repo under third_party**. You can either compile using these or place symlinks to your preferred versions (I have seen recent versions of cub not compiling on some platforms).  

  * Boost Preprocessor (http://www.boost.org/ ) (Extracted using bcp, just a few headers)  
  * CUB 1.5.2+ (https://github.com/NVlabs/cub ) (Currently 1.8.0 in third_party/cub_bak*)  
  * nod (https://github.com/fr00b0/nod ) (A lightweight C++11 signal/slot library)
  * SaruPRNG (http://dx.doi.org/10.1016/j.cpc.2012.12.003 ) (A parallel friendly RNG)
    
\*cub 1.8.0 might give compilation errors for GPU arquitecture < 30. If you need to compile for these arquitectures please replace cub for a previous version. cub is included with CUDA starting with CUDA 11, if you are using an older CUDA installation you must place cub in the include path (for example by renaming cub_bak to cub in third_party).

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
  