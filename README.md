# **Universally Adaptable Multiscale Molecular Dynamics (UAMMD) ver 2.5**

[![Documentation Status](https://readthedocs.org/projects/uammd/badge/?version=latest)](https://uammd.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/RaulPPelaez/UAMMD/actions/workflows/ci.yml/badge.svg)](https://github.com/RaulPPelaez/UAMMD/actions/workflows/ci.yml)

<img src="https://github.com/raulppelaez/uammd/blob/master/.res/poster.png" width="300"><img src="https://github.com/raulppelaez/uammd/blob/master/.res/shotlogo.png" width="500">  


**See the [documentation](https://uammd.readthedocs.io/en/latest/) for more info!**  
**You can find videos on the youtube channel**  http://bit.ly/2r5WoOn

-----------------  

A C++14+ header-only fast generic multiscale CUDA Molecular Dynamics framework made with moduarity, expandability and generality in mind. UAMMD is intended to be hackable and copy pastable.  

Although "Molecular Dynamics" is part of the name,the UAMMD framework allows for much more than that. To this moment multiple integrators are implemented allowing it to perform:  

	-Molecular dynamics (MD)  
	-Brownian Dynamics  (BD)  
	-Brownian Hydrodynamics (BDHI)  
	-Dissipative Particle Dynamics (DPD)  
	-Smoothed Particle Hydrodynamics (SPH)  
	-Metropolis Monte Carlo (MC)   
	-Lattice Boltzmann (LBM)(WIP)  
	-Fluctuating Hydrodynamics (coupled with particles with Immerse Boundary Method (IBM))  
		

Building blocks are provided for the user to construct a certain simulation. Most are highly templated to ease adaptability.  

For example, there is no harmonic trap module, but you can write a simple functor (directly in device code!) stating that each particle should experiment a force when it is trying to leave the box. Then you can pass this functor to the ExternalForces module. Similar things can be achieved with a bonded force, an interaction that needs to trasverse a neighbour list, an nbody interaction...   

Hop on to the examples folder for an introduction or check the [documentation](https://uammd.readthedocs.io) for more information.  


# Currently Implemented

See the documentation page at https://uammd.readthedocs.io for a full list of available modules.  

----------------------
## Usage

-------------------

You can use UAMMD as a library for integration into other codes or as a standalone engine.

### CMake integration  

-------------------------

UAMMD is compatible with CMake's FetchContent:

```cmake
FetchContent_Declare(
  uammd
  GIT_REPOSITORY https://github.com/RaulPPelaez/uammd
  GIT_TAG        v2.5.4
)
FetchContent_MakeAvailable(uammd)
add_executable(my_executable my_source.cu)
uammd_setup_target(my_executable)
```


#### DEPENDENCIES  

---------------------
Depends on:

	1. CUDA 9.x+                                :   https://developer.nvidia.com/cuda-downloads

Some modules make use of certain NVIDIA libraries included with CUDA:
	
	1. cuBLAS
	2. cuFFT
	
Some modules also make use of lapacke and cblas (which can be replaced by mkl).  
Apart from this, any dependency is already included in the repository under the third_party	folder.  
See [Compiling UAMMD](https://uammd.readthedocs.io/en/latest/Compiling-UAMMD.html) in the documentation for more information.  

Every required dependency can be installed using conda with the environment file provided in the repository. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) as a replacement for conda.  

```bash
mamba env create -f environment.yml
```

### Library mode

**UAMMD does not need to be compiled separatedly (it is header only)**.  

The top-level CMakeLists.txt file will install all UAMMD headers to `$CMAKE_INSTALL_PREFIX/include/uammd`. Additionally, a cmake module file (`FindUAMMD.cmake` will be installed at `$CMAKE_INSTALL_PREFIX/share/cmake/Modules`). To install the headers, go to the root of this repo and run:

```shell
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX .. # If you wish to install the headers to the conda environment
$ make install
```

Now other CMake scripts can find the UAMMD headers with:

```cmake
find_package(UAMMD REQUIRED)
include_directories(${UAMMD_INCLUDE_DIR})
# Or
# uammd_setup_target(target_name)
```

Some special flags might be needed to compile codes including with certain UAMMD headers, see [Compiling UAMMD](https://uammd.readthedocs.io/en/latest/Compiling-UAMMD.html).  
Here you have a short example of how a typical UAMMD code looks like, encoding a simple Brownian dynamics simulation of non interacting particles.:  

```c++
//Ideal brownian particles
#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
using namespace uammd;
int main(int argc, char * argv[]){
	int numberParticles = 1e5;
	auto pd = make_shared<ParticleData>(numberParticles);
	{
		auto pos = pd->getPos(access::cpu, access::write);
		std::generate(pos.begin(), pos.end(), [&](){ return make_real4(sys->rng.uniform3(-0.5, 0.5), 0);});	
	}
	BD::EulerMaruyama::Parameters par;
	par.temperature = 1.0;
	par.viscosity = 1.0;
	par.hydrodynamicRadius = 1.0;
	par.dt = 0.1;
	auto bd = make_shared<BD::EulerMaruyama>(pd, par);
	for(int i = 0; i<numberSteps; i++){
		bd->forwardTime();
	}
	sys->finish();
	return 0;
}

```

Drop by the examples folder to get started with UAMMD or go to the [documentation page for the examples](https://uammd.readthedocs.io/en/latest/Examples.html).  

### Stand alone engine

The example `generic_md` includes almost every module available in UAMMD and can be configured from a parameter file. Go to `examples/generic_md` for instructions.

## Running Tests  

---------------------

The `tests` folder contains instructions on how to run the UAMMD correctness tests. You can also go to the [documentation page for tests](https://uammd.readthedocs.io/en/latest/Tests.html).


## Repository structure  

-------------------------

- The `docs` folder contains the documentation source files for [](https://uammd.readthedocs.io/) and scripts to build it.

- The `examples` folder contains several examples of UAMMD usage.

- The `src` folder contains the source code for UAMMD. Inside it, there are the following folders:
1. `global`: Definitions proper to the whole code.
2. `Integrator`: Source code for the [integrators](https://uammd.readthedocs.io/en/latest/Integrators.html).
3. `Interactor`: Source code for the [interactors](https://uammd.readthedocs.io/en/latest/Interactors.html).
4. `misc`: Miscellaneous source code.
5. `ParticleData`: Source code for [particle data](https://uammd.readthedocs.io/en/latest/ParticleData.html).
6. `System`: Source code for [system](https://uammd.readthedocs.io/en/latest/System.html).
7. `third_party`: Third party source code.
8. `utils`: Utilities for UAMMD.

- The `test` folder contains the source code for the UAMMD tests.

- The `extensions` is part of the subsystem for UAMMD extensions, currently a work in progress.

------------------------------------------

## ACKNOWLEDGMENTS

UAMMD is being developed at the Departamento de Física Teórica de la Materia Condensada of Universidad Autónoma de Madrid (UAM) under supervision of Rafael Delgado-Buscalioni. Acknowledgment is made to the Donors of the American Chemical Society Petroleum Research Fund (**PRF# 54312-ND9**) for support of this research and to Spanish MINECO projects **FIS2013- 47350-C05-1-R, FIS2013-50510-EXP** and mostly **FIS2017-86007-C3-1-P**.  

Acknowledgment is made to NVIDIA Corporation for their GPU donations.  

## Collaborators

Raul P. Pelaez is the main developer of UAMMD.  

Other people that have contributed to UAMMD (thanks!):  

Marc Melendez Schofield  
Pablo Ibañez Freire (https://github.com/PabloIbannez)  
Pablo Palacios Alonso (http://github.com/PabloPalaciosAlonso) 
Sergio Panzuela  
Nerea Alcazar  
Salvatore Assenza
