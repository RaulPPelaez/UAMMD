# **Universally Adaptable Multiscale Molecular Dynamics (UAMMD) ver 1.0**


<img src="https://github.com/raulppelaez/uammd/blob/master/.res/poster.png" width="300"><img src="https://github.com/raulppelaez/uammd/blob/master/.res/shotlogo.png" width="500">  


**See the wiki for more info!**  
**You can find videos on the youtube channel**  http://bit.ly/2r5WoOn

## DESCRIPTION  

-----------------  

Raul P. Pelaez 2018-2021. (raul.perez(at)uam.es)  


A C++14+ header-only fast generic multiscale CUDA Molecular Dynamics framework made with moduarity, expandability and generality in mind. UAMMD is intended to be hackable and copy pastable.  

Although "Molecular Dynamics" is part of the name,the UAMMD framework allos for much more than that. To this moment multiple integrators are implemented allowing it to perform:  

	-Molecular dynamics (MD)  
	-Brownian Dynamics  (BD)  
	-Brownian Hydrodynamics (BDHI)  
	-Dissipative Particle Dynamics (DPD)  
	-Smoothed Particle Hydrodynamics (SPH)  
	-Metropolis Monte Carlo (MC)   
	-Lattice Boltzmann (LBM)(WIP)  
	-Fluctuating Hydrodynamics (coupled with particles with Immerse Boundary Method (IBM))  
		

Multiple building blocks are provided for the user to construct a certain simulation. Most are highly templated to ease adaptability.  

For example, there is not a harmonic trap module, but you can write a simple functor (directly in device code!) stating that each particle should experiment a force when it is trying to leave the box. Then you can pass this functor to the ExternalForces module. similar things can be done with a bonded force, an interaction that needs to trasverse a neighbour list, an nbody interaction...   

Hop on to the examples folder for an introduction or check the [wiki](https://github.com/RaulPPelaez/UAMMD/wiki) for more information.  


# Currently Implemented

See the wiki page at https://github.com/RaulPPelaez/UAMMD/wiki for a full list of available modules.  

----------------------
## USAGE

-------------------

**UAMMD does not need to be compiled separatedly (it is header only)**.  

Some special flags might be needed to compile some codes with certain UAMMD headers, see [Compiling UAMMD](https://github.com/RaulPPelaez/UAMMD/wiki/compiling-uammd).  
Here you have a short example of how a typical UAMMD code looks like:  


```c++
//Ideal brownian particles
#include"uammd.cuh"
#include"Integrator/BrownianDynamics.cuh"
using namespace uammd;
int main(int argc, char * argv[]){
	int numberParticles = 1e5;
	auto sys = make_shared<System>(argc, argv);
	auto pd = make_shared<ParticleData>(numberParticles, sys);
	{
		auto pos = pd->getPos(access::location::cpu, access::mode::write);
		std::generate(pos.begin(), pos.end(), [&](){ return make_real4(sys->rng.uniform3(-0.5, 0.5), 0);});	
	}
	BD::EulerMaruyama::Parameters par;
	par.temperature = 1.0;
	par.viscosity = 1.0;
	par.hydrodynamicRadius = 1.0;
	par.dt = 0.1;
	auto bd = make_shared<BD::EulerMaruyama>(pd, sys, par);
	for(int i = 0; i<numberSteps; i++){
		bd->forwardTime();
	}
	sys->finish();
	return 0;
}

```

Drop by the examples folder to get started with UAMMD or go to the [wiki](https://github.com/RaulPPelaez/UAMMD/wiki).  

## DEPENDENCIES  

---------------------
Depends on:

	1. CUDA 9.x+                                :   https://developer.nvidia.com/cuda-downloads

Some modules make use of certain NVIDIA libraries included with CUDA:
	
	1. cuBLAS
	2. cuFFT
	
Some modules also make use of lapacke and cblas (which can be replaced by mkl).  
Apart from this, any dependency is already included in the repository under the third_party	folder.  
See [Compiling UAMMD](https://github.com/RaulPPelaez/UAMMD/wiki/Compiling-UAMMD) in the wiki for more information.  

------------------------------------------

## ACKNOWLEDGMENTS

UAMMD is being developed at the Departamento de Física Teórica de la Materia Condensada of Universidad Autónoma de Madrid (UAM) under supervision of Rafael Delgado-Buscalioni. Acknowledgment is made to the Donors of the American Chemical Society Petroleum Research Fund (**PRF# 54312-ND9**) for support of this research and to Spanish MINECO projects **FIS2013- 47350-C05-1-R and FIS2013-50510-EXP**.  

Acknowledgment is made to NVIDIA Corporation for their GPU donations.  

## Collaborators

Raul P. Pelaez is the main developer of UAMMD.  

Other people that have contributed to UAMMD (thanks!):  

Marc Melendez Schofield  
Sergio Panzuela  
Nerea Alcazar  
Pablo Ibañez Freire (https://github.com/PabloIbannez)  
Salvatore Assenza
