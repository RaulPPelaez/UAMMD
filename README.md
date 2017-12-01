# **Universally Adaptable Multiscale Molecular Dynamics (UAMMD) ver 0.5.alpha**

<img src="https://github.com/raulppelaez/uammd/blob/development/src/res/poster.png" width="300"><img src="https://github.com/raulppelaez/uammd/blob/development/src/res/shotlogo.png" width="500">  


**See the wiki for more info!**  
**You can find videos on the youtube channel**  http://bit.ly/2r5WoOn

<img src="https://github.com/raulppelaez/uammd/blob/development/src/res/poster.png" width="300"><img src="https://github.com/raulppelaez/uammd/blob/development/src/res/shotlogo.png" width="500">  


**See the wiki for more info!**  
**You can find videos on the youtube channel**  http://bit.ly/2r5WoOn

## DESCRIPTION

-----------------

Raul P. Pelaez 2016


A fast generic multiscale CUDA Molecular Dynamics code made into modules for expandability and generality.

UAMMD can perform several types of simulations, to this moment multiple integrators are implemented allowing it to perform:

	-Molecular dynamics (MD)
	-Brownian Dynamics  (BD)
	-Brownian Hydrodynamics (BDHI)
	-Dissipative Particle Dynamics (DPD)

Multiple building blocks are provided in order for the user to construct a certain simulation, 
highly templated so the user can easily add in the input the specific interactions when they are not implemented by default. 

For example, there is not a harmonic trap module, but you can write a simple functor in the input file (directly in device code!) telling that each particle should experiment a force when it is trying to leave the box and you are set!. You can do the same with a bonded force, an interaction that needs to transverse a neighbour list, an nbody interaction... See the examples folder and the wiki for more info!

UAMMD is coded into separated modules, with a SimulationConfig driver in C++ that can hold many modules in order to construct a simulation. For example, the simulation could have a VerletNVT module and and PairForces interactor module to create a molecular dynamics simulation. Or a DPD integrator module with Nbody interactor module, etc.

There are three types of modules:

      1. Integrators
      2. Interactors
	  3. Measurables

**Interactors**

An Interactor is an abstract entity that has the ability of computing the forces acting of each particle due to some interaction.
For example, an Interactor could compute the pair Lennard Jonnes forces between each particle pair of the system or sum the forces due to the particles being joined by springs. 

**Integrators**

An Integrator is an entity that has the ability of moving the particle positions to the next time step. 
In order to do so it can hold any number of Interactors and use them to compute the forces at any time.

**Measurables**

A Measurable is any computation that has to be performed between steps of the simulation, it can be any magnitude that is calculated from the simulation state (positions, forces..).
A measurable can compute the energy, the radial function distribution or any arbitrary computation that does not change the simulation state.

----------------------

These objects are abstract classes that can be derived to create all kinds of functionality and add new physics. Just create a new class that inherits Interactor, Integrator or Measurable and override the virtual methods with the new functionality.



# Currently Implemented

See the wiki page for each interactor for more info and instructions!

-----------------------
**Interactors:**

	1.Pair Forces: Implements hash (Morton hash) sort neighbour cell list construction algorithm to evaluate pair forces given some short range potential function, LJ i.e. Ultra fast
	2.Bonded forces: Allows to join pairs of particles via bonds (i.e a harmonic spring) (Instructions in BondedForces.h)
	3.Three body angle bonded forces: Allows to join triples of particles via angle springs (Instructions in BondedForces.h)
    4.NBody forces: All particles interact with every other via some force.
	5.External forces: A custom force function that will be applied to each particle individually.
	6.Pair Forces DPD: A thermostat that uses the Pair Forces module to compute the interactions between particles as given by dissipative particle dynamics.
	
**Integrators:**

	1.Two step velocity verlet NVE
	2.Two step velocity verlet NVT with BBK thermostat
	3.Euler Maruyama Brownian dynamics (BD)
	4. Brownian Dynamics with Hydrodynamic interactions (BDHI)
	4.1 Euler Maruyama w/HI via RPY tensor 
	4.1.1 Using the Cholesky decomposition on the full Mobility matrix to compute the stochastic term. Open Boundaries.
	4.1.2 Using the Lanczos algorithm and a matrix free method to compute the stochastic term. Open Boundaries.
	4.1.3 Using the Positively Split Ewald method with rapid stochastic sampling. Periodic Boundary Conditions

**Measurables**
	
	1.Energy Measurable. Computes the total, potential and kinetic energy and the virial pressure of the system

----------------------

You can select between single and double precision via defines.h. Single precision is used by default, remember to recompile the entire code when changing the precision. This last step is very important, as failing to do so will result in unexpected behavior.


## USAGE

-------------------
If you dont have cub (thrust comes bundled with the CUDA installation) clone or download the v1.5.2 (see dependencies).
The whole cub repository uses 175mb, so I advice to download the v1.5.2 zip only.  

**UAMMD does not need to be compiled (it is header only)**.  

To use it in your project, include the modules you need, create a System and ParticleData instances and configure the simulation as you need.  
See examples/LJ.cu for a tutorial!  

In order to compile a source file that uses UAMMD, you only have to inform the compiler of the location of the project (with -I) and give the flag "--expt-relaxed-constexpr" to nvcc.  
See examples/Makefile for an example.  

You can use the --device X flag to specify a certain GPU.  


## DEPENDENCIES

---------------------
Depends on:

	1. CUB       (v1.5.2 used)                  :   https://github.com/NVlabs/cub
	2. thrust    (v1.8.2 bundled with CUDA used):   https://github.com/thrust/thrust
	3. CUDA 6.5+ (v7.5 used)                    :   https://developer.nvidia.com/cuda-downloads

This code makes use of the following CUDA packages:
	
	1. cuRAND
	2. cuBLAS
	3. cuSolver
	

## REQUERIMENTS  

--------------------  
Needs an NVIDIA GPU with compute capability sm_2.0+  
Needs g++ with full C++11 support, 4.8+ recommended  

## TESTED ON  

------------
	 - GTX980 (sm_52)  on Ubuntu 14.04 with CUDA 7.5 and g++ 4.8
     - GTX980 (sm_52)  on Ubuntu 16.04 with CUDA 7.5 and g++ 5.3.1
     - GTX980 (sm_52), GTX780 (sm_35), GTX480(sm_20) and GTX580(sm_20) on CentOS 6.5 with CUDA 7.5 and g++ 4.8
	 - GTX1080 (sm_61), Tesla P1000 (sm_60) on CentOS 6.5 with CUDA 8.0 and g++ 4.8
     - K40 (sm_35), GTX780(sm_35) on CentOS 6.5 with CUDA 8.0 and g++ 4.8



## NOTES FOR DEVELOPERS

The procedure to implement a new module is the following:

	1. Create a new class that inherits from one of the parents (Interactor, Integrator, Measurable...) and overload the virtual methods. You can do whatever you want as long as the virtual methods are overloaded.	
	2. Take as input shared_ptr's to a ParticleData and a System at least, use them to interface with UAMMD (ask ParticleData for properties like pos, force, torque..)
	3. Include the new module in the source file that makes use of it
		
See available modules for a tutorial (i.e PairForces.cuh or VerletNVT.cuh)  
	

-------------------------------


In the creation of a new module (Interactor or Integrator) for interoperability with the already existing modules, the code expects you to use the variables from ParticleData when available, the containers storing the positions, forces, velocities... of each particle.  
These containers start with zero size and are initialized by ParticleData the first time they are asked for.  


**Guidelines**

Each module should be under the uammd namespace.

If you want to make small changes to an existing module, without changing it. Then you should create a new module that inherits it, and overload the necesary functions.

------------------------------------------

## ACKNOWLEDGMENTS

UAMMD was developed at the Departamento de Física Teórica de la Materia Condensada of Universidad Autónoma de Madrid (UAM) under supervision of Rafael Delgado-Buscalioni. Acknowledgment is made to the Donors of the American Chemical Society Petroleum Research Fund (**PRF# 54312-ND9**) for support of this research and to Spanish MINECO projects **FIS2013- 47350-C05-1-R and FIS2013-50510-EXP**.

Acknowledgment is made to NVIDIA Corporation.

## Colaborators

Raul P. Pelaez is the main developer of UAMMD.  

Other people that have contributed to UAMMD:  

Marc Melendez Schofield  
Sergio Panzuela  
Nerea Alcazar  
