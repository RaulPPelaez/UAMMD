# **Universally Adaptable Multiscale Molecular Dynamics (UAMMD) ver 0.5**

<img src="https://github.com/raulppelaez/uammd/blob/master/.res/poster.png" width="300"><img src="https://github.com/raulppelaez/uammd/blob/master/.res/shotlogo.png" width="500">  


**See the wiki for more info!**  
**You can find videos on the youtube channel**  http://bit.ly/2r5WoOn

## DESCRIPTION  

-----------------  

Raul P. Pelaez 2016. (raul.perez(at)uam.es)  


A header-only fast generic multiscale CUDA Molecular Dynamics framework made into modules for expandability and generality.  

UAMMD can perform several types of simulations, to this moment multiple integrators are implemented allowing it to perform:  

	-Molecular dynamics (MD)  
	-Brownian Dynamics  (BD)  
	-Brownian Hydrodynamics (BDHI)  
	-Dissipative Particle Dynamics (DPD)  
	-Smoothed Particle Hydrodynamics (SPH)  

Multiple building blocks are provided in order for the user to construct a certain simulation, 
highly templated so the user can easily add in the input the specific interactions when they are not implemented by default.  

For example, there is not a harmonic trap module, but you can write a simple functor in the input file (directly in device code!) telling that each particle should experiment a force when it is trying to leave the box and you are set!. You can do the same with a bonded force, an interaction that needs to transverse a neighbour list, an nbody interaction... See the examples folder and the wiki for more info!  

UAMMD is coded into separated types of modules. A code that uses UAMMD needs to create/instantiate some of this modules and update them when necessary (i.e to forward the simulation time). For example, the simulation could have a VerletNVT integrator module and a PairForces interactor module to create a molecular dynamics simulation. Or a DPD integrator module with Nbody interactor module, etc. See the example folder.  

There are two basic types of modules:  

      1. Integrators  
      2. Interactors  

**Interactors**

An Interactor is an abstract entity that has the ability of computing the forces, energies... acting on each particle due to some interaction.  
For example, an Interactor could compute the pair Lennard Jonnes forces between each particle pair of the system or sum the forces due to the particles being joined by springs.  

**Integrators**

An Integrator is an entity that has the ability of taking the simulation state to the next next time step.  
In order to do so it can hold any number of Interactors and use them to compute the forces, energies... at any time.  
For example, the VerletNVT module updates the positions and velocities of particles according to the interactors it holds to ensure that the temperature is conserved each time the simulation time is updated.  

----------------------  

These objects are abstract classes that can be derived to create all kinds of functionality and add new physics. Just create a new class that inherits Interactor or Integrator and override the virtual methods with the new functionality. See any of the available modules for an example, like ExternalForces.cuh 


# Currently Implemented

See the wiki page for each interactor for more info and instructions!

-----------------------
**Interactors:**

	1.Pair Forces: Implements hash (Morton hash) sort neighbour cell list construction algorithm to evaluate pair forces given some potential function, LJ i.e. Ultra fast. In the case of the range of the potential being too large, it falls back to NBodyForces.
	2.Bonded forces: Allows to join pairs of particles via bonds (i.e a harmonic spring) (Instructions in BondedForces.cuh)
	3.Angle bonded forces: Allows to join triples of particles via semiflexible springs (Instructions in AngularBondedForces.cuh)
    4.NBody forces: All particles interact with every other via some potential.
	5.External forces: A custom interaction function that will be applied to each particle individually.
	
**Integrators:**

	1.Two step velocity verlet NVE
	2.Two step velocity verlet NVT with BBK thermostat
	3.Euler Maruyama Brownian dynamics (BD)	
	4. Brownian Dynamics with Hydrodynamic interactions (BDHI)
	4.1 Euler Maruyama w/HI via RPY tensor 
	4.1.1 Using the Cholesky decomposition on the full Mobility matrix to compute the stochastic term. Open Boundaries.
	4.1.2 Using the Lanczos algorithm and a matrix free method to compute the stochastic term. Open Boundaries.
	4.1.3 Using the Positively Split Ewald method with rapid stochastic sampling. Periodic Boundary Conditions
	5.Smoothed Particle Hydrodynamics (SPH) WIP!

----------------------

You can select between single and double precision via global/defines.h. Single precision is used by default, remember to recompile the entire code when changing the precision. This last step is very important, as failing to do so will result in unexpected behavior.


## USAGE

-------------------
If you dont have cub (thrust comes bundled with the CUDA installation) clone or download the v1.5.2 (see dependencies).
The whole cub repository uses 175mb, so I advice to download the v1.5.2 zip only.  

**UAMMD does not need to be compiled (it is header only)**.  

To use it in your project, include the modules you need, create a System and ParticleData instances and configure the simulation as you need.  
See examples/LJ.cu and examples/Makefile for a tutorial!  

In order to compile a source file that uses UAMMD, you only have to inform the compiler of the location of the project (with -I) and give the flag "--expt-relaxed-constexpr" to nvcc.  
See examples/Makefile for an example.  

You can use the --device X flag to specify a certain GPU.  


## DEPENDENCIES

---------------------
Depends on:

	1. CUB       (v1.5.2 used)                  :   https://github.com/NVlabs/cub
	2. thrust    (v1.8.2 bundled with CUDA used):   https://github.com/thrust/thrust
	3. CUDA 6.5+ (v7.5 used)                    :   https://developer.nvidia.com/cuda-downloads
	4. BOOST 1.58.0+                            :   http://www.boost.org/
This code makes use of the following CUDA packages:
	
	1. cuRAND
	2. cuBLAS
	3. cuSolver
	
	
## REQUERIMENTS  

--------------------  

Needs a c++ compiler with full C++11 support, 4.8+ recommended  

## TESTED ON  

------------
	 - GTX980 (sm_52)  on Ubuntu 14.04 with CUDA 7.5 and g++ 4.8
     - GTX980 (sm_52)  on Ubuntu 16.04 with CUDA 7.5 and g++ 5.3.1
     - GTX980 (sm_52), GTX780 (sm_35), GTX480(sm_20) and GTX580(sm_20) on CentOS 6.5 with CUDA 7.5 and g++ 4.8
	 - GTX1080 (sm_61), Tesla P1000 (sm_60) on CentOS 6.5 with CUDA 8.0 and g++ 4.8
     - K40 (sm_35), GTX780(sm_35) on CentOS 6.5 with CUDA 8.0 and g++ 4.8
     - Same as above with CUDA 9.0 and CUDA 8.0


## NOTES FOR DEVELOPERS

The procedure to implement a new module is the following:

	1. Create a new class that inherits from one of the parents (Interactor, Integrator...) and overload the virtual methods. You can do whatever you want as long as the virtual methods are overloaded.	
	2. Take as input shared_ptr's to a ParticleData and a System at least, use them to interface with UAMMD (ask ParticleData for properties like pos, force, torque..)
	3. If the new module needs a new particle property (i.e torque) include it in ParticleData.cuh ALL_PROPERTIES_LIST macro
	4. If the new module needs to communicate a new parameter change to all modules (i.e it changes the simulation box with time) include it in ParameterUpdatable.cuh  PARAMETER_LIST macro	
	5. Include the new module in the source file that makes use of it
		
See available modules for a tutorial (i.e PairForces.cuh or VerletNVT.cuh)  

Some things to take into account:
	
	1. ParticleData can regularly update the particle order and/or the number of particles, it will communicate this changes through signals. See ParticleData.cuh for a tutorial on how to handle a reorder.
	2. ParticleData can also change the storage location of the particle arrays, so do not store raw pointers to particle properties, always ask PD for them before using them with ParticleData::get*()
	3. In the modules where it makes sense, make them be able to handle ParticleGroups (which will contain all particles by default). See PairForces.cuh for an example of a module handling ParticleGroups.
	4. UAMMD usually uses the lazy initialization scheme, nothing is initialized unless it is absolutely necessary. For example, the CPU version of a particle property (and the GPU version FWIW) will not be allocated until someone asks for it with pd->get*().  
	
Some advice:

	1. Make use of the existing modules and submodules when possible, inherit from them if you need an extra level of control. For example with a neighbourList.
	2. Use cub when possible.
	3. When constructing a new kind of simulation compile the modules in one file and compile another separate one for using the first (to reduce compilation time), or better yet make the code read all needed parameters from a file or script.
	4. Use uammd::GPUfillWith instead of cudaMemset (it is MUCH faster)
	5. Use the iterator scheme and the full extent of C++11 philosophy whenever possible.

-------------------------------


In the creation of a new module (Interactor or Integrator) for interoperability with the already existing modules, the code expects you to use the variables from ParticleData when available, the containers storing the positions, forces, velocities... of each particle.  
These containers start with zero size and are initialized by ParticleData the first time they are asked for.  


**Guidelines**

Each module should be under the uammd namespace. And if helper functions are needed, they should be under another, module specific, namespace.  
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

