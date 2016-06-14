##DESCRIPTION

-----------------
Raul P. Pelaez 2016

A CUDA Molecular Dynamics code made into modules for expandability and generality.
It is coded into separated modules, with a Driver that can hold many modules in order to construct a simulation. For example, Driver could have a velocityVerlet module and and PairForces interactor module to create a molecular dynamics simulation. Or a DPD integrator module with Nbody interactor module, etc.

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

A Measurable is any computation that has to be performed between steps of the simulation, it can be any magnitud that is calculated from the simulation state (positions, forces..).
A measurable can compute the energy, the radial function distribution or any arbitrary computation that does not change the simulation state.

----------------

These objects are abstract classes that can be derived to create all kinds of functionality and add new physics. Just create a new class that inherits Interactor or Integrator and override the virtual methods with the new functionality.


Finally there is a Driver that puts them all together and controls the flow of the simulation.


#Currently Implemented

-----------------------
**Interactors:**

	1.Pair Forces: Implements hash (cell index) sort neighbour list construction algorithm to evaluate pair forces given some potential function, LJ i.e. Ultra fast
	2.Bonded forces: Allows to join pairs of particles via springs (Instructions in BondedForces.h)
	
**Integrators:**

	1.Two step velocity verlet
	2.Euler Maruyama Brownian dynamics
	3.Euler Maruyama Brownian dynamics with hydrodynamic interactions via Rotne Prager (WIP)

**Measurables**
	
	1.Energy Measurable. Computes the total, potential and kinetic energy and the virial pressure of the system

##USAGE

-------------------
If you dont have cub (thrust comes bundled with the CUDA installation) clone or download the v1.5.2 (see dependencies).
The whole cub repository uses 175mb, so I advice to download the v1.5.2 zip only.
The Makefile expects to find cub in /usr/local/cub, but you can change it. CUB doesnt need to be compiled.

Hardcode the configuration (Integrator, Interactor, initial conditions..) in Driver.cpp, set number of particles, size of the box, dt and time of the simulation in main.cpp.

Then compile with make and run

You may need to adequate the Makefile to you particular system

##DEPENDENCIES

---------------------
Depends on:

	1. CUB       (v1.5.2 used)                  :   https://github.com/NVlabs/cub
	2. thrust    (v1.8.2 bundled with CUDA used):   https://github.com/thrust/thrust
	3. CUDA 6.5+ (v7.5 used)                    :   https://developer.nvidia.com/cuda-downloads

This code makes use of the following CUDA packages:
	
	1. cuRAND
	2. cuBLAS
	3. cuSolver
	

##REQUERIMENTS

--------------------
Needs an NVIDIA GPU with compute capability sm_2.0+
Needs g++ with full C++11 support, 4.8+ recommended

##TESTED ON

------------
	 - GTX980 (sm_52)  on Ubuntu 14.04 with CUDA 7.5 and g++ 4.8
     - GTX980 (sm_52)  on Ubuntu 16.04 with CUDA 7.5 and g++ 5.3.1
