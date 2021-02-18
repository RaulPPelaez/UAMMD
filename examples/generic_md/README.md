## A generic simulation using UAMMD

The code in this folder produces an executable capable of performing simulations with most schemes available in UAMMD.  
This program allows to construct simulations using the following modules via a data.main configuration file:  

Integration schemes (one must be chosen):  

   * **VerletNVT**: Molecular dynamics with a thermostat.  
   * **VerletNVE**: Molecular dynamics with constant energy (starting with a velocity distribution to match the given temperature).  
   * **BD**: Brownian dynamics  
   * **BDHI**: Brownian dynamics with Hydrodynamic interactions (via positively split Ewald Rotne-Prager-Yamakawa or Force Coupling Method)  
   * **SPH**: Smooth Particle Hydrodynamics  
   * **DPD**: Dissipative Particle Dynamics  
	
Interaction modules:  

   * Short range interaction: *LJ potential,  
   * Bonded forces: Pairs of particles joined by harmonic bonds*.  
   * Angular bonds: Groups of three particles joined by angular bonds.*  
   * Torsional bonds: Groups of four particles joined by torsional bonds.*  
   * External potential: An external force acting on each particle. Gravity + a wall in this example*  
   * Electrostatics: Periodic electrostatics using an Ewald splitted spectral Poisson solver.  
  
*The different potentials are taken from the file customizations.cuh accompanying this code, you can modify it to your needs.  

With this components as-is one can simulate systems from a microscale LJ-liquid with VetletNVE to a macroscale chunk of an ocean with SPH.   


### USAGE  
Compile with ```$ make```, you might have to customize the Makefile first to adapt it to your system.  
If you are having trouble with it, start by going into [Compiling UAMMD](https://github.com/RaulPPelaez/UAMMD/wiki/Compiling-UAMMD)  

Then execute ```$ ./generic```. If you have a valid CUDA environment this will generate an example data.main file and then halt.  
Modify this file as desired and then run ```$ ./generic``` again.  

The default data.main will perform a LJ-liquid simulation with a VerletNVT integrator.  

Additional information can be found in the header comments in generic_simulation.cu and customization.cuh.  

When a simulation is needed that cannot be constructed by simply modifying the data.main, customization.cuh offers a lot of options and information into how to further adapt the code to your needs.  


### Additional notes:  
All simulations that can be performed via this code assume a periodic box. But you can put an arbitrarily large box size. Keep in mind though that some modules encode algorithms that are inherently periodic, like electrostatics and hydrodynamics. These inifinite ranged interactions make a big box not equivalent to an open system.  
  
This code is also intended to be a skeleton code if you are trying to write some kind of UAMMD simulation code. For example, the function initialize (which starts the basic UAMMD structures and the particles) is probably going to appear everytime you write an UAMMD simulation.  
If you need to create a BD integrator, you can copy paste the function createIntegratorBD.  
Need to read parameters from a file? just copy paste the function readParameters.  
And so on.  
  
If you would like a more bottom up approach to UAMMD, you can surf the examples/basic folder, which will give you through UAMMD with an increasingly complex set of example codes.  
Additionally, you can drop by the wiki:   
https://github.com/RaulPPelaez/UAMMD/wiki  
Which has a lot of information. From basic functionality to descriptions and references for the algorithms implemented by each module.    

This code is an example on how to work inside UAMMD's simulation ecosystem, however UAMMD is not restricted to this kind of usage.  
It is designed to be as hackable as possible and exposes a lot of the core functionality as stand alone with the intention of making it work as an external GPU accelerator in other codes.  
See the folder examples/uammd_as_a_library for examples on how to do this.  
