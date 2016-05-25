##DESCRIPTION
Raul P. Pelaez 2016

A CUDA Molecular Dynamics code that implements hash (cell index) sort neighbour list construction algorithm to evaluate pair forces.

This work in progress is intended to be used as a submodule that only computes the forces acting on each particle given the positions.

Currently it is a full MD code that computes the forces and integrates the trajectories using a two step velocity verlet algorithm.

##USAGE

Hardcode the configuration in main.cpp, set number of particles, size of the box, dt and time of the simulation.

The particles will start in a cubic lattice unless an initial configuration is readed using psystem->write(fileName);

Then compile with make and run


##DEPENDENCIES

Depends on:

	1. CUB       (v1.5.2 used)                  :   https://github.com/NVlabs/cub
	2. thrust    (v1.8.2 bundled with CUDA used):   https://github.com/thrust/thrust
	3. CUDA 6.5+ (v7.5 used)                    :   https://developer.nvidia.com/cuda-downloads


##REQUERIMENTS

Needs an NVIDIA GPU with compute capability sm_2.0+

##TESTED ON
	 - GTX980 on Ubuntu 14.04