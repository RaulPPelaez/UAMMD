## ABOUT THIS SIMULATION:

A group of equally charged particles are let to evolve in a slab via Brownian Dynamics. Two charged walls are located at the bottom and top of the slab, their charge is equal and with a value opposite to that of the particles such that the overall system particles+walls is electroneutral.  
The three regions demarcated by the walls (top, inside, below) can have different permittivities.  
In the reference data.main, the three domains have the permittivity of water.  
Particles repell each other and the walls via a softened, LJ-like potential.  
See data.main and PoissonSlab.cu for more information.  

## USAGE:

Ensure PoissonSlab.cu is compiled and available in the father directory "../" as "poisson".  
A data.main must be available in the same folder as test.bash  

Run test.bash  

This script will generate starting positions using tools/init.sh.  
Then run the simulation according to the parameters in data.main.  
It will then histogram the charges height using all generated snapshots and compute a density profile.  
Finally the PNP theoretical density profile will be generated.  

All results will be placed under a results folder.  
When finished, results/density.dat and results/density.theo should contain similar values.  

Theory depends on the parameter "K" in test.bash. This number depends on density and should be changed accordingly for the comparison to be valid.  
