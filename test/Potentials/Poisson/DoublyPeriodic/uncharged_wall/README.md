### ABOUT THIS SIMULATION:

A group of equally charged particles with random sign (but ensuring electroneutrality) are let to evolve in a slab via Brownian Dynamics. Two uncharged walls are located at the bottom and top of the slab.  
The three regions demarcated by the walls (top, inside, below) can have different permittivities.  
In the reference data.main, the inside and top have water permittivity and there is vacuum below.  
Particles repell each other and the walls via a LJ-like potential.  
See data.main and PoissonSlab.cu for more information.  

### USAGE:

Ensure PoissonSlab.cu is compiled and available in the father directory "../" as "poisson".  
A data.main must be available in the same folder as test.bash  

Run test.bash  

This script will generate starting positions using tools/init.sh.  
Then run the simulation according to the parameters in data.main.  
It will then histogram the charges height using all generated snapshots and compute a density profile.  
If the data.main is not modified the results can be compared with fig. 4 in [1].  

Results will be placed under a results folder.  

[1] T. Croxton, D. A. McQuarrie, G. N. Patey, G. M. Torrie, and J. P. Valleau. Ionic solution near an uncharged surface with image forces. Canadian Journal of Chemistry-Revue Canadienne De Chimie, 59(13):1998–2003, 1981.
