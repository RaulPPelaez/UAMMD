## UAMMD examples  
In these folders you will find a lot of examples showcasing the different capabilities of UAMMD.  

UAMMD is presented with two distinct identities, it can work as a standalone molecular simulation GPU code (such as gromacs or hoomd) or as a library exposing GPU algorithms for you to use to accelerate your own code. Keep in mind that UAMMD is designed to be hackable and copy pastable, so feel free to take what you want and give nothing back!  

This folder is organized in the following way:  

### basic_concepts  
Contains a tutorial following the basic functionality of UAMMD with a bottom-up approach.  
This should be the first place to go if you are trying to learn how to use UAMMD.  

### advanced  
Stuff not covered by the basic tutorial with more complex and obscure functionality.  

### generic_md  
This code has almost every module that UAMMD offers in a single file that you can compile and then tweak via a data.main parameter file.  
If you are not looking to learn how UAMMD works and how to modify it or extend it get in here.  
You might be able to encode your simulation by tweaking the data.main.  

### integration_schemes  
The basic tutorial covers only a couple of integration modules, in this folder you will find copy pastable functions to create any UAMMD Integrator. From Brownian Dynamics to Smoothed Particle Hydrodynamics.  

### interaction_modules  
In a similar way, the tutorial only gives you one example of an interaction. Luckily once you know how to use one the rest come in similar form. You can find here copy pastable examples for every interaction module.  

### uammd_as_a_library  
This family of examples shows off want you can do outside the UAMMD simulation ecosystem, with a couple of includes you can obtain a neighbour list from a list of positions or expose a section of UAMMD to python (or any other language really).  

### misc  
Examples that do not fit in the other categories.  

