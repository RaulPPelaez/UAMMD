.. UAMMD documentation QuickStart file, created by
   RaulPPelaez on Fri Jan 14 11:28:46 2022.

Quick Start
=================================


UAMMD is presented under two distinct identities, it can work as a standalone molecular simulation GPU code (such as gromacs or hoomd) or as a library exposing GPU algorithms for you to use to accelerate your own code.  
Keep in mind that UAMMD is designed to be hackable and copy-pastable, so feel free to take what you want!  

There are therefore two choices when first encountering UAMMD, are you looking for a stand alone molecular dynamics package or are you looking to accelerate your existing code with some of the algorithms exposed by UAMMD?


UAMMD as a standalone framework    
-------------------------------------------
Go to the examples/generic_md folder and follow the instructions, you will end up with a single binary that can be configured to run a variety of simulations. If you are lucky you will be all set and if the family of default interactions is not enough you might still be able to tweak them to suit your needs.     

UAMMD as a library  
---------------------------------

Go to the examples/uammd_as_a_library folder. You will find some copy-pastable codes that you an use to accelerate your existing code, such as a GPU neighbour list or a way to expose any UAMMD module to python.  
 
Extending or adapting UAMMD
----------------------------------

If your are looking to learn how UAMMD works the best place to star is the tutorial under examples/basic_concepts, follow it and you will have every tool needed to modify it to suit your needs.  


Additional notes
-----------------------------------
In this wiki you will find a comprehensive list of modules with a detailed description on what they do and how to use them.  
