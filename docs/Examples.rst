Examples
=========

The `examples <https://github.com/RaulPPelaez/UAMMD/tree/v2.x/examples>`_ folder in the UAMMD repository contains a lot of examples showcasing the different capabilities of UAMMD.  

UAMMD is presented with two distinct identities, it can work as a standalone molecular simulation GPU code (such as gromacs or hoomd) or as a library exposing GPU algorithms for you to use to accelerate your own code. Keep in mind that UAMMD is designed to be hackable and copy pastable, so feel free to take what you want and give nothing back!  

The examples/ folder is organized in the following way:  

basic_concepts
----------------
Contains a tutorial following the basic functionality of UAMMD with a bottom-up approach.  
This should be the first place to go if you are trying to learn how to use UAMMD.  

advanced
-------------
Stuff not covered by the basic tutorial with more complex and obscure functionality.  

generic_md
---------------
This code has almost every module that UAMMD offers in a single file that you can compile and then tweak via a data.main parameter file.  
If you are not looking to learn how UAMMD works and how to modify it or extend it get in here.  
You might be able to encode your simulation by tweaking the data.main.  

integration_schemes
-----------------------
The basic tutorial covers only a couple of integration modules, in this folder you will find copy pastable functions to create any UAMMD Integrator. From Brownian Dynamics to Smoothed Particle Hydrodynamics.  

interaction_modules
---------------------
In a similar way, the tutorial only gives you one example of an interaction. Luckily once you know how to use one the rest come in similar form. You can find here copy pastable examples for every interaction module.  

uammd_as_a_library
--------------------
This family of examples shows off want you can do outside the UAMMD simulation ecosystem, with a couple of includes you can obtain a neighbour list from a list of positions or expose a section of UAMMD to python (or any other language really).  

misc
---------
Examples that do not fit in the other categories.  



Compiling the examples
-----------------------

The examples/ folder contains both a Makefile and a CMakeLists.txt file as examples, both will try to compile every example.
The Makefile will go into each subfolder and compile each source code to a binary in its folder. To use it simply  cd to examples/ and run:

.. code:: bash

	  $ make

You might need to edit the Makefile with the particular configuration of your system. You can also tweak here UAMMD variables such as the log level or the floating precision.

The CMake option will instead dump every binary inside a bin/ folder. Use it as a standard CMake file:

.. code:: bash

	  $ mkdir -p build && cd build
	  $ cmake ..
	  $ make

You can pass to CMake any UAMMD related definitions, for instance:

.. code:: bash

	  $ mkdir -p build && cd build
	  $ cmake -DDOUBLE_PRECISION ..
	  $ make
