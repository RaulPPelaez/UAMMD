### Tutorial examples  
This folder contains a series of examples of increasing complexity, detailing the usage of the different parts of the UAMMD ecosystem.  
When followed, this codes will teach you to go from an empty C++ program to an UAMMD powered LJ-liquid simulation, with the tools to modify it to anything that UAMMD offers.  
Take into account that this is not all that you can do with UAMMD, since a lot of functionality will be ommited for simplicity.  
The folder examples/advanced showcases these topics not covered here.  

UAMMD is written in C++ and makes extensive use of the C++14 standard, I would say at least a basic level of familiarity with the language is needed to follow this tutorial.  
Although it is not strictly needed to understand most of these examples, a basic knowledge of GPU programming is highly advisable.  

The tutorial in these codes follow a bottom-up approach, introducing concepts one by one from the ground up.  
If you would rather start from a more elaborate example and learn by playing around with it I suggest you start from examples/generic_md, which presents a sofisticated simulation in a hackable and copy-pastable manner.  

On the other hand, if you are looking to use UAMMD more as a black box, without the intention of writing or reading any code at all, also check out examples/generic_md. Once compiled that example allows to perform a lot of different simulations.  

If you want additional information about a topic covered in one of the tutorials check it's [documentation](https://uammd.readthedocs.io) page.  

### USAGE  
Compile with ```$ make```, you might have to customize the Makefile first to adapt it to your system.  
If you are having trouble with it, start by going into [Compiling UAMMD](https://uammd.readthedocs.io/en/latest/Compiling-UAMMD.html)  
Since there are a lot of examples, consider passing the option ```-j X``` to make so it compiles X files in parallel.  

Now start by reading the code **1-system.cu** and then running ```$ ./1-system```. Then go from there to **2-hello-world.cu** and so on.  
