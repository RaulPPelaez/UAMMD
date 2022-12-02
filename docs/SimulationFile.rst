Simulation file
-----------------

Any file that includes uammd.cuh is considered an UAMMD simulation code.

See :ref:`Compiling UAMMD` on how to compile a file that includes UAMMD.

A minimal input file that initializes the basic UAMMD objects (:ref:`System` and :ref:`ParticleData`) and does nothing:

.. code:: cpp
	  
  //Compile with: nvcc -std=c++14 -I$(UAMMD_SRC)/src  -I$(UAMMD_SRC)/src/third_party 
  #include"uammd.cuh"
  using namespace uammd;
  int main(int argc, char * argv[]){
    int numberParticles = 1<<14;
    //Creating a System will initialize uammd. All modules need a system to work. Most of them will ask ParticleData for it.
    auto sys = make_shared<System>(argc, argv);
    //ParticleData will hold arrays for any particle property needed by any module, every module needs a ParticleData
    //ParticleData will create an instance of System if not provided with one
    auto pd = make_shared<ParticleData>(numberParticles, sys);
    //This call will ensure all UAMMD operations are gracefully terminated. 
    //When all modules are out of scope, any remaining memory allocated by UAMMD will be freed.
    sys->finish();
    return 0;
  }


For more interesting UAMMD examples, see the examples folder.

After ParticleData creation you may create any number of :ref:`Integrators <Integrator>` and/or :ref:`Interactors <Interactor>`.

In general, a module constructor will need:  
   * A :ref:`ParticleData` shared_ptr
   * A Parameters struct of type ModuleName::Parameters (you can see in the modules header or wiki page what each parameter does and what the default value is).  

For more info on c++ specific topics (like make_shared<>) see :ref:`here <Programming Tools>`.

For more on the available modules and how to use them see their respective wiki page. A list of all available modules can be found in the side bar.   

You can easily read parameters/options from a file using :ref:`InputFile`.  

