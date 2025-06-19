Interactor
===========

Interactor is one of the base modules of UAMMD.

Interactor encapsulates the concept of a group of particles interacting, either with each other or with some external influence.
An Interactor can be issued to compute, for each particle, the forces, energies and/or virial due to a certain interaction.
To do so it can access the current state of the particles (like positions, velocities, etc) via :ref:`ParticleData`.

The Interactor interface class has the following API:
		   
.. doxygenclass:: uammd::Interactor
   :project: uammd
   :protected-members:	     
   :members:



.. hint:: Interactors can subscribe to the :ref:`ParameterUpdatable` interface, which :ref:`Integrator` will use to communicate changes in parameters (such as the current simulation time).


.. warning:: Interactors should throw an exception if an unsatisfiable Computable is requested (for instance, due to a lack of implementation).
	     
Adding new computables:
-----------------------
			
.. doxygendefine:: EXTRA_COMPUTABLES
   :project: uammd		   
		 



Using an already available Interactor:
---------------------------------------

Each Interactor has its own requirements for initialization, visit the page for the one you are interested in for further instructions.

Once initialized all Interactors can be seamlessly added to any :ref:`Integrator`.

Let us see, for instance, how to create a :ref:`triply periodic electrostatics` Interactor and add it to an already created :ref:`Brownian Dynamics` Integrator:

.. code:: cpp
	  
  #include"Interactor/SpectralEwaldPoisson.cuh" //The Interactor for triply periodic electrostatics
  ...
  //Assume bd is a BD::EulerMaruyama Integrator previously created
  //Assume pd is a ParticleData
  //Each Interactor requires a different set of parameters, that exists in an structure called
  // ModuleName::Parameters
  Poisson::Parameters par;
  par.box = Box({128, 128, 128});
  par.epsilon = 1;
  par.gw = 1.0;
  par.tolerance = 1e-4;
  //The Interactor is created here by providing it with a ParticleData and the required parameters, which you can learn about in the page of the module page
  auto poisson = make_shared<Poisson>(pd, par);
  //At this point we can either issue the Interactor to compute the forces/energies/virials
  //poisson->sum({.force= true, .energy = false, .virial=false});
  //Or pass it to an Integrator
  bd->addInteractor(poisson);
  
Creating a new Interactor
---------------------------

In order to create a new Interactor module, write a class that inherits from it and overrides the :code:`sum` method. You will then have access to all its members, and will be able to use it as an Interactor for all intends and purposes.

See :ref:`PairForces` for an example of an Interactor.

Note that the :code:`sum` method is expected to update the relevant properties in :ref:`ParticleData` (for instance using :code:`pd->getForce(...)` if force is true in Computables).

A minimal example of an Interactor:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp
   
  #include<uammd.cuh>
  #include<Interactor/Interactor.cuh>
  using namespace uammd;
  
  //A class that needs to behave as 
  // an UAMMD Interactor must inherit from it
  class MyInteractor: public Interactor{
    public:
    //The constructor must initialize the base Interactor class, for which a ParticleData instance is required.
    //Other than that, it can take any necessary arguments (such as a group of parameters).
    MyInteractor(std::shared_ptr<ParticleData> pd):
            Interactor(pd, "MyInteractor"){
      //Any required initialization 
    }
  
    //An Interactor can be issued, mainly
    // by Integrators, to sum
    // forces, energies and/or virial
    // on the particles
    virtual void sum(Computables comp, cudaStream_t st) override{
      //"sys" and "pd" are provided by the Interactor base class
      sys->log<System::MESSAGE>("Computing interaction");
      if(comp.force){
        //Sum forces to each particle
        //For instance, adding a force to the x coordinate
        // of the first particle
        auto forces = pd->getForce(access::cpu, access::write);
        forces[0].x += 1;
      }
      if(comp.energy){
        //Sum energies to each particle
      }
      if(comp.virial){
        //Sum virial to each particle
      }
    }
  };

.. cpp:namespace:: uammd

Note that :cpp:any:`Interactor` is what is a pure-virtual class. This means that Interactor is not a class that can be used by itself (such as, for instance, :cpp:any:`ParticleData`). It is a conceptual base class that must be inherited from.

.. hint:: Any class inheriting from :cpp:any:`Interactor` will have access to an instance of :cpp:any:`System` with the name :code:`sys`, that can be used to query properties of the GPU and log messages, and a :cpp:any:`ParticleData` instance with the name :code:`pd`.

Available Interactors
----------------------

You can see a list of implemented Interactors in the side bar.


