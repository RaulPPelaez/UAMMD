Interactor
===========

Interactor is one of the base modules of UAMMD.

Interactor encapsulates the concept of a group of particles interacting, either with each other or with some external influence.
An Interactor can be issued to compute, for each particle, the forces, energies and/or virial due to a certain interaction.
To do so it can access the current state of the particles (like positions, velocities, etc) via :ref:`ParticleData`.

The Interactor interface class has the following API:

..
   .. cpp:class:: Interactor

      .. cpp:function:: Interactor(std::shared_ptr<ParticleData> pd, std::string name = "noName");

	 Constructor

      .. cpp:function:: virtual void sum(Interactor::Computables comp, cudaStream_t st = 0) = 0;

	Computes the forces, energies and/or virials on each particle according to the interaction. Adds the results to the relevant arrays in the :ref:`ParticleData` instance that was provided to it at creation.

	:param comp: An interactor is expected to update the properties of the particles in :ref:`ParticleData` for the members of :cpp:any:`Interactor::Computables` that are true.
	:param st: (Optional) A CUDA stream.

      .. cpp:function:: std::string getName();

	Returns the given name of the Interactor.



      .. cpp:type:: Computables

	 A POD structure containing a series of booleans like :cpp:`force`, :cpp:`energy` and :cpp:`virial`. Used to denote computation requirements for a function across UAMMD. For instance, the function :cpp:any:`Interactor::sum` takes a Computables as argument to inform about what the Interactor is supposed to compute.
	 New computables can be added at compile time by populating the :code:`EXTRA_COMPUTABLES` preprocessor macro. See below

	 .. cpp:member:: bool force

		      Defaults to :cpp:`false`.


	 .. cpp:member:: bool energy

		      Defaults to :cpp:`false`.


	 .. cpp:member:: bool virial

		      Defaults to :cpp:`false`.

	 .. cpp:member:: bool stress

		      Defaults to :cpp:`false`.
		   
.. doxygenclass:: uammd::Interactor
   :project: uammd
   :protected-members:	     
   :members:

      
.. hint:: Interactors can subscribe to the :ref:`ParameterUpdatable` interface, which :ref:`Integrator` will use to communicate changes in parameters (such as the current simulation time).

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
  

The Computables type in the :code:`sum` function simply contains a list of boolean values describing the needs of the caller (which will typically be an Integrator). As of today, an Interactor can be asked to compute only forces, energies and or virials acting on the particles. The Computables structure exists also to facilitate the future inclusion of additional quantities to the Interactor responsibilities.

Note that Interactor is what is called a pure-virtual class in C++ (and programming in general). This means that Interactor is not a class that can be used by itself (such as, for instance, ParticleData). It is a conceptual base class that must be inherited

Any class inheriting from Interactor will have access to an instance of :ref:`System` with the name :code:`sys`, that can be used to query properties of the GPU and log messages, and a :ref:`ParticleData` instance with the name :code:`pd`.

Available Interactors
----------------------

You can see a list of implemented Interactors in the side bar.


