Integrator
------------
		   
.. doxygenfile:: Integrator/Integrator.cuh
   :project: uammd
   :sections: briefdescription detaileddescription
		


The Integrator interface exposes the following API:


.. doxygenclass:: uammd::Integrator
   :project: uammd
   :members:	     
   :protected-members:

.. cpp:namespace:: uammd
		   
After calling :cpp:any:`Integrator::forwardTime()` on a given Integrator the relevant particle properties (i.e. positions, velocities...) will be updated and can be accessed via :ref:`ParticleData`.


Usage
=========

This is just a base class that cannot be used by its own.
Children of this class are instanced in a :doc:`code using UAMMD <../SimulationFile>` and :ref:`Interactors <Interactor>` are added to it to configure a simulation.
The simulation is then advanced by calling the method :cpp:any:`Integrator::forwardTime()` any number of times.

Creation
~~~~~~~~


The arguments for the constructor of an Integrator may vary on a case by case basis (see the page for the particular one you want to use). Most Integrators, however, share the same two arguments of a shared_ptr to either a :ref:`ParticleData` or a :ref:`ParticleGroup` and a set of parameters via a structure called :code:`[ModuleName]::Parameters`.


Using an already available Integrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

  #include<uammd.cuh>
  ...
  //Modules often need parameters, which are set by passing an struct of the type ModuleName::Parameters
  //See the page for each particular module for a list of parameters.
  BD::EulerMaruyama::Parameters params;
  params.dt = 0.01;
  ...
  //In general an integrator needs a (shared_ptr to) ParticleData or a ParticleGroup and some Parameters
  auto bd = make_shared<BD::EulerMaruyama>(pd, params);
  ...
  //Now bd will take into account the interaction described by "an_interactor".
  bd->addInteractor(an_interactor);
  ...
  //Run 1000 steps
  fori(0,1000) bd->forwardTime();

Where make_shared creates a `shared_ptr <https://en.wikipedia.org/wiki/Smart_pointer#shared_ptr_and_weak_ptr>`_, a kind of smart pointer.

You can see a list of currently implemented Integrators in the sidebar.

Example: Methods available in any Integrator
==============================================

We will create a :ref:`Brownian Dynamics` Integrator and list the available methods for it. Note that all Integrators will provide the same methods.

.. code:: c++

   #include"Integrator/BrownianDynamics.cuh"
   int main(){
     int N = 100;
     auto pd = std::make_shared<ParticleData>(N);
     //Initialize particles here
     //....
     //Some arbitrary parameters
     BD::Parameters par;
     par.dt = 0.1;
     par.temperature = 0;
     par.viscosity = 1;
     par.hydrodynamicRadius = 1;
     auto bd = std::make_shared<BD::EulerMaruyama>(pd, par);
     //Once the Integrator is created (see the page for the particular one you need for initialization instructions) you can:
     //Add an Interactor to the Integrator.
     //This also adds it as an updatable, so there is no need to also call addUpdatable for Interactors.
     bd->addInteractor(some_interactor);
     //Take the simulation to the next time step
     bd->forwardTime();
     //Add to each particle (via ParticleData::getEnergy) the energy due to the Integrator (typically the kinetic energy)
     bd->sumEnergy();
     //Get a list of all the interactors in the Integrator
     // You will get a list of type: std::vector<std::shared_ptr<Interactor>>
     auto interactors =  bd->getInteractors();
     //Adds a ParameterUpdatable to the Integrator.
     bd->addUpdatable(an_updatable);
     //Get a list of all the updatables in the Integrator
     auto updatables = bd->getUpdatables();
     return 0;
   }


Writing a new Integrator module
===============================

In order to create a new Integrator module, write a class that inherits from it and overrides its virtual functions. You will then have access to all its members, and will be able to use it as an Integrator.
See :ref:`VerletNVE` for an example of an integrator.

Whenever a module needs a particle property (i.e the position of the particles). It must ask for it to :ref:`ParticleData` in the following way:

.. code:: c++

  //The scope of pos must be the scope of the usage of pos_ptr, never store pos or any other particle property reference, always ask pd for them when you need them and release then when you are done using them.
  auto pos = pd->getPos(access::gpu, access::readwrite);
  real4* pos_ptr= pos.raw();


You can go through every Interactor with this construction, for example to sum the forces:

.. code:: c++

  for(auto forceComp: interactors) forceComp->sum({.force=true, .energy=false, .virial=false},cudaStream);

Where :cpp:`cudaStream` is a `CUDA stream <https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/>`_; 0, also known as the default stream, is a valid CUDA stream and will result in all GPU operations running one after the other. If a different stream is passed it is advisable for it to be the same for all Interactors.
In a similar manner you can inform the interactors of changes in parameters using the :ref:`ParameterUpdatable` interface.
The :cpp:`updatables` member holds a list of :ref:`ParameterUpdatable`-derived objects in the Integrator, including the Interactors and any other :ref:`ParameterUpdatable` object added:

.. code:: c++

  for(auto updatable: updatables){
  updatable->updateSimulationTime(steps*dt);
  updatable->updateTemperature(temperature);
  ...
  }


A basic Integrator
~~~~~~~~~~~~~~~~~~~


Here you have a bare bones template for an Integrator that you could follow:

.. code:: c++

  class MyIntegrator: public Integrator{
    real time, dt=0.1;
  public:
    MyIntegrator(shared_ptr<ParticleData> pd,
                 ,...Whatever I need...): Integrator(pd, "MyIntegrator"){
    ...
    }
    //Take the simulation to the next time step
    virtual void forwardTime() override{
      time += dt;
      //Integrators have access to the member "updatables", holding a list of ParameterUpdatables.
      //Note that this includes the Interactors as well.
      for(auto updatable: updatables) updatable->updateSimulationTime(time);
      //Before computing the new forces we probably want to fill the current ones with zero:
      {
        auto force = pd->getForce(access::gpu, access::write);
	thrust::fill(force.begin(), force.end(), real4());
      }
      //Integrators have access to the member "interactors", holding a list of Interactors
      for(auto forceComp: interactors){
        //forceComp->updateSimulationTime(time); //This call is redundant, since the interactor is already added in updatables
        forceComp->sum({.force=true, .energy=false, .virial=false});
      }
      auto pos = pd->getPos(access::cpu, access::readwrite);
      auto force = pd->getForce(access::cpu, access::read);
      //Update positions, for instance with a forward Euler rule
      for(int i=0; i<pos.size(); i++) pos[i] += force[i]*dt;
    }
    //Sum any energy due to the integrator (i.e kinetic energy)
    virtual real sumEnergy()override { return 0;}

  };
