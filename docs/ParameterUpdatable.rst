ParameterUpdatable
===================

.. cpp:class:: ParameterUpdatable

	  A parameter communication interface, anything that inherits from ParameterUpdatable can be called through a series of update* methods to communicate a parameter change. Parameters related with the particles' data are communicated using :cpp:class:`ParticleData` (like number of particles).  


Parameters like a simulation box or the current simulation time are updated through ParameterUpdatable. See utils/ParameterUpdatable.h for a list of parameters handled by this interface.   

Every :cpp:any:`Interactor` is a ParameterUpdatable by default.  

If a module needs to be aware of a parameter change, it should define the particular update method. Which will do nothing by default.  

If a module needs to delegate the ParameterUpdatable behavior to a member (i.e :ref:`ExternalForces` to the provided potential) it must then inherit from ParameterUpdatableDelegate and call :code:`setDelegate(&member)`. From that moment, calls to the :code:`update*()` family of functions will be called on "member".  
This will work (although will do nothing) even when "member" is not ParameterUpdatable (through SFINAE).   



   


Usage
---------

Example of a class that is ParameterUpdatable:  

.. code:: cpp
	  
  class User: public ParameterUpdatable{
      Box myBox;
      real time;
    public:
      User(){}
      virtual void updateBox(Box newBox) override{
        myBox = newBox;
      }
      virtual void updateSimulationTime(real newtime) override{
        time = newtime;
      }
      //All the other update* functions do not need to be overrided, calls to them on User will just be ignored.
  
  };

To update some parameter of some ParameterUpdatable entity:  

.. code:: cpp
	  
 //Imagine an Integrator wants to inform all its updatables of the current simulation time after a step
  time += dt;
 for(auto inte: updatables) inte->updateSimulationTime(time); 
 //Now all its updatables have been informed of the new simulation time.
 //Some will do something with this information, some will just ignore the function call, which is the default behavior. 


Every :cpp:any:`Integrator` is expected to call update* on its list of updatables (which includes all Interactors added to it) with every parameter that changes during the course of the simulation. That means that, among other things, the first thing happening at each step will typically be a call to :cpp:`updateSimulationTime()` for each updatable.   

Advanced usage
----------------

Sometimes a certain User class may not be ParameterUpdatable itself, but would rather pass the ParameterUpdatable behavior to a member. In this case User will be a ParameterUpdatableDelegate. From the outside the User class will appear to be ParameterUpdatable, but the functionality will be redirected to a certain member.   
Here you have an example of a class that is ParameterUpdatableDelegate:   

.. code:: cpp
	  
  template<class Child>
  class User: public ParameterUpdatableDelegate<Child>{
      
    public:
      User(shared_ptr<Child> child){
        //This must be called to register a Child instance to call update* on
        this->setDelegate(child);
      }
      //User may need to handle some update*, 
      //if an update* function is overrided in User it will have the priority over Child.
      virtual void updateSimulationTime(real newtime) override{
        time = newtime;
        //Call the Child update function.
        ParameterUpdatableDelegate<Child>::updateSimulationTime(newtime);
      }
      //All the other update* functions will be calls to the update* functions in Child. If an update* function is not present here nor in Child, the call will be ignored.
  
  };
  
