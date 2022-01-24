
Frequently Asked Questions
==========================

 #.  `How can I access the different properties of each particle?`_
 #.  `How can I perform a simple Brownian Dynamics simulation?`_  
 #.  `What is all this particle reordering thing?`_  
 #.  `Why are the "force" container elements of type real4?`_  
 #.  `I need particles to have the X property not provided by UAMMD`_  
 #.  `How do I use a neighbour list?`_  
 #.  `I have an object (such as an Interactor) that needs to know when some state variable changes (temperature, simulation time, box size...)`_  
 #.  `I need a module acting only on a subset of particles`_

****


How can I access the different properties of each particle?
------------------------------------------------------------

The information about each particle is handled by the :doc:`ParticleData` class, every UAMMD module needs a pointer to an instance of this class, which is typically created early in the :doc:`simulation code <SimulationFile>`. This means you will usually have access to a ParticleData instance, usually called "pd".  
ParticleData provides an interface allowing to read and or modify the properties of each individual particle such as the position, force, mass... Refer to :doc:`ParticleData` for more information on how to do this.  
Here is some copy-pastable example:

.. code:: cpp

  //If a ParticleData instance is not available you can create one with:
  //int numberParticles = 1000;
  //auto pd = std::make_shared<ParticleData>(numberParticles);
  //Get a handle to particle positions, accesible from the cpu with the intention of reading its contents (no modification allowed):
  auto pos = pd->getPos(access::cpu, access::read);
  //Get the particle masses for writing from the gpu:
  auto mass = pd->getMass(access::gpu, access::write);


**Do not store these handles**, request them every time you need them and destroy them as soon as possible. Requesting a handle takes ownership of the property during the lifetime of the handle, so UAMMD is not allowed to use it while it exists.

*********

How can I perform a simple Brownian Dynamics simulation?
---------------------------------------------------------

#. Create an instance of :doc:`ParticleData`.  
#. Initialize particle positions.
#. Configure and create an instance of the :doc:`BD Integrator <Integrator/BrownianDynamics>`.
#. Call its forwardTime method as many times as necessary.


.. code:: cpp
	  
  #include"uammd.cuh"
  #include"Integrator/BrownianDynamics.cuh"
  using namespace uammd;
  int main(){
    int numberParticles = 1e6;
    auto pd = std::make_shared<ParticleData>(numberParticles);
    {
      auto pos = pd->getPos(access::cpu, access::write);
      fori(0, numberParticles)
        pos[i] = make_real4(sys->rng().uniform3(-32, 32), 0);
    }
    using BD = BD::EulerMaruyama;
    BD::Parameters par;
    par.temperature=1;
    par.viscosity=1;
    par.hydrodynamicRadius=1;
    par.dt=0.01;
    auto bd = std::make_shared<BD>(pd, par);
    int nsteps = 1e3;
    for(int i = 0; i<nsteps; i++){
      bd->forwardTime();
      //Do whatever between steps
    }
    return 0;
  }

**********

What is all this particle reordering thing?
--------------------------------------------

UAMMD might decide to sort the particles to increase the spatial locality of the data in memory. This causes particles to loose its initial indexing. This initial index of a particle is referred to as its id or name.  
The user can always keep track of particles via their id if needed, ParticleData is in charge of this, see :ref:`particle_id_assignation`.

*********

Why are the "force" container elements of type real4?
-------------------------------------------------------

Originally it was designed this way for the performance befits of this type in the CUDA architecture as opposed to float3. The fourth element is currently unused by UAMMD, although some modules might set it to zero when summing their force contributions. You might be able to use it for your purpose, but keep in mind that it is not guaranteed to be untouched by UAMMD.  
If you need a new per particle property you should append it to ALL_PROPERTIES_LIST in :doc:`ParticleData` as explained :ref:`here <list-of-available-properties>`.  


*************

I need particles to have the X property not provided by UAMMD
-----------------------------------------------------------------

Say you implement a new Integrator which needs a currently not present property of a particle, for example the torque. You might just treat this as an internal implementation detail of your Integrator and be done with it. But now you realize that this new Interactor you are writing also needs the same torque to compute the force/energy.  

The UAMMD way to do this would be to add the property "torque" to ParticleData as described :ref:`here <list-of-available-properties>`.  
By doing so, a family of functions will be auto generated to allow access to the new property "torque". These are, among others described in :doc:`ParticleData`, :code:`getTorque()`, :code:`isTorqueAllocated()`, etc.  
Now you can write to :code:`pd->getTorque()` in your Integrator and make use of it in your Interactor through the UAMMD provided interface.  
Adding new properties is not expected to have a negative performance impact, and no memory will be wasted when the property is unused so do not fear adding new properties to this list.   



************

How do I use a neighbour list?
--------------------------------

You can find the workings of UAMMD's neighbour lists :doc:`here <Interactor/NeighbourList>.
UAMMD offers several ways to interface with a neighbour list. The prefered way is to use a so-called :doc:`"Transverser" <Interactor/Transverser>`, although there are other ways. This structure provides the building blocks for a very generic computation using a neighbour list, in the below example you have a simple "neighbour counter" you can adapt.  

Here you have some copy pastable example on how to use a :doc:`CellList <Interactor/CellList>` (although any neighbour list will behave the same).

.. code:: cpp
	  
  #include"uammd.cuh"
  #include"Interactor/NeighbourList/CellList.cuh"
  #include<thrust/device_vector.h>
  using namespace uammd;
  
  //A transverser that counts particle pairs
  struct NeighbourCounter{
    struct Info{
      real3 nothingUseful;
    };
    
    struct returnInfo{
      int quantity;
    };
    
    NeighbourCounter(int* perParticleNeighbourCounter):
      perParticleNeighbourCounter(perParticleNeighbourCounter){}
  
    __device__ returnInfo zero(){ return {make_real3(0)};}
     
    __device__ Info getInfo(int pi){
      real3 somePerParticleQuantity = real3();
      Info info;
      info.nothingUseful = somePerParticleQuantity;
      return info;
    }
    
    __device__ returnInfo compute(real4 position_i, real4 position_j, Info infoi, Info infoj){
      int iJustCountANeighbour = 1;
      return {iJustCountANeighbour};
    }
    
    __device__ void accumulate(returnInfo &total, const returnInfo &current){
      total.quantity += current.quantity; 
    }
    
    __device__ void set(uint pi, returnInfo total){
      perParticleNeighbourCounter[pi] = total.quantity;
    }
  private:
    int* perParticleNeighbourCounter;
  };
  
  int main(){
    int numberParticles = 16384;
    real boxSize = 128;
    real cutOffDistance = 2.5;  
    auto pd = std::make_shared<ParticleData>(numberParticles);
    //... Initialization of positions, etc would go here
    auto nl = std::make_shared<CellList>(pd);
    cudaStream_t st = 0;
    Box box({boxSize,boxSize, boxSize});
    nl.updateNeighbourList(box, cutoffDistance, st);
    thrust::device_vector<int> perParticleNeighbourCounter(numberParticles);
    thrust::fill(perParticleNeighbourCounter.begin(), perParticleNeighbourCounter.end(), 0);
    NeighbourCounter neighbourCounter(thrust::raw_pointer_cast(perParticleNeighbourCounter.data()));
    nl.transverse(neighbourCounter, st);
    int numberOfNeighboursOfTheFirstParticle = perParticleNeighbourCounter[0];
    std::cerr<<"Number of neighbours of particle i=0: "<<numberOfNeighboursOfTheFirstParticle<<std::endl;
    return 0;
  }
  
There are other ways to get use a neighbour list besides a Transverser. for instance you can call :code:`getNeighbourList()` to get linear arrays with the information of neighbours or you can use the :doc:`NeighbourContainer <Interactor/NeighbourContainer>` interface (which is usually the fastest and the one that CellList internally uses).  

******************

I have an object (such as an Interactor) that needs to know when some state variable changes (temperature, simulation time, box size...)
-------------------------------------------------------------------------------------------------------------------------------------------
Use the :ref:`ParameterUpdatable` interface. Inheriting from Interactor will automatically inherit this capability and Integrators assume their Interactors need to be aware of any state variable they change, including the current simulation time (which is updated each time step).  
Furthermore you can add any ParameterUpdatable derived object to an :ref:`Integrator`. This will make that object aware of any changes. 

*********

I need a module acting only on a subset of particles
------------------------------------------------------

In modules (mostly :ref:`Interactors <Interactor>` and :ref:`Integrators <Integrator>`) where it makes sense for them to act only on a subset of particles will have an optional argument of type :ref:`ParticleGroup` in their constructor (replacing :ref:`ParticleData`).  
For example say that you have particles of two types (0 and 1), you may want an :ref:`ExternalForces`, say a potential wall, acting only on type 0 particles.  
You would achieve this by creating a ParticleGroup containing only the type 0 particles and passing it to ExternalForces as an argument:

.. code:: cpp
	  
  auto pd = make_shared<ParticleData>(N);
  //There are several selectors available and you can easily create new ones.
  auto pg = make_shared<ParticleGroup>(particle_selector::Type(0), pd, "Type 0");
  //Only particles in group pg will exist for extf
  auto wall =...
  auto extf = make_shared<ExternalForces<MyWall>>(pg, wall);
  ...
  integrator->addInteractor(extf);
  ...

   


