ParticleData
=============

One of the basic assumptions in UAMMD is that simulations are based on the state of "particles". ParticleData is the class in UAMMD that stores the properties of all the particles in the system.

This class takes care of allocating the necessary memory and providing accesses to any property in either the GPU or CPU.

Most UAMMD modules will need to be provided with a reference to a ParticleData instance.  

ParticleData handles and stores all properties a particle can have (however, they are only initialized when they are asked for the first time). It offers several ways to access these properties. See USAGE.  

ParticleData can change the size of the properties, shuffle the particles indices or change the container of a particular property at any given time. The relevant changes are announced using signals, so code using ParticleData should handle these events as needed, see Connecting to a signal.  

*****

Creation
----------

.. cpp:class:: ParticleData

   .. cpp:function:: ParticleData::ParticleData(int numberParticles, std::shared_ptr<System> sys = nullptr)
	       
		     ParticleData only needs a number of particles at creation. Optionally, a :ref:`System` instance can also be provided as a second argument. If not provided, ParticleData will handle System initialization.

   .. cpp:function:: std::shared_ptr<System> ParticleData::getSystem();
		  
   Returns the instance of :ref:`System` used by this ParticleData instance.



.. note:: Typically, UAMMD modules will require a shared_ptr to an instance of ParticleData:


**Example: creating a ParticleData instance**
   
.. code:: c++
	  
  auto pd = make_shared<ParticleData>(numberParticles);


Getting a handle get a certain property
------------------------------------------

You can access a property via ParticleData in GPU or CPU memory. You must specify the kind of access (read, write, readwrite). 

.. cpp:function:: property_ptr<type> ParticleData::getProperty(access::location loc, access::mode mode);

		  Returns a :cpp:type:`property_ptr` with the type of the property called Property. Note that a different function will be generated for each available property. Do not call :cpp:`getProperty`, rather :cpp:`getPos`, :cpp:`getForce`, etc. A list of properties is available below and in :code:`ParticleData/ParticleData.cuh`.
		  
		  :param loc: Access location specifier.
		  :param mode: Access read/write mode specifier.
		  :return: Reference to the property.

			   
.. cpp:enum:: access::location

	      .. cpp:enumerator:: access::location::gpu
				  
	      .. cpp:enumerator:: access::location::cpu
	      

				  
.. cpp:enum:: access::mode

	      .. cpp:enumerator:: access::mode::read
				  
	      .. cpp:enumerator:: access::mode::write
	      
	      .. cpp:enumerator:: access::mode::readwrite
				  

.. note:: The enumerators :cpp:enum:`access::location` and  :cpp:enum:`access::mode` can be used without the second scope. In other words, you can write :cpp:any:`access::gpu` instead of :cpp:any:`access::location::gpu`.
	  
The type returned by :cpp:any:`ParticleData::getProperty` is a lightweight standard-library-like pseudo-container defined as


.. cpp:class:: template<class T> property_ptr<T>

	   A pseudo-container that signals ParticleData when it is destroyed.

	   .. cpp:function:: T* property_ptr::begin()

			     An iterator to the first element of the property data.
	       
	   .. cpp:function:: T* property_ptr::raw()

			     A raw pointer to the first element of the property data.

	   .. cpp:function:: T* property_ptr::end()

			     An iterator to the last element of the property data (simply :cpp:expr:`T*`)
			     
	   .. cpp:function:: int property_ptr::size()

			     The size of the container, i.e the number of particles.

	   .. cpp:function:: access::location property_ptr::location()

			     The location of the data in the property_ptr, such as :cpp:any:`access::gpu` or :cpp:any:`access::cpu` 

Example
~~~~~~~~~

.. code:: cpp
	  
  auto radius = pd->getRadius(access::gpu, access::write);
  thrust::fill(thrust::cuda::par, radius.begin(), radius.end(), 1.0); 
  auto force = pd->getForce(access::cpu, access::write);
  std::fill(force.begin(), force.end(), real4());
  auto id = pd->getId(access::cpu, access::read); //It is not legal to write to ID, one can only read from it.
  int* raw_id_property_pointer = id.raw();

If the mode is set to write, the handle will gain exclusivity and no one else will be able to access it until it is released (the handle is deleted).

.. note:: There is no real difference between :cpp:any:`access::write` and :cpp:any:`access::readwrite` (at the moment) beyond informing the reader of the intention of modifying the contents (readwrite) vs ignoring the current contents and overwriting (write).	  
UAMMD cannot write to a property that is currently being read and cannot read from a property that is currently being written to.   
For this **it is important to control the scope of the property handles**.  
Handles are compatible with std and thrust algorithms and can be considered c++ iterators for all porpoises.  

.. _list-of-available-properties:

List of available properties
-----------------------------

The beginning of ParticleData.cuh contains a list of available per particle properties (such as positons, velocities, forces...).  
You can see a list of all the available ones and add more properties by appending to the macro ALL_PROPERTIES_LIST.  
A family of access functions will be autogenerated for each property inside this macro (such as get[Name] (), [Name]WrittenSignal(), ...).   

For instance, ParticleData holds the positions of the particles in :cpp:type:`real4` variables in an array named "pos". Thus, the function :cpp:`property_ptr<real4> ParticleData::getPos()` is available.


Basic properties include (type name):
  * :cpp:`real4 pos`
  * :cpp:`real3 vel`
  * :cpp:`real4 force`
  * :cpp:`real energy`
  * :cpp:`real virial`
  * :cpp:`real mass`
  * :cpp:`real charge`
  * And more defined in ParticleData.cuh


.. _particle_id_assignation:

Particle id assignation
------------------------

When added each particle is assigned an unique id or name (which corresponds to its position in the underlying container just after ParticleData creation). At this moment one can access the position of particle with id=i at pos[i].  
A reordering or some other internal processes may alter this fact, making the index of the particle with id=i not correspond to i anymore.   
While the location of each particle in the internal containers might change, the particles ids (names) will never change.  
The current ids are available through the property "Id" in ParticleData:

.. code:: cpp
	  
  auto index2id = pd->getId(access::cpu, access::read);
  int someIndex=0;
  int nameOfParticleAtSomeIndex = index2id[someIndex];
  
The opposite indirection is also accessible through ParticleData. That is finding the current index of a certain particle through its id (name):

.. cpp:function:: const int* ParticleData::getIdOrderedIndices(access::location loc);

	Returns an array with memory residing at the given location with the current indices of the particles given their id.


.. code:: cpp
	  
  auto id2index = pd->getIdOrderedIndices(access::cpu);
  int someId=0;
  int indexOfParticleWithSomeId = id2index[someId];

Thrust offers a permutation_iterator that can be used to mask this behavior to access a certain property by either id or index:

.. code:: cpp
	  
  auto positionWithArbitraryOrder = pd->getPos(access::cpu, access::read);
  //Accessing particles when order is not important 
  int someIndex = 0;
  real4 positionOfParticleAtSomeIndex = pos[someIndex];
  auto index2id = pd->getId(access::cpu, access::read);
  int idOfParticleAtSomeIndex = index2id[someIndex];
  //Accessing particles so index = name
  int someId = 0;
  auto id2Index = pd->getIdOrderedIndices(access::cpu);
  //Using a simple indirection
  real4 positionOfParticleWithSomeId = positionWithArbitraryOrder[id2index[someId]];
  //Using a permutation iterator
  auto positionOrderedById = thrust::make_permutation_iterator(positionWithArbitraryOrder, id2index);
  real4 positionOfParticleWithSomeId = positionOrderedById[someId];

    
Advanced usage
---------------

ParticleData allocates properties the first time they are requested. Sometimes one would like to know if a certain property has been previously requested to decide upon a fall back behavior.
For example, one would like to use the mass of each particle for a certain algorithm. If mass has not been set for each particle independently one would like to assume that all particles have the same mass, equal to some default value. There are two ways to obtain this information:


.. cpp:function:: property_ptr<type> ParticleData::getPropertyIfAllocated(access::location loc, access::mode mode);

		  Returns a :cpp:type:`property_ptr` with the type of the property called Property. If the property has not been requested before via :cpp:`ParticleData::getProperty` the returned pointer is null.
		  
		  :param loc: Access location specifier.
		  :param mode: Access read/write mode specifier.
		  :return: Reference to the property.

.. code:: cpp
	  
  bool isMassAllocated = pd->isMassAllocated();
  auto mass = pd->getMassIfAllocated(access::gpu, access::read);
  //mass.raw() and mass.begin() will be nullptr if mass has not been asked for before (either in GPU or CPU). 
  //Note that this call will never allocate the property


  
Triggering a sorting
~~~~~~~~~~~~~~~~~~~~~~


.. cpp:function:: void ParticleData::sortParticles();

   ParticleData can sort the particles to increase spatial locality of the data, which might be beneficial for some algorithms.
   Use this function to force a sorting.

ParticleData uses its own internal heuristic to spatially order the particles. This heuristic can be influenced by providing some information to ParticleData about the simulation domain and the typical range of the interactions in the simulation.

.. cpp:function:: void ParticleData::hintSortByHash(Box box, real3 typicalDistance)

	Informs :cpp:class:`ParticleData` of the simulation domain and the typical distance of the interactions in each direction. ParticleData will use this information to improve the effectiveness of the spatial hashing.

	
	
Signals
~~~~~~~~~~~

ParticleData broadcasts a signal every time some internal processes undergo, such as a particle reordering or a resize.  
One can subscribe to these signals like this:

.. code:: cpp
	  
  class User{
    connection reorderConnection, numParticlesChangedConnection;
    public:
     User(std::shared_ptr<ParticleData> pd){
       reorderConnection = pd->getReorderSignal()-> connect([this](){this->handle_reorder();});
       numParticlesChangedConnection = pd->getNumParticlesChangedSignal()->connect([this](int Nnew){this->handle_numChanged(Nnew);});
     }
     ~User(){
       reorderConnection.disconnect();
       numParticlesChangedConnection.disconnect();
     }
     void handle_reorder(){
       std::cout<<"A reorder occured!!"<std::endl;
     }  
     void handle_numChanged(int Nnew){
       std::cout<<"Particle number changed, now it is: "<<Nnew<<std::endl;
     }
  };

Note that it is possible that a module does not need to track the specific order of the particles or do anything special when the number of them changes. See for example NbodyForces or PairForces. Actually, most of the time you will get away without needing to connect to the signals.  

UAMMD uses the :cpp:type:`signal`/:cpp:type:`connection` classes from `fr00b0/nod <https://github.com/fr00b0/nod>`_.

.. cpp:type:: template<class Function> signal<Function> = nod::unsafe_signal<Function>

	       UAMMD's signal class. Must be specialized with a function signature, for instance :cpp:`using non_broadcasting_signal = signal<void()>`.


.. cpp:type:: connection = nod::connection

	      Keeps track of an open signal connection. Its main use is to be able to safely detach from a signal via :cpp:`connection::disconnect()`
	       

		   

List of available signals
%%%%%%%%%%%%%%%%%%%%%%%%%



.. cpp:function:: std::shared_ptr<signal<void(int)>> ParticleData::getNumParticlesChangedSignal();

        Returns a handle to the signal emitted when the number of particles changes.
	This signal is triggered when the total number of particles changes
	Broadcasts an :code:`int` with the new number of particles.

		 
.. cpp:function:: std::shared_ptr<signal<void()>> ParticleData::getReorderSignal();
		  
        Returns a handle to the signal emitted when global particle sorting occurs.
	This signal is triggered when the global sorting of particles changes.
	Does not broadcasts any value.

      
.. cpp:function:: std::shared_ptr<signal<void()>> getPropertyWrittenSignal();

		  Triggered when property named Property has been requested with the write or readwrite flag. Notice that the signal is emitted at requesting of the property, so the requester has writing rights. These are auto generated for all properties (pos, vel, mass...) . One should use this callback merely for setting a flag for later work.
		  Does not broadcast any value.
		  Note that a different function is defined for each property. So do not call :code:`getPropertyWrittenSignal`, rather :code:`getPosWrittenSignal`, :code:`getMassWrittenSignal`, etc. 
		 
      

