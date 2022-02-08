ParticleGroup
=============


Most UAMMD modules can be instructed to act only on a subset of particles, this is handled by creating a ParticleGroup.  
A group can contain all particles, no particles or anything in between.

.. cpp:class:: ParticleGroup

   Keeps track of a subset of particles which properties reside in :cpp:any:`ParticleData`.
            
   .. cpp:function:: std::shared_ptr<ParticleData> ParticleGroup::getParticleData()

      Get the instance of :cpp:class:`ParticleData` the :cpp:class:`ParticleGroup` works on.

   .. cpp:function:: void ParticleGroup::clear();

		        Remove all particles from the group.
			
   
   .. cpp:function:: void ParticleGroup::addParticlesById(access::location loc, const int *ids, int N);

		     Add particles to the group via an array (with memory residing according to :cpp:`loc`) of :ref:`particle ids <particle_id_assignation>`.

   .. cpp:function:: void ParticleGroup::addParticlesByCurrentIndex(access::location loc, const int *indices, int N);
	     		     
		     Add particles to the group via an array with the current indices of the particles in :ref:`ParticleData` (this is faster than :cpp:any:`addParticlesById`).
		     
   .. cpp:function:: const int * ParticleGroup::getIndicesRawPtr(access::location loc);

		     Get a raw memory pointer to the index list if it exists (:ref:`ParticleGroup` not always creates an actual list of particles).
		     
    
   .. cpp:function:: IndexIterator ParticleGroup::getIndexIterator(access::location loc);

		     Get an iterator with the indices of particles in this group


   .. cpp:function:: template<class Iterator> accessIterator<Iterator> ParticleGroup::getPropertyIterator(Iterator property, access::location loc);

      Returns an iterator that will have size pg->getNumberParticles() and will iterate over the particles in the group.For example, If a group contains only the particle with :cpp:`id=10`, passing :cpp:`pd->getPos(...).begin()` to this function will return an iterator so that :cpp:`iterator[0] = pos[10];` and it will take into account any possible reordering of the pos array. The location does not have to be specified if the property argument is a :cpp:class:`property_ptr` provided by :cpp:class:`ParticleData`.

   .. cpp:function:: int ParticleGroup::getNumberParticles();

      Returns the number of particles currently in the group.
		     
		     
   .. cpp:function:: std::string ParticleGroup::getName();

      Returns the given name of the group.


.. hint:: :ref:`ParticleGroup` does not always create an actual list of particles. The iterator returned by :cpp:any:`getPropertyIterator` takes advantage of this
      
Creation
---------

:cpp:any:`ParticleGroup` exposes several constructors:

.. cpp:function:: ParticleGroup::ParticleGroup(std::shared_ptr<ParticleData> pd, std::string name = std::string("noName"));

   This constructor creates a :cpp:class:`ParticleGroup` containing all particles in the provided :cpp:class:`ParticleData` instance.

.. cpp:function:: ParticleGroup::ParticleGroup(ParticleSelector selector, std::shared_ptr<ParticleData> pd, std::string name = std::string("noName"));

   Fills the group with the particles according to the provided :cpp:any:`ParticleSelector`.

.. cpp:function:: template<class InputIterator> ParticleGroup::ParticleGroup(InputIterator begin, InputIterator end, std::shared_ptr<ParticleData> pd, std::string name = std::string("noName"));

  Fills the group with the :ref:`particles ids <particle_id_assignation>` provided in the iterator range begin:end.

  
Example
**********

.. code:: c++
	  
  //By default a ParticleGroup will contain all particles
  auto allParticlesGroup = make_shared<ParticleGroup>(pd, sys, "AGroupWithAllParticles");

  //Different selectors offer different criteria
  //In this case, it will result in a group with particles whose ID lies between 4 and 8
  particle_selector::IDRange selector(4,8);
  auto aGroupWithSomeIDs = make_shared<ParticleGroup>(selector, pd, sys, "SomeName");

  //Equivalently a list of particle IDs can be provided directly
  auto idrange = std::vector<int>(4); std::iota(idrange.begin(), idrange.end(), 4);
  auto anEquivalentGroup = make_shared<ParticleGroup>(idrange.begin(), idrange.end(), pd, sys, "SomeOtherName");

  //A group containing all particles of a certain type (or types) (type being the value of pos.w)
  auto groupOfParticlesWithType0 = make_shared<ParticleGroup>(particle_selector::Type(0), pd, sys, "Type 0 particles");
  auto groupOfParticlesWithType0And3 = make_shared<ParticleGroup>(particle_selector::Type({0,3}), pd, sys, "Type 0 and type 3 particles");

  //A group of 10 random particles
  std::vector<int> randomlyOrderedIds(numberParticles); 
  std::iota(randomlyOrderedIds.begin(), randomlyOrderedIds.end(), 0);
  std::shuffle(randomlyOrderedIds.begin(), randomlyOrderedIds.end(), std::mt19937{std::random_device{}()});
  randomlyOrderedIds.resize(10);
  auto groupOf10RandomParticles = make_shared<ParticleGroup>(randomlyOrderedIds.begin(), randomlyOrderedIds.end(), pd, sys, "10 Random Particles");


Instructions on how to create a selector are located in ParticleGroup.cuh but the easiest way to create a group with a custom criteria is to just pass a list of particle ids as in the examples.

Particle selectors
--------------------

Selectors are small :ref:`functors <Functor>` providing a member that checks if a given particle should be in a group or not.

.. cpp:class:: ParticleSelector
	       
   This is a concept, not a virtual class that must be inherited. Any class defining a member with the signature below will act as  a valid selector for :cpp:class:`ParticleGroup`
   
   .. cpp:function:: bool isSelected(int particleIndex, std::shared_ptr<ParticleData> pd);

      This function should use the provided :cpp:class:`ParticleData` instance to decide if the particle with index :cpp:any:`particleIndex` should be included in the group or not.

.. important:: Selectors are only used for particle inclusion into a group when the group is created. :ref:`ParticleGroup` will not track the changes in the inclusion conditions.

	       
Example
********

A selector that returns true for every particle.

.. code:: c++

   class All{
    public:
      All(){}
      bool isSelected(int particleIndex, std::shared_ptr<ParticleData> pd){
	return true;
      }
    };

Available particle selector
*****************************

Creating a group by providing the ids of the relevant particles can be in many cases the most acceptable way of creating a group. However, several selectors are available for convenience under the :cpp:`particle_selector` namespace.




.. cpp:class:: particle_selector::All;

	       Selects all the particles.
	       

.. cpp:class:: particle_selector::None;

	       Results in an empty group.

.. cpp:class:: particle_selector::IDRange
	       
	       Select particles with ID in a certain range
	       
   .. cpp:function:: IDRange::IDRange(int first, int last);



.. cpp:class:: particle_selector::Domain

	       Select particles inside a certain rectangular region of the simulation box.

   .. cpp:function:: Domain::Domain(real3 origin, Box domain, Box simulationBox);

      This selector will first fold the particles into :cpp:`simulationBox` and then choose any particle that lies inside a region given by :cpp:`domain` with origin :cpp:`origin`.
      
.. cpp:class:: particle_selector::Type

	       Select particles by type (using the fourth element of the positions, pos.w)
	       
   .. cpp:function:: Type::Type(std::vector<int> typesToSelect)

      A list of types that should go into the group.
	       


Usage with UAMMD modules
-----------------------------

When it makes sense, UAMMD modules will have an optional ParticleGroup argument at creation. See for example :ref:`PairForces`.


General usage
---------------
:cpp:any:`ParticleGroup` will keep track of its particles and will always provide their up to date global indices.
     
.. code:: c++
	  
  //You can request an iterator with the current indices of the particles in a group with:
  auto indicesOfParticlesInGroup = pg->getIndexIterator(access::location::gpu);

  //Or get a plain array with the indices directly, if it exists.
  auto rawMemoryPtrOfIndices = pg->getIndicesRawPtr(access::location::gpu); //or cpu, it will be nullptr if all (or none) particles are in the group

  //You can also request an iterator that will read a ParticleData array using the group indices directly.
  //This allows to write generic code that will work both with a group or with a ParticleData array.
  auto allPositions = pd->getPos(access::location::gpu, access::mode::read);
  auto IteratorWithPositionsInGroup = pg->getPropertyIterator(allPositions);
  ...
  //In device code
  real4 positionOfFirstParticleInGroup = IteratorWithPositionsInGroup[0];

.. hint:: As a general rule, when writing UAMMD code, it is wise to access particle properties using :cpp:class:`ParticleGroups` instead of :ref:`ParticleData` directly.

.. note:: A default group contains all particles, it is a special case and incurs no overhead (besides maybe a couple of registers) when created or used.  



