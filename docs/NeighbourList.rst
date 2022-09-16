
.. _NeighbourList:

Neighbour Lists
================


.. figure:: img/verletlist.*
	    :width: 75%
	    :align: center

	    A depiction of a neighbour list in a section of a particle distribution (left). Particles inside the blue circle (of radius :math:`r_c` ) are neighbours of the red particle. The Verlet list strategy (see below) defines a second safety radius, :math:`r_s` , that can be leveraged to reuse the list even after particles have moved. In the worst-case scenario of the red particle and another particle just outside rs approaching each other (right), the list will be invalidated only when each has moved :math:`r_t = \frac{1}{2}(r_s - r_c )` since the last rebuild.


All neighbour lists in UAMMD are provided under a common interface:
	    
.. cpp:class:: NeighbourList

	       A conceptual interface class that all UAMMD neighbour lists share.
	       

   .. cpp:function:: NeighbourList(std::shared_ptr<ParticleData> pd);

      Initializes the neighbour list acting on all particles.

   .. cpp:function:: NeighbourList(std::shared_ptr<ParticleGroup> pg);

      Initializes the neighbour list acting on a group of particles.

   .. cpp:function:: void update(Box box, real cutOff, cudaStream_t st = 0);

      Uses the latest positions of the particles in the :ref:`ParticleData` instance to generate a list of neighbours.

   .. cpp:function:: template<class Transverser>\
		     void transverseList(Transverser &tr, cudaStream_t st = 0);

      Applies a given :cpp:any:`Transverser` to the current state of the neighbour list.

   .. cpp:function:: auto getNameList();

      Each NeighbourList provides a function (where "Name" is the name of the particular list) that returns the internal data structures of the list (see :ref:`available neighbour lists`). Arrays in the structure returned by this function will be allocated in the GPU.

   .. cpp:function:: auto getNeighbourContainer();

      Returns a :cpp:any:`NeighbourContainer` that allows accessing the neighbours of each particle from the GPU.


.. note:: Given that both :cpp:any:`transverseList` must be templated and :cpp:any:`getNeighbourContainer` returns a different type depending on the particular list, the :cpp:any:`NeighbourList` concept cannot be a virtual base class. All neighbour lists in UAMMD provide the functions above, but not using a language-enforced mechanism.


Example
---------

Constructing and using a neighbour list.

Once constructed, there are three ways to use a neighbour list:
 * Providing a :ref:`Transverser` (via :cpp:any:`NeighbourList::transverseList`).
 * Obtaining a :cpp:any:`NeighbourContainer` (via :cpp:any:`NeighbourList::getNeighbourContainer`).
 * Using the internal data structures of the particular neighbour list (see for instance :cpp:any:`CellList::getCellList`).

.. code:: cpp

  #include"uammd.cuh"
  #include"Interactor/NeighbourList/CellList.cuh"
  using namespace uammd;
  
  int main(){
    //Alias to show that any neighbour list could be used.
    using NeighbourList = CellList;
    int N = 1000;
    real3 boxSize = make_real3(128,128, 128);
    real rcut = 2.5;
    auto pd = std::shared_ptr<ParticleData>(N);
    //.. initialize particle positions here...
    nl = make_shared<NeighbourList>(pd);
    Box box(boxSize);
    nl->update(box, rcut);
    
    //auto ni = nl->getNeighbourContainer();    
    //nl.transverseList(some_transverser);
    return 0;
  }
  
.. hint:: See :cpp:any:`NeighbourContainer` below.
.. hint:: Here, :cpp:`some_transverser` is an instance of a :ref:`Transverser`.

	  

Available neighbour lists
--------------------------


CellList
~~~~~~~~

.. figure:: img/celllist_sketch.*
	    :width: 50%
	    :align: center
		    
	    Sketch of the cell list algorithm. Space is binned (black grid) and the bin (cell) of each particle is computed. In order to look for the neighbours of the black particle (those inside the green dashed circle) all the particles inside the adjacent cells (bins inside the orange dashed square, 27 cells in three dimensions) are checked. The orange particles are therefore false positives. Finally, the yellow particles are never considered when looking for neighbours of the black one.

	    
The main idea behind the cell list is to perform a spatial binning and assign a hash to each particle according to the bin it is in. If we then sort these hashes we get a list in which all the particles in a given cell are contiguous. By accessing, for a certain particle, the particles in the 27 surrounding cells we can find its neighbours without checking too many false positives.

The algorithm for the cell list construction can be summarized in three separate steps:
 * Hash (label) the particles according to the cell (bin) they lie in.
 * Sort the particles and hashes using the hashes as the ordering label (technically this is known as sorting by key). So that particles with positions lying in the same cell become contiguous in memory.
 * Identify where each cell starts and ends in the sorted particle positions array.

After these steps we end up with enough information to visit the 27 neighbour cells of a given particle.
We have to compute the assigned cell of a given position at several points during the algorithm. Doing this is straightforward. For a position inside the domain, :math:`x \in [0, L)`, the bin assigned to it is :math:`i = \textrm{floor}(x/n_x) \in [0, n_x- 1]`. It is important to notice that a particle located at exactly :math:`x = L` will be assigned the cell with index :math:`n_x`, special consideration must be taken into account to avoid this situation. In particular, in a periodic domain, a particle at :math:`x=L` should be assigned to the cell :math:`i=0`.


The cell list stores a copy of the particle positions sorted in such a way that the indices of the particles that are located in the same cell lie contiguous in memory:
 * SortPos: [all particles in cell 0, all particles in cell 1,..., all particles in cell ncells]

Two other arrays are provided providing, for each cell, the index of the first and last particles in the cell in :cpp:`sortPos`.

.. cpp:class:: CellList

	       Besides the functions defined in :cpp:any:`NeighbourList`, the cell list also exposes some functions proper to this particular algorithm. 

   .. cpp:function:: CellListBase::CellListData getCellList();

      Returns the internal structures of the CellList.


      
.. cpp:struct:: CellListBase::CellListData;
		
   .. cpp:member:: const uint * cellStart;

      Encodes the index (in :cpp:any:`sortPos`) of the first particle in a given cell. In particular, particles with indices (in :cpp:any:`sortPos`) :math:`j\in [\text{cellStart}[i]-\text{VALID\_CELL}, \text{cellEnd}[i])` are located in cell i and have positions given by :cpp:expr:`sortPos[j]` (j is an internal indexing of the cell list, to get the group index of the particle you can use :cpp:expr:`groupIndex[j]`).
      
   .. cpp:member:: const int  * cellEnd;

      :cpp:`cellEnd[i]` stores the last particle in :cpp:any:`sortPos` that lies in cell :cpp:`i`.
      
   .. cpp:member:: const real4 *sortPos;

      Particle positions sorted so that particles in the same cell are contiguous in this array.
		   
   .. cpp:member:: const int* groupIndex; 

      The group index (see :ref:`ParticleGroup`) of the particle with position :cpp:expr:`sortPos[i]` is :cpp:expr:`groupIndex[i]`.
		   
   .. cpp:member:: Grid grid;

      The grid information of the cell list (stores things like number of cells).
		   
   .. cpp:member:: uint VALID_CELL;

      A value in cellStart less than VALID_CELL means the cell is empty. Subject to change between updates.


.. cpp:class:: CellListBase

   This class exposes the same functions as :cpp:any:`CellList`, but it does not depend on :ref:`ParticleData` (to facilitate its usage outside the ecosystem). In particular, this class offers an alternative constructor and update function.

   
   .. cpp:function:: CellListBase();

      Default constructor.
      
      
   .. cpp:member:: template<class PositionIterator>  void update(PositionIterator pos, int numberParticles, Grid grid, cudaStream_t st = 0);

      This function works similar to :cpp:any:`NeighbourList::update`, but constructs the list based on the contents of the provided :cpp:`pos` iterator and grid.

      
Example
...........

The different ways of using a :ref:`CellList` in UAMMD.

.. code:: cpp

  #include<uammd.cuh>
  #include<Interactor/NeighbourList/CellList.cuh>
  using namespace uammd;
  
  //Construct a list using the UAMMD ecosystem
  void constructListWithUAMMD(UAMMD sim){
    //Create the list object
    //It is wise to create once and store it
    CellList cl(sim.pd);
    //Update the list using the current positions in sim.pd
    cl.update(sim.par.box, sim.par.rcut);
    //Now the list can be used via the
    //  various common interfaces
    //-Providing a Transverser:
    //cl.transverseList(some_transverser);
    //-Requesting a NeighbourContainer
    auto nc = cl.getNeighbourContainer();
    //Or by getting the internal structure of the Cell List
    auto cldata = cl.getCellList();
  }
  //Construct a CellList without UAMMD
  template<class Iterator>
  void constructListWithPositions(Iterator positions, 
                                  int numberParticles,
                                  real3 boxSize, 
                                  int3 numberCells){
    //Create the list object
    //It is wise to create once and store it
    CellListBase cl;
    //CellListBase requires specific cell 
    // dimensions for its construction
    Grid grid(Box(boxSize), numberCells);
    //Update the list using the positions
    cl.update(positions, numberParticles, grid);
    //Now the internal structure of the Cell List
    // can be requested
    auto cldata = cl.getCellList();
    //And a NeighbourContainer can be constructed from it
    auto nc = CellList_ns::NeighbourContainer(cldata);
  }	  

.. note:: Here the :cpp:`UAMMD` struct contains an instance of :ref:`ParticleData` and a series of parameters related to this particular example.
  
.. hint:: This example makes use of the :ref:`Transverser` and :ref:`NeighbourContainer<NeighbourContainer>` interfaces.
  
VerletList
~~~~~~~~~~~~

This list uses :ref:`CellList` to construct a neighbour list up to a distance :math:`r_{s} > r_{cut}`, in this case the list only has to be reconstructed when any given particle has travelled more than a threshold distance,

.. math::
   
   r_t = \frac{r_{s}-r_{c}}{2}.

See the representation at the start of :ref:`NeighbourList`.

.. hint:: A good default is usually around :math:`r_s\approx 1.15r_c`. 

The list is constructed by storing a private list of neighbours for each particle in a column-major fashion. In order to achieve a cache-friendly memory pattern this "private" lists are stored in the same contiguous array. Since we do not know in advance how many neighbours each particle has we set up a maximum number of neighbours per particle, :math:`N_{\text{max}}`, and allocate an array of size :math:`N_{\text{max}}N` elements. Later on we traverse the list by assigning a thread per particle, which prompts for a column-major layout of the list. That is, threads will tend to read contiguous memory locations if we place the first neighbours of all particles contiguously, then the second, and so forth and so on.

.. note::
   
   A row-major layout (in which we place all neighbours of a certain particle contiguously) will be beneficial if we assign a block of threads per particle when traversing.
   Measuring is required to know which strategy is best in each case (thread-per-particle vs block-per-particle). UAMMD chooses a column-major format, as testing suggests this is the better choice in our habitual use-cases.\footnote{
   Nonetheless this fact is abstracted away in the interface and changing between column- and row-major formats can be done easily and without affecting the users code.

Finally, the maximum number of neighbours per particles, which affects both performance and memory consumption, is autotuned at each update to be the nearest multiple of 32 (the CUDA warp size) of the particle with the greatest number of neighbours.
	  
As with the :cpp:any:`CellList`, UAMMD exposes the Verlet list algorithm as part of the ecosystem and as an external accelerator. If the internal option is used the safety factor is automatically autotuned. Regardless, the safety radius can be modified via the :cpp:any:`VerletList::setCutOffMultiplier` member function.

In both cases, accessing the :cpp:any:`VerletList` via the common UAMMD interfaces (:ref:`Transverser` and :ref:`NeighbourContainer<NeighbourContainer>`) makes it interchangeable with a :cpp:any:`CellList`, as evidenced in the example code below.

.. cpp:class:: VerletList

   This class adheres to the :cpp:any:`NeighbourList` UAMMD interface. Besides the functions defined in :cpp:any:`NeighbourList`, the cell list also exposes some functions proper to this particular algorithm.

   .. cpp:member:: VerletListBase::VerletListData getVerletListData();

		   Returns the internal structures of the Verlet list

   .. cpp:member:: void setCutOffMultiplier(real newMultiplier);
		   
		   Sets the cutoff multiplier, i.e :math:`r_s/r_c=`  :cpp:`newMultiplier`.

   .. cpp:member:: int getNumberOfStepsSinceLastUpdate();

		   Returns the number of times the update has been called since the last time a list rebuild was required.

.. cpp:struct:: VerletListBase::VerletListData;

   .. cpp:member:: real4* sortPos;

      Positions sorted to be contiguous if they fall into the same cell. Same as with :ref:`CellList`.
		
   .. cpp:member:: int* groupIndex;

      Given the index of a particle in :cpp:`sortPos`, this array returns the index of that particle in :ref:`ParticleData` (or the original positions array if the list is used outside the ecosystem). This indirection is necessary when something other than the positions is needed (like the forces). Same as with :ref:`CellList`.
      
   .. cpp:member:: int* numberNeighbours;

      The number of neighbours for each particle in sortPos.
      
   .. cpp:member:: StrideIterator particleStride;

      For a given particle index, :cpp:`i`, :cpp:`particleStride[i]` holds the index of its first neighbour in :cpp:`neighbourList`. :cpp:`StrideIterator` is a random access iterator. 

      .. note:: Note that, as evidenced by its type name, the particle stride (or offset) is not necessarily a raw array but rather a generic C++ random access iterator (that behaves as a raw array for most purposes). For instance the offset might just be the same for all particles. In UAMMD's current implementation, the particle stride is simply the maximum number of neighbours per particle. However, this interface allows for the possibility of compacting the list in the future, with a similar behavior as the :cpp:`cellStart` array in :ref:`CellList`.
		
      .. hint:: In particular, in the current implementation this iterator simply returns, for a given index :cpp:`i`, :cpp:`maxNeighboursPerParticle*i`.

	     
   .. cpp:member:: int* neighbourList;
		   
      The actual neighbour list. The neighbour :cpp:`j` for particle with index :cpp:`i` (of a maximum of :cpp:`j=numberNeighbours[i]`) is located at :cpp:`neighbourList[particleStride[i] + j];`.



.. cpp:class:: VerletListBase

   This class exposes the same functions as :cpp:any:`VerletList`, but it does not depend on :ref:`ParticleData` (to facilitate its usage outside the ecosystem). In particular, this class offers an alternative constructor and update function.

   
   .. cpp:function:: VerletListBase();

      Default constructor.
      
      
   .. cpp:member:: template<class PositionIterator>  void update(PositionIterator pos, int numberParticles, Box box, real cutOff, cudaStream_t st = 0);

      This function works similar to :cpp:any:`NeighbourList::update`, but constructs the list based on the contents of the provided :cpp:`pos` iterator.
      
Example
.........
      
The different ways of using a :ref:`VerletList`.

.. code:: cpp
	  
  #include<uammd.cuh>
  #include<Interactor/NeighbourList/VerletList.cuh>
  using namespace uammd;
  
  //Construct a list using the UAMMD ecosystem
  void constructListWithUAMMD(UAMMD sim){
    //Create the list object
    //It is wise to create once and store it
    VerletList vl(sim.pd);
    //Update the list using the current positions in sim.pd
    vl.update(sim.par.box, sim.par.rcut);
    //Now the list can be used via the
    //  various common interfaces
    //-With a Transverser:
    //vl.transverseList(some_transverser);
    //-Requesting a NeighbourContainer
    auto nc = vl.getNeighbourContainer();
    //Or by getting the internal structure of the Verlet List
    auto vldata = vl.getVerletList();
    //The safety radius can be specified
    vl.setCutOffMultiplier(sim.par.rsafe/sim.par.rcut);
    //The number of update calls since the last time
    // it was necessary to rebuild the list can be 
    // obtained
    int nbuild = vl.getNumberOfStepsSinceLastUpdate();
  }
  //Construct a VerletList without UAMMD
  template<class Iterator>
  void constructListWithPositions(Iterator positions, 
                                  int numberPartivles,
                                  real3 boxSize, 
                                  int3 numberParticles){
    //Create the list object
    //It is wise to create once and store it
    VerletListBase vl;
    //The safety radius can be specified
    vl.setCutOffMultiplier(sim.par.rsafe/sim.par.rcut);
    //Update the list using the positions
    vl.update(positions, numberParticles, 
              sim.par.box, sim.par.rcut);
    //Now the internal structure of the Verlet List
    // can be requested
    auto vldata = vl.getVerletList();
    //And a NeighbourContainer can be constructed from it
    auto nc = VerletList_ns::NeighbourContainer(vldata);
    //The number of update calls since the last time
    // it was necessary to rebuild the list can be 
    // obtained
    int nbuild = vl.getNumberOfStepsSinceLastUpdate();
  }


.. note:: Here the :cpp:`UAMMD` struct contains an instance of :ref:`ParticleData` and a series of parameters related to this particular example.

.. hint:: The :ref:`Transverser` and :ref:`NeighbourContainer<NeighbourContainer>` options are identical to the case of a :ref:`CellList`.

  
Linear Bounding Volume Hierarchy list (LBVH)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The Linear Bounding Volume Hierarchy (LBVH) neighbour list works by partitioning space into boxes according to a tree hierarchy in such a way that interacting pairs of particles can be located quickly. The innermost level of partitioning encases single particles, and boxes sharing faces are hierarchically bubbled together to form a tree structure. In contrast, the cell list partitions space in identical cubes independently of the particles positions or their distribution inside the domain. The "object awareness" of the LBVH results in a better handling of systems with large density fluctuations or objects of highly different size.

.. note:: In contrast, the cell list is not aware of the size of the particles, only accepting a cut off distance. If a system contains particles of different sizes (and therefore interaction cut offs), the cell list must be constructed using the largest one. In the presence of large size disparities (say a set of large particles interacting with tiny ones), a lot of unnecessary particle pairs will be visited (in particular when checking the neighbours of a tiny particle).

	  
A full detailed description of the algorithm is beautifully laid out in detail in references [Howard2016]_, [Howard2019]_ and [Torres2009]_, with only minor modifications to them present in UAMMD's implementation.

A brief summary of the algorithm:
  * We start by assigning a different type to each particle based on its size (understanding that if particles have not been assigned a size, they will all have the same type).
  * Then, we sort the particles by assigning a hash to each one in a way such that two given particles of the same type that are close in physical space tend to be close in memory. We achieve this by sorting particles first by type and then by Z-order hash (actually, UAMMD uses the Morton hash presented at \ref{alg:mortonhash]_, encoding the type in the last two bits, which are typically unused).
  * The sorted particle hashes are included in a binary tree structure following Karra's algorithm [Karras2012]_. By including the type in the particle hashing, we can generate a single tree, ensuring that a different subtree is constructed for each type. The root of the subtree for each type can be then identified by descending the tree. This appears to scale well with the number of types when compared to generating an entirely new tree for every type as in [Howard2016]_, [Howard2019]_.
  * After, we Assign an Axis Aligned Bounding Box (AABB) to each node of the tree that joins the AABBs of the nodes below it (with the particles, aka leaf nodes, being at the innermost level). This is done using Karra's algorithm [Karras2012]_. The AABBs are stored in a ``quantized'' manner that allows to store a node in a single int4, improving traversal time [Howard2019]_. The bubbling of boxes stops at the root of every type subtree.
  * Finally, the neighbours of a given particle are found by traversing the AABB subtrees of every type [Torres2009]_.


Tree traversal is carried out in a top-down approach, where each particle starts by checking its distance to the root of a given subtree and subsequently descending as needed. If a particle's AABB overlaps a node within a given cut off, the algorithm goes to the next child node, otherwise it skips to the next node/tree.
For a given particle, overlap with the 27 (in 3D) periodic images of the current subtree is computed before traversal of a tree and encoded in a single integer to reduce divergence (except the main box, which is traversed first by default) (see [Howard2019]_).
After a type subtree is entirely processed, the process is repeated with next one until none remain.

In my personal experience, the sheer raw power of the cell list in the GPU makes this algorithm not worth the effort in general. Note, however, that this algorithm is bound to outperform the cell list in certain situations, mainly when the size disparities between the different particles in the simulation is pronounced (as in the largest particle having at least twice the size of the smallest) or when the configuration presents a very low density (in terms of the cut-off distance of the interaction).



Example
..........

The interface for this neighbour list in UAMMD is more restricted than those of the previously introduced ones. The reason for this being its, yet to be found, applicability in our simulations. Nonetheless, it can be used in any place where :ref:`CellList` or :ref:`VerletList` can be used.

The internal data structure of the LBVH list can be queried, but we have not discussed in detail the algorithm in this manuscript. A reader who is particularly interested in making use of the LBVH list or a more in-depth understanding of its inner workings is referred to the code itself, located at the source file :code:`Interactor/NeighboutList/LBVH.cuh`.

.. code:: cpp
  
  #include<uammd.cuh>
  #include<Interactor/NeighbourList/LBVHList.cuh>
  using namespace uammd;
  
  //Construct a list using the UAMMD ecosystem
  void constructListWithUAMMD(UAMMD sim){
    //Create the list object
    //It is wise to create once and store it
    LBVHList vl(sim.pd);
    //Update the list using the current positions in sim.pd
    vl.update(sim.par.box, sim.par.rcut);
    //Now the list can be used via the
    //  various common interfaces
    //-With a Transverser:
    //vl.transverseList(some_transverser);
    //-Requesting a NeighbourContainer
    auto nc = vl.getNeighbourContainer();
    //Or by getting the internal structure of the LBVH List
    auto vldata = vl.getLBVHList();
  }
  	  
.. _NeighbourContainer:

The Neighbour Container interface
---------------------------------

A pseudo-container that provides, for each particle, a list of its neighbours.

.. note:: In some instances a neighbour list algorithm does not need to build an actual list of neighbours. This is the case with the :cpp:any:`CellList`, where traversing the 27 neighbouring bins of a particle is enough to go through its neighbours  (and often more performant than constructing an individual list of neighbours for each particle).
	  Furthermore, :cpp:any:`NeighbourContainer` allow to abstract away things like the underlying memory layout of a neighbour list (for instance to unify a row-major and a column-major layouts).
	  
	  
.. cpp:class:: NeighbourContainer

	       
   .. cpp:function:: NeighbourContainer(ListData nl);

      The constructor of a NeighbourContainer takes as argument the internal data structures of a particular list. Normally the user does not call the constructor explicitly, rather obtain an instance via the :cpp:any:`getNeighbourContainer` functions provided by the different lists.

   .. cpp:function:: __device__ void set(int i);

      Set the active particle to "i", being "i" the index (in the internal indexing of the list). To get the group index of a particle you can use :cpp:any:`getGroupIndexes`.

      .. hint:: If the :ref:`id <particle_id_assignation>` of a particle is needed, the array of ids (via :cpp:any:`ParticleData::getId`) can be used once the group index is obtained. See :ref:`here <particle_id_assignation>`. In other words, the conversion between the internal index of a list and the id of a particle would go like internal index -> group index -> global index -> id (take into account that many times the group index and the global index will be equal).
      
   .. cpp:function:: __device__ NeighbourIterator begin();

      `Forward input iterator <https://en.cppreference.com/w/cpp/named_req/ForwardIterator>`_ pointing to the first neighbour of the particle selected by :cpp:any:`set`. Note that this iterator can only be advanced and dereferenced.

   .. cpp:function:: __device__ NeighbourIterator end();

      The distance between :cpp:any:`NeighbourContainer::begin` and the iterator provided by this function will be the number of neighbours. Note however that computing this distance can, in principle, be an :math:`O(N_{\text{neigh}})` operation.

   .. cpp:function:: __host__ __device__ const real4* getSortedPositions();

      Particle positions sorted in the internal ordering of the list. Lists usually spatially hash and sort the particles as part of the list constructing algorithm. This is a convenience function to take advantage of that already available, fast access, array.

   .. cpp:function:: __host__ __device__ const int* getGroupIndexes();

      Transforms an internal index in the list to the current index in the :ref:`ParticleGroup` (or :ref:`ParticleData` if the group contains all particles or when a group is not provided at all).

      .. note:: Even when the last two functions are decorated as "__device__" "__host__" the memory for the provided arrays lives in the GPU.


Example
~~~~~~~~~

Counting the number of neighbours of each particle using a :cpp:any:`NeighbourContainer`.

.. code:: cpp

  #include"uammd.cuh"
  #include"Interactor/NeighbourList/CellList.cuh"
  using namespace uammd;
  
  int main(){
    //Alias to show that any neighbour list could be used.
    using NeighbourList = CellList;
    int N = 1000;
    real3 boxSize = make_real3(128,128, 128);
    real rcut = 2.5;
    auto pd = std::shared_ptr<ParticleData>(N);
    //.. initialize particle positions here...
    CellList cl;
    Box box(boxSize);
    nl = make_shared<NeighbourList>(pd);
    nl->update(box, rcut);
    auto ni = nl->getNeighbourContainer();
    
    //Use the container in the GPU to count the neighbours per particle
    auto cit = thrust::make_counting_iterator<int>(0);
    thrust::for_each(cit, cit + numberParticles,
      [=] __device__ (int i){
           //Set ni to provide iterators for particle with index i (in the internal list order)
           ni.set(i);
	   //Position of a particle given an index in the internal list order
           const real3 pi = make_real3(ni.getSortedPositions()[i]);
           int numberNeighbours = 0;
           real rc2 = rcut*rcut; 
           for(auto neigh: ni){
             //int j = neigh.getGroupIndex();			
             const real3 pj = make_real3(neigh.getPos());
             const real3 rij = box.apply_pbc(pj-pi);
             const real r2 = dot(rij, rij);
             if(r2>0 and r2<rc2){
               numberNeighbours++;
             }
	 	}
            //Assuming the group used to construct the list contains all the particles in the system.
            const int global_index = ni.getGroupIndexes()[i];
            printf("The particle with index %d has %d particles closer than %g\n", global_index, numberNeighbours, rcut);
	 });
    return 0;
  }


.. hint:: The group index of a particle in the above example can be used to get its global index (see :ref:`ParticleGroup`) and then any of its properties via :ref:`ParticleData`. When no group is used (as in the example above), the default group (containing all particles) is assumed and the group index is equal to the global index.



.. rubric:: References:

.. [Karras2012] Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. Karras 2012.

.. [Torres2009] Ray Casting Using a Roped BVH with CUDA. Torres et.al. 2009. https://doi.org/10.1145/1980462.1980483

.. [Howard2019] Quantized bounding volume hierarchies for neighbor search in molecular simulations on graphics processing units. Howard et. al. 2019. https://doi.org/10.1016/j.commatsci.2019.04.004

.. [Howard2016] Efficient neighbor list calculation for molecular simulation of colloidal systems using graphics processing units. Howard et. al. 2016. https://doi.org/10.1016/j.cpc.2016.02.003
