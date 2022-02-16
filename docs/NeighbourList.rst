
.. _NeighbourList:

Neighbour Lists
================



.. cpp:class:: NeighbourList

	       A conceptual interface class that all UAMMD neighbour lists share.


   .. cpp:function:: NeighbourList(std::shared_ptr<ParticleData> pd);

      Initializes the neighbour list acting on all particles.

   .. cpp:function:: NeighbourList(std::shared_ptr<ParticleGroup> pg);

      Initializes the neighbour list acting on a group of particles.

   .. cpp:function:: void update(Box box, real cutOff, cudaStream_t st = 0);

      Uses the latest positions of the particles in the :ref:`ParticleData` instance to generate a list of neighbours.

   .. cpp:function:: void transverseList(Transverser &tr, cudaStream_t st = 0);

      Applies a given :cpp:any:`Transverser` to the current state of the neighbour list.

   .. cpp:function:: auto getNameList();

      Each NeighbourList provides a function (where "Name" is the name of the particular list) that returns the internal data structures of the list (see :ref:`available neighbour lists`). Arrays in the structure returned by this function will be allocated in the GPU.

   .. cpp:function:: auto getNeighbourContainer();

      Returns a :cpp:any:`NeighbourContainer` that allows accessing the neighbours of each particle from the GPU.



Example
---------

Constructing and using a neighbour list.

Once constructed, there are three ways to use a neighbour list:
 * Providing a :ref:`Transverser` (:cpp:any:`NeighbourList::transverseList`).
 * Obtaining a :cpp:any:`NeighbourContainer` (:cpp:any:`NeighbourList::getNeighbourContainer`).
 * Using the internal data structures of the particular neighbour list (see for instance :cpp:any:`CellList::getCellList`).

.. code:: cpp

  #include"uammd.cuh"
  #include"Interactor/NeighbourList/CellList.cuh"

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
    
    //auto ni = nl->getNeighbourContainer();    
    //nl.transverseList(some_transverser);
    return 0;
  }
  
.. hint:: See :cpp:any:`Neighbourcontainer` below.
.. hint:: Here, :cpp:expr:`some_transverser` is an instance of a :ref:`Transverser`.

	  

Available neighbour lists
--------------------------


CellList
~~~~~~~~

The main idea behind the cell list is to perform a spatial binning and assign a hash to each particle according to the bin it is in. If we then sort these hashes we get a list in which all the particles in a given cell are contiguous. By accessing, for a certain particle, the particles in the 27 surrounding cells we can find its neighbours without checking too many false positives.

The algorithm for the cell list construction can be summarized in three separate steps:
 * Hash (label) the particles according to the cell (bin) they lie in.
 * Sort the particles and hashes using the hashes as the ordering label (technically this is known as sorting by key). So that particles with positions lying in the same cell become contiguous in memory.
 * Identify where each cell starts and ends in the sorted particle positions array.

After these steps we end up with enough information to visit the 27 neighbour cells of a given particle.
We have to compute the assigned cell of a given position at several points during the algorithm. Doing this is straightforward. For a position inside the domain, :math:`x \in [0, L)`, the bin assigned to it is :math:`i = \textrm{floor}(x/n_x) \in [0, n_x- 1]`. It is important to notice that a particle located at exactly :math:`x = L` will be assigned the cell with index :math:`n_x`, special consideration must be taken into account to avoid this situation. In particular, in a periodic domain, a particle at :math:`x=L` should be assigned to the cell :math:`i=0`.

.. cpp:class:: CellList

	       Besides the functions defined in :cpp:any:`NeighbourList`, the cell list also exposes some functions proper to this particular algorithm. 

   .. cpp:function:: CellListBase::CellListData getCellList();

      Returns the internal structures of the CellList.


The cell list stores a copy of the particle positions sorted in such a way that the indices of the particles that are located in the same cell lie contiguous in memory:
 * SortPos: [all particles in cell 0, all particles in cell 1,..., all particles in cell ncells]


      
.. cpp:struct:: CellListBase::CellListData;
		
   .. cpp:member:: const uint * cellStart;

      Encodes the index (in :cpp:any:`sortPos`) of the first particle in a given cell. In particular, particles with indices (in :cpp:any:`sortPos`) :math:`j\in [\text{cellStart}[i]-\text{VALID\_CELL}, \text{cellEnd}[i])` are located in cell i and have positions given by :cpp:expr:`sortPos[j]` (j is an internal indexing of the cell list, to get the group index of the particle you can use :cpp:expr:`groupIndex[j]`).
      
   .. cpp:member:: const int  * cellEnd;

      Stores the last particle in :cpp:any:`sortPos` that lies in cell i.
      
   .. cpp:member:: const real4 *sortPos;

      Particle positions sorted in the order described by :cpp:any:`cellStart` and :cpp:any:`cellEnd`.
		   
   .. cpp:member:: const int* groupIndex; 

      The group index (see :ref:`ParticleGroup`) of the particle with position :cpp:expr:`sortPos[i]` is :cpp:expr:`groupIndex[i]`.
		   
   .. cpp:member:: Grid grid;

      The grid information of the cell list (stores things like number of cells).
		   
   .. cpp:member:: uint VALID_CELL;

      A value in cellStart less than VALID_CELL means the cell is empty. Subject to change between updates.



.. _NeighbourContainer:

The Neighbour Container interface
---------------------------------


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
                       //Set ni to provide iterators for particle i
                       ni.set(i);
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
