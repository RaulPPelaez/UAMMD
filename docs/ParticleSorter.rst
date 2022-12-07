Particle Sorter
================



.. cpp:class:: ParticleSorter

   A helper class to sort particles according to their positions following a certain rule.
   This rule can be a morton hash, so the particle positions are sorted to follow a Z-order curve, a cell hash, particle ID...
	       
   .. cpp:function:: template<class HashIterator> void updateOrderWithCustomHash(HashIterator &hasher, uint N, uint maxHash = std::numeric_limits<uint>::max(), cudaStream_t st = 0);

      Sets the current sorting according to the input hashes.
      The maximum possible hash (does not have to be included in the list) can be provided, which will improve performance.
    
   .. cpp:function:: template<class CellHasher = Sorter::MortonHash, class InputIterator>   void updateOrderByCellHash(InputIterator pos, uint N, Box box, int3 cellDim, cudaStream_t st = 0);

      Sets the current sorting using the linear index of the cell of the given positions

   .. cpp:function:: void updateOrderById(int *id, int N, cudaStream_t st = 0);

      Sets the current sorting using the input list of ids as hashes

   .. cpp:function:: template<class InputIterator, class OutputIterator> void applyCurrentOrder(InputIterator d_property_unsorted, OutputIterator d_property_sorted, int N, cudaStream_t st = 0);

      Applies the latest updated order to some input vector.
      WARNING: _unsorted and _sorted cannot be aliased!

   .. cpp:function:: int * getSortedIndexArray(int N);

      

   .. cpp:function:: uint * getSortedHashes();

      Returns the current stored sorted hashes.

   .. cpp:function:: int * getIndexArrayById(int * id, int N, cudaStream_t st = 0);

      Sort the sequence 0:N using a copy of the input id array as keys.



Example
--------

.. code:: c++
	  
   //Create an instance of ParticleSorter:
   ParticleSorter ps;
   //Update/create a sorted index list from the positions using a certain hash
   ps.updateOrderByHash<Sorter::MortonHash>(pos, numberParticles, BoxSize, cudaStream);
   //Apply new order to some array
   ps.applyCurrentOrder(array_in, array_out, numberParticles, cudaStream);

   //Order the id array (without changing it) as keys and a 0..numberparticles array as values,
   //return this device array. It does not affect the current order.
   ps.getIndexArrayById(id, numberParticles, cudaStream);
