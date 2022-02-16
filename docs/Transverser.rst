Transverser
============


Some modules in UAMMD require some kind of particle traversal.

Say, for instance, that for each particle we want to visit the rest of the particles that are closer than a certain distance. Or simply all of the other particles. More generally, we might want to perform some kind of operation equivalent to a matrix-vector multiplication, for which in order to compute one element of the result, the vector needs to go through a row of the matrix.
In these cases, a :cpp:any:`Transverser` is used.

.. note:: The word "Transverser" was chosen to convey that it is used to traverse and transform

A :cpp:any:`Transverser` holds information about what to do with a pair of particles, what information is needed to compute this interaction, and what to do when a particle has interacted with all pairs it is involved in.  

Being such a general concept, a :cpp:any:`Transverser` is used as a template argument, and therefore cannot be a base virtual class that can be inherited. This is why it is a "concept". No assumption can be made about the return types of each function, or the input parameters, the only common things are the function names. In other words, a class abiding to the Transverser concept does not require to inherit from a certain base class (such as for instance, an :ref:`Integrator`) rather simply define a series of functions following the rules laid our below.

For each particle to be processed the :cpp:any:`Transverser` will be called for:
 * Setting the initial value of the interaction result (function :cpp:any:`Transverser::zero`)
 * Fetching the necessary data to process a pair of particles  (function :cpp:any:`Transverser::getInfo`)
 * Compute the interaction between the particle  and each of its neighbours (function :cpp:any:`Transverser::compute`)
 * Accumulate/reduce  the result for each neighbour (function :cpp:any:`Transverser::accumulate`)
 * Set/write/handle the accumulated result for all neighbours (function :cpp:any:`Transverser::set`)

The same :cpp:any:`Transverser` instance will be used to process every particle in an arbitrary order. Therefore, the Transverser must not assume it is bound to a specific particle.

The :cpp:any:`Transverser` interface requires a given class/struct to provide the following public device (unless, "prepare", that must be a host function) member functions:

.. cpp:class:: Transverser

	       
   .. cpp:function:: Compute compute(real4 position_i, real4 position_j,Info info_i, Info info_j);

      For a pair of particles characterized by position and info this function must return the  result from the interaction for that pair of particles. The last two arguments must be present only when :cpp:any:`getInfo` is defined.The returning type, :cpp:any:`Compute`, must be a POD type (just an aggregate of plain types), for example a :cpp:any:`real4`.

   .. cpp:function:: void set(int particle_index, Compute &total);
  
      After calling compute for all neighbours this function will be called with the contents of "total" after the last call to "accumulate".  Can be used to, for example, write the final result to main memory.

   .. cpp:function:: Compute zero();

      This function returns the initial value of the computation, for example {0,0,0} when computing the force. The returning type, :cpp:any:`Compute`, must be a POD type (just an aggregate of plain types), for example a :cpp:any:`real4`. Furthermore it must be the same type returned by the "compute" member.
      This function is optional and defaults to zero initialization (it will return Compute() which works even for POD types).
    
   .. cpp:function:: Info getInfo(int particle_index);
   
      Will be called for each particle to be processed and returns the per-particle data necessary for the interaction with another particle (except the position which is always available). For example the mass in a gravitational interaction or the particle index for some custom interaction. The returning type, :cpp:any:`Info`, must be a POD type (just an aggregate of plain types), for example a :cpp:any:`real4`. **This function is optional and if not present it is assumed the only per-particle data required is the position**. In this case the function "compute" must only have the first two arguments.

   .. cpp:function:: void accumulate(Compute &total, const Compute &current);
   
      This function will be called after :cpp:any:`compute` for each neighbour with its result and the accumulated result. It is expected that this function modifies :cpp:any:`total` as necessary given the new data in :cpp:any:`current`.  The first time it is called :cpp:any:`total` will be have the value as given by the :cpp:any:`zero` function. This function is optional and defaults to summation: :cpp:`total = total + current`. Notice that this will fail for non trivial types.
     
   .. cpp:function:: void prepare(std::shared_ptr<ParticleData> pd);

      This function will be called one time on the CPU side just before processing the particles.
      This function is optional and defaults to simply nothing.


Example
----------

The example code below contains a very bare-bones instance of a :cpp:any:`Transverser`. In particular, :cpp:`NeighbourCounter` relies on as much default behavior as possible, presenting only a :cpp:any:`compute` and :cpp:any:`set` functions.
If we apply the :cpp:`NeighbourCounter`  :cpp:any:`Transverser` to one of the :ref:`neighbour lists` in UAMMD, the output (:code:`nneigh` array) will hold, for each particle, the number of neighbour particles.

.. code:: c++
  	  
  struct NeighbourCounter{
    int *nneigh;
    real rc;
    Box box;
    NeighbourCounter(Box i_box, real i_rc,int *nneigh):
      rc(i_rc),box(i_box),
      nneigh(nneigh){}
  
    //There is no "zero" function so the total result starts being 0.
    
    //For each pair computes counts a neighbour 
    //if the particle is closer than rcut
    __device__ auto compute(real4 pi, real4 pj){
      const real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));
      const real r2 = dot(rij, rij);
      if(r2>0 and r2< rc*rc){
        return 1;
      }
      return 0;
    }
    //There is no "accumulate"
    // the result of "compute" is added every time.
    //The "set" function will be called with the accumulation
    // of the result of "compute" for all neighbours. 
    __device__ void set(int index, int total){
      nneigh[index] = total;
    }
  };

Alternatively, if we apply the :cpp:any:`Transverser` above to the :ref:`NBody` module each particle will go through every other one, and thus all the elements of the :cpp:`NeighbourCounter` output will be equal to the total number of particles.
