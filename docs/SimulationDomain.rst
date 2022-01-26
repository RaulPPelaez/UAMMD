Simulation domain
==================

Some modules require a domain to work, mainly when periodicity is involved. For instance as in the case of the :ref:`force coupling method` triply periodic hydrodynamics module.

In these cases the :cpp:class:`Box` class is used.

.. cpp:class:: Box

   A structure containing a domain size. Can describe a domain that is periodic in any direction.
   
   .. cpp:function:: Box::Box(real3 L)

      Constructor taking a box size in each direction. The resulting box is periodic by default except when L is infinite in some direction.

   .. cpp:function:: real3 Box::apply_pbc(real3 position)

      Applies the minimum image convention (MIC) to the provided position in the directions in which the box is periodic (leaving the rest untouched).
      Returns :math:`q^\alpha= q^\alpha - \text{floor}\left(q^\alpha/L^\alpha + 0.5\right)L^\alpha` applied only to the periodic directions. Being :math:`L^\alpha` the box size in the direction :math:`\alpha`.
      Note that this functions works for finding the minimum distance between two particles as well as for finding the position of the provided image in the primary cell.

   .. cpp:function:: void Box::setPeriodicity(bool x, bool y, bool z)

      Sets the periodicity of the box in each direction (note that the box can be finite yet non-periodic).

   .. cpp:function:: bool Box::isPeriodicX()

      Returns true if the box is periodic in the X direction, false otherwise.

   .. cpp:function:: bool Box::isPeriodicY()

      Returns true if the box is periodic in the Y direction, false otherwise.
      
   .. cpp:function:: bool Box::isPeriodicZ()

      Returns true if the box is periodic in the Z direction, false otherwise.

   .. cpp:member:: real3 Box::boxSize

      A public member that holds the box size in each direction.


Example
----------


.. code:: cpp

  #include<uammd.cuh>
  using namespace uammd;
  int main(){
    real lx, ly, lz;
    lx = ly = lz = 32.0;
    //A Box requires the size of the domain in each direction, which can be infinite
    Box box({lx, ly, lz});
    //Periodicity can be set independently for each direction. 1 meaning periodic and 0 aperiodic
    box.setPeriodicity(1,1,0);
    //The unit cell goes from -L/2 to L/2 in each direction.
    //Let's store in a variable a position just outside the unit cell.
    real3 position_outside_box = {0.5*lx+1, 0, 0};
    //Given that we have set the box as periodic in X, the position will be folded to the other side of the domain
    real3 position_in_box = box.apply_pbc(position_outside_box);
    //position_in_box holds {-lx*0.5+1,0,0}
    return 0; 
  }
  
