Simulation domain
==================

Some modules require a domain to work, mainly when periodicity is involved. For instance as in the case of the :ref:`force coupling method` triply periodic hydrodynamics module.

In these cases the :cpp:`Box` class is used.

.. cpp:class:: Box

   A structure containing a domain size. Can describe a domain that is periodic in any direction.
   
   .. cpp:function:: Box::Box(real3 L)

      Constructor taking a box size in each direction. The resulting box is periodic by default except when L is infinite in some direction.

   .. cpp:function:: real3 apply_pbc(real3 position)

      Applies the minimum image convention (MIC) to the provided position in the directions in which the box is periodic (leaving the rest untouched).
      Returns :math:`q^\alpha= q^\alpha - \text{floor}\left(q^\alpha/L^\alpha + 0.5\right)L^\alpha` applied only to the periodic directions. Being :math:`L^\alpha` the box size in the direction :math:`\alpha`.
      Note that this functions works for finding the minimum distance between two particles as well as for finding the position of the provided image in the primary cell.

   .. cpp:function:: void setPeriodicity(bool x, bool y, bool z)

      Sets the periodicity of the box in each direction (note that the box can be finite yet non-periodic).

   .. cpp:function:: bool isPeriodicX()

      Returns true if the box is periodic in the X direction, false otherwise.

   .. cpp:function:: bool isPeriodicY()

      Returns true if the box is periodic in the Y direction, false otherwise.
      
   .. cpp:function:: bool isPeriodicZ()

      Returns true if the box is periodic in the Z direction, false otherwise.



