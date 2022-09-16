Bonded Forces
==============

BondedForces is an :ref:`Interactor` Module.

Computes the interaction between (small) groups of particles (or a particle with a point in space) due to a bond-like ligature (like an harmonic or FENE spring). This module is generic for any number of particles per bond (pair, angular, torsional...).
This :ref:`Interactor` must be specialized with a bond potential (see :ref:`below <Available bond potentials>` for available potentials).
BondedForces delegates the :ref:`ParameterUpdatable` behavior to the provided bond potential.

This module is defined in Interactor/BondedForces.cuh.

.. _bondUsage:

Usage
********************

In order to create a BondedForces :ref:`Interactor` you need to specialize the module with the number of particles per bond and a bond potential. You can provide one of the :ref:`already available potentials <Available bond potentials>` or a custom one (such as the one defined :ref:`below <defining a new bond potential>`).

The function below uses the already defined Harmonic bond potential to create an instance of a particle-particle BondedForces Interactor.

The information for the bonds (a list of connected particles groups with bond-specific parameters) is read from a file provided in the parameter argument. See :ref:`below <bond file format>` for the format of this file.

.. code:: cpp
	  
  //You can use this function to create an interactor that can be directly added to an integrator
  std::shared_ptr<Interactor> createBondInteractor(std::shared_ptr<ParticleData> pd){
    using Bond = BondedType::Harmonic;
    //Specialize BondedForces for 2 particles per bond and the Harmonic bond potential.
    using BF = BondedForces<Bond,2>;
    typename BF::Parameters params;
    params.file = "bonds.dat";
    //Optionally, you can pass an instance of the bond potential as a shared_ptr, which will allow you to modify the bond properties at any time from outside BondedForces
    auto bond = std::make_shared<Bond>();
    auto bf = std::make_shared<BF>(pd, params, bond);
    return bf;
  }


Bond file format
*****************

The bond file contains a line for each bond in the system in an arbitrary order. The line for a bond contains a list of :ref:`ids <Particle id assignation>` for the connected particles and a bond-potential-specific list of parameters for that bond. Each bond must be provided only once (in a particle-particle bond file, only the pair i-j or the j-i should be present, but not both). The first line of the file should contain the total number of bonds in the file.

.. code:: bash
	  
    nbonds
    i0 i1...iN BONDINFO
    .
    .
    .
    nbondsFixedPoint <- Can be zero or not be in at all in the file
    i px py pz BONDINFO


Where :math:`i_0, \dots, i_N` are the indices of the particles in the bond. :code:`BONDINFO` can be any number of rows, as described by the bond potential BondedForces is used with. For example, in BondedType::Harmonic BONDINFO must be "k r0", meaning that the file needs 4 columns for particle-particle bonds.    

Note that a **particle pair has to be added only once**. So if particles 0 and 1 are bonded, only the line 0 1 k r0 (or 1 0 k r0) is needed.  

**Nbonds** is the number of particle-particle bonds. Note that 0 is a valid number of bonds.  

In the special case of two particles per bond, particles can be tethered to points in space (instead of other particle). This is referred to as a **fixed point** bond. **If BondedForces is specialized with 2 particles per bond**, the bond file can contain a list of fixed point bonds.
**NbondsFixedPoint** is the number of particle-point bonds. Each line for a fixed point bond contains the id of the particle, the 3D coordinates of the point in space and the parameters for the bond. The bond potential is informed of a fixed point bond by interpreting the point in space as a second particle with id=-1.


Example: The bond file for a single harmonic bond between two particles 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lets join particles with ids 0 and 1 with an Harmonic bond with k=2 and r0=0.1
Additionally, given that this will be used with a 2-particle BondedForces module, lets tether the particle with id=3 to the point in space (10,11,12) with the same k and r0

.. code:: bash
	  
	  1
	  0 1 2 0.1
	  1
	  3 10 11 12 2 0.1
	  
This bond file can be used as the *bond.dat* file in the example :ref:`above <bondUsage>`

Defining a new bond potential
*******************************

A bond potential must abide to the following interface

.. cpp:class:: BondPotential

   An interface that must be used by any class to be used as a bond potential in BondedForces.

   .. cpp:struct:: BondInfo

      A POD structure containing any required per-bond information.
   
   .. cpp:function:: ComputeType BondPotential::compute(int bond_index, int ids[NperBond], real3 pos[NperBond], Interactor::Computables comp, BondInfo bi)

     This function will be called for every bond read in the bond file and is expected to compute force/energy and or virial. **This must be a __device__ function**.
     
     :param bond_index: The index of the particle to compute force/energy/virial on
     :param ids: list of indexes of the particles involved in the current bond (in the same order as they were provided in the input file)
     :param pos: list of positions of the particles involved in the current bond
     :param comp: computable targets (wether force, energy and or virial are needed).
     :param bi: bond information for the current bond
     :return: The force/energy/virial for the particle in the bond, of type :cpp:any:`ComputeType`
	      
   .. cpp:function:: BondInfo BondPotential::readBond(std::istream &in)

     This function will be called for each bond in the bond file with the contents of the line after the particle indices. It must use the stream that is handed to it to construct and return a :cpp:any:`BondInfo`.  


.. note:: Note that this is not a virtual class to inherit, the BondedForces module is templated for the bond potential, so any class implementing the necessary methods can be used.
	  
     
.. cpp:class:: ComputeType

   A POD type holding members for the force, energy and virial

   .. cpp:member:: real3 force

   .. cpp:member:: real energy

   .. cpp:member:: real virial


      
   


.. code:: cpp

   __device__ real sq (real a){ return a*a;}
   
   //Harmonic bond for pairs of particles
   struct HarmonicBond{
     HarmonicBond(/*Parameters par*/){
       //In this case no parameter is needed beyond whats in the bond file.
     }
     //Place in this struct whatever static information is needed for a given bond
     //In this case spring constant and equilibrium distance
     //the function readBond below takes care of reading each BondInfo from the file
     struct BondInfo{
       real k, r0;
     };

     
     __device__ ComputeType compute(int bond_index,
                                    int ids[2], real3 pos[2],
				    Interactor::Computables comp,
				    BondInfo bi){
       real3 r12 = pos[1]-pos[0];
       real r2 = dot(r12, r12);
       const real invr = rsqrt(r2);
       const real f = -bi.k*(real(1.0)-bi.r0*invr); //F = -k·(r-r0)·rvec/r
       ComputeType ct;
       ct.force = f*r12;
       ct.energy = comp.energy?(real(0.5)*bi.k*sq(real(1.0)/invr-bi.r0)):real(0.0);
       ct.virial = comp.virial?dot(ct.force,r12):real(0.0);
       return (r2==real(0.0))?(ComputeType{}):ct;
     }
     
     static BondInfo readBond(std::istream &in){
       //BondedForces will read i j, readBond has to read the rest of the line
       BondInfo bi;
       in>>bi.k>>bi.r0;
       return bi;
     }
   };


.. hint:: Note that the :cpp:`compute` function takes arrays with as many elements as particles per bond.

   
.. hint:: Note that a bond potential functor may be :ref:`ParameterUpdatable`.
	  
.. note:: As usual, this :ref:`Interactor` can be added to an :ref:`Integrator`.


The argument :cpp:`comp` in the compute function is of type :cpp:type:`Interactor::Computables`, a POD structure containing boolean flags for the energy, force and virial.

The interface for a bond potential involving more than two particles is similar, but the :cpp:`compute` function would take as an argument a larger array (with as many elements as particles per bond).

The example in :code:`examples/interaction_schemes/Bonds.cu` contains more examples of Bond potentials.

Available bond potentials
******************************

Bond potentials are available for several types of bonds, all of them under the :cpp:any:`BondedType` namespace.

Pair bonds
~~~~~~~~~~~~~~

Bonds with two particles per bond

.. cpp:class:: BondedType::Harmonic

	       An harmonic bond encoding the potential :math:`U(r) = \half K\left(r-r_0\right)`
	       Requires the strength, :math:`K`, and the equilibrium distance :math:`r_0` in the bond file.
	       The constructor requires no arguments.
	       If a :cpp:any:`Box` is not provided this potential will not apply periodic boundary conditions to the distances between particles (to allow for bonds with equilibrium distances greater than L/2).

   .. cpp:function:: Harmonic(Box box = Box())

      The default constructor will make the box infinite (no periodic boundary conditions).


.. cpp:class:: BondedType::FENE

	       Implements the FENE potential, :math:`U(r) = \half K r_0^2\ln\left[1-\left(\frac{r}{r_0}\right)^2\right]`.
	       Requires the strength, :math:`K`, and the equilibrium distance :math:`r_0` in the bond file.
	       The constructor requires no arguments.
	       If a :cpp:any:`Box` is not provided this potential will not apply periodic boundary conditions to the distances between particles (to allow for bonds with equilibrium distances greater than L/2).

   .. cpp:function:: FENE(Box box = Box())

      The default constructor will make the box infinite (no periodic boundary conditions).


Angular bonds
~~~~~~~~~~~~~~~

Bonds with three particles per bond.

.. cpp:class:: BondedType::Angular
	       
	       Implements the potential :math:`U(\theta) = 2K\left[\sin(\theta/2) - sin(\theta_0/2)\right]^2`.
	       Requires the strength, :math:`K`, and the equilibrium angle :math:`\theta_0` in the bond file.
	       Applies the minimum image convention to the particle pair distances.

   .. cpp:function:: Angular(real3 boxSize)

		     The constructor of this class requires a domain size to apply periodic boundary conditions.

		     
Dihedral (torsional) bonds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bonds with four particles per bond.

.. cpp:class:: BondedType::FourierLAMMPS
	       
	       Implements the `Fourier LAMMPS <https://docs.lammps.org/dihedral_fourier.html>`_ like potential :math:`U(\phi) = 2K\left[1+\cos(\phi -\phi_0)\right]` (where :math:`\phi\in [-\pi, \pi]`).
	       Requires the strength, :math:`K`, and the equilibrium angle :math:`\phi_0` in the bond file.
	       Applies the minimum image convention to the particle pair distances.

   .. cpp:function:: FourierLAMMPS(Box box)

		     The constructor of this class requires a :cpp:any:`Box` object to deal with boundary conditions.

.. cpp:class:: BondedType::Torsional

	       An harmonic torsional bond defined in :code:`Interactor/TorsionalBondedForces.cuh`.
	       Applies the minimum image convention to the particle pair distances.

   .. cpp:function:: Torsional(real3 lbox)

		     The constructor of this class requires a domain size to apply periodic boundary conditions.


