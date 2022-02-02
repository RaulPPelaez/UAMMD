Molecular Dynamics
=====================

At the lowest level our simulation units are interacting atoms or molecules whose motion can be described using classical mechanics. We refer to the numerical techniques used in this regime as Molecular Dynamics (MD).
In MD molecules are moving in a vacuum following the Newtonian equation of motion. If we need to include some kind of solvent, i.e. water, in an MD simulation we must do so by explicitly solving the motion of all the involved molecules of water (with some suitable force field).
Even so, MD represents the basis for all particle-based methods, where the term *particle*, depending on the level of coarse graining, might refer to anything from an atom to a colloidal particle.
Although it is not the most fundamental way of expressing the equations of motion, we will stick to a somewhat simplified but still quite general approach. For a system of :math:`N` molecules interacting via a certain potential, Newton's second law states that the acceleration experienced by each particle comes from the total force, :math:`\vec{F}`, acting on it.

.. math::
   
   \vec{F} =  m\ddot{\vec{\ppos}} = m\vec{a},
   
where :math:`m` is the mass of the molecule, :math:`\vec{\ppos}` its position in cartesian coordinates and :math:`\vec{a}` its acceleration.
The force can usually be expressed as the gradient of an underlying potential energy landscape, :math:`U`.

.. math::   
   
   \vec{F} = -\nabla_{\vec{\ppos}} U(\{\vec{\ppos}_1,...,\vec{\ppos}_N\}),
   
which in general is a function of the positions of the particles. In UAMMD :ref:`Interactors <Interactor>` are used to provide forces.


.. sidebar::
   
   .. note:: :math:`^1`: Although other systems can also be simulated with MD. For example, MD could be used to simulate the dynamics a group of gas molecules inside a piston (in which volume is not conserved).

At their core, these equations are an expression of the conservation of the total energy of the entire system. Consequently, these equations of motion can be used to perform simulations in the so-called microcanonical ensemble (NVE) :math:`^1`, where the number of particles (N), the volume of the domain (V) and the total energy (E) are conserved. Note, however, that the energy is only conserved when forces stem from the gradient of a potential.


.. hint:: Molecular Dynamics allow to model the Microscopic level of description, where the relevant variables are the positions and moments of every particle in the system, including the solvent and solute particles. This level is dominated by the mean collision time between the particles, of the order of :math:`\tau \sim 10^{-12} s`. The (purely Lagrangian) Newtonian equations of motion describe the dynamics of the system.


UAMMD offers the well known velocity Verlet algorithm [1]_, which is second-order accurate in the velocity and third-order accurate in the positions.
The velocity Verlet update rules, taking the simulation from the time step :math:`n` (time :math:`t=\dt n`) to the next, :math:`n+1`, can be summarized as

.. math::
   
   \vec{\pvel}^{n+\half}&= \vec{\pvel}^n + \half \vec{a}^n\dt\\
   \vec{\ppos}^{n+1}      &= \vec{\ppos}^n +  \vec{\pvel}^{n+\half}\dt\\
   \vec{\pvel}^{n+1}      &= \vec{\pvel}^{n+\half} + \half\vec{a}^{n+1}\dt


The velocity Verlet algorithm presents all the necessary properties for being a popular integrator. It has good numerical stability, a small memory footprint, good energy conservation, guaranteed momentum conservation,...

.. node:: The code for this module is located in the source code :code:`Integrator/VerletNVE.cuh`

Usage
--------------

Use it as any other integrator module.  
The following parameters are available:  
  * :cpp:`real energy` Target energy per particle, can be omitted if :cpp:`initVelocities=false`.
  * :cpp:`real dt` Time step.
  * :cpp:`real mass = -1` Mass of all the particles. If >0 all particles will have this mass, otherwise the mass for each particle in :ref:`ParticleData` will be used. If masses have not been set in :ref:`ParticleData` the default mass is 1 for all particles.  
  * :cpp:`bool is2D = false` Set to true if the system is 2D  
  * :cpp:`bool initVelocities=true` Modify starting velocities to ensure the target energy. When :cpp:`false` the velocities of the particles are left untouched at initialization. Note that :cpp:`false` will cause the :cpp:`energy` parameter to be ignored.

.. code:: cpp
   
  #include"uammd.cuh"
  #include"Integrator/VerletNVE.cuh"
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    using NVE = VerletNVE;
    NVE::Parameters params;
    params.energy = 1.0; //Target energy per particle, can be omitted if initVelocities=false
    params.dt = 0.1;
    //params.is2D = true; //If true, VerletNVE will not modify the Z coordinate of the particles. This parameter defaults to false.
    //params.initVelocities=true; //Modify starting velocities to ensure the target energy, if not present it defaults to true.
    verlet = make_shared<NVE>(pd,  params);
      ...
    //Add any interactor
    verlet->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    verlet->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.
  
.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.

	    
.. warning:: Beware that the NVE Verlet algorithm will present an energy drift coming from numerical accuracy that will be most visible in single precision mode [2]_.


****

.. rubric:: References
	    
.. [1] https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet  
.. [2] A common, avoidable source of error in molecular dynamics integrators. Ross A. Lippert et. al. Journ. of Chem. Phys. 2007. http://dx.doi.org/10.1063/1.2431176
