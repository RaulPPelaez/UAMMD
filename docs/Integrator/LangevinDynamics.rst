Langevin Dynamics
====================

At the Langevin level of description, the dynamics of the solvent molecules (e.g. water) are orders of magnitude faster than the solute particles, leaving the positions and momenta of the solute particles as relevant variables. The dynamics of the solute particles are governed by a so-called Langevin equation, a manifestation of the Fokker-Plank equation that is often described as ":ref:`Molecular Dynamics` with a thermostat".

We can write the Langevin stochastic differential equation as

.. math::

   m\, d\vec{\pvel} = \underbrace{\vec{F}}_{\vec{\partial}_{\vec{\ppos}}U(\ppos)}dt - \overbrace{\xi\vec{\pvel}}^{\text{Drag}}dt + \underbrace{\vec{\beta}}_{\text{Fluctuations}},

   
which can be interpreted as a form of :ref:`Molecular Dynamics` coupled with a thermostat so that noise and drag forces are balanced according to the fluctuation-dissipation relation, guaranteeing the correct thermalization of the system (NVT ensemble). Here :math:`\vec{\pvel} = \dot{\vec{\ppos}}` are the velocities of the particles.

On the other hand, the LD equation is an expression of the equivalent stochastic differential equation (SDE) for the Fokker-Plank equation with the particle momenta and positions as the relevant variables. The first term in the Langevin SDE , :math:`\vec{F}`, is the sum of the conservative forces acting on particle :math:`i` (the ones coming from the underlying potential energy landscape), these interactions can be steric, electrostatic, bonded, etc. In UAMMD, forces are provided to the :ref:`Integrator` modules via :ref:`Interactors <Interactor>`.
The second term represents the drag exerted by the solvent particles in the form of a dissipative force.
Finally, the third term represents the fluctuations produced by the fast and constant collisions of the solvent particles. Here :math:`\vec{\beta}` is a random increment which must be in fluctuation-dissipation balance with the friction term.

The friction coefficient, :math:`\xi`, is related with the damping rate as :math:`\gamma = \xi/m`, which represents the decorrelation time of the velocity, :math:`\tau = m/\xi`. Additionally, :math:`\xi` can be formally derived from a Green-Kubo relation involving the integral of the solvent-solute force time-correlation. Its value is often approximated by the Stokes (macroscopic) value :math:`\xi=6\pi\eta a`, of a particle of radius :math:`a` in a solvent with viscosity :math:`\eta`.

The Fokker-Planck formalism teaches us that when a system dissipates energy, for instance, due to a force like the drag force in the Langevin SDE, the fluctuation-dissipation theorem states that there must be an opposite process that reintroduces this energy via thermal fluctuations.

In this case, fluctuation-dissipation enforces a fluctuating term with zero mean,

.. math::
   
  \left\langle\beta\right\rangle = 0,

which is uncorrelated in time (since we are assuming the time interval is small enough to consider the transport coefficients as constants) with standard deviation given by

.. math::

   \left\langle\beta(t)\beta(t')\right\rangle = 2\xi\kT dt\delta(t-t').


We can then write the Langevin equation as

.. math::
   
  m\, d\vec{\pvel} = \vec{F}dt - \xi\vec{\pvel}dt +  \sqrt{2\xi\kT}\vec{\noise},

  
Where we have defined :math:`\vec{\beta} := \sqrt{2\xi\kT}\vec{\noise}`.

Technically, the restrictions for :math:`\vec{\beta}` would be compatible with any random distribution for the noise, :math:`\vec{\noise}`, that has zero mean and variance :math:`dt`. However, as the noise term comes from the action of countless independent random variables (collisions with the solvent particles) the central limit theorem applies. Thus, it is convenient to choose :math:`\vec{\noise}` as a Gaussian white noise (i.e. a Wiener process).


UAMMD exposes a Verlet-like algorithm to numerically integrate the Langevin SDE called Grønbech-Jensen [1]_ 


The update rule, taking the simulation from the time step :math:`n` (time :math:`t=\dt n`) to the next, :math:`n+1`, can be summarized as

.. math::
   
   \vec{\ppos}^{n+1}  &=  \vec{\ppos}^n + b \dt \vec{\pvel}^n + \frac{b\dt^2}{2m}\vec{F}^n + \frac{b\dt}{2m}\vec{\beta}^{n+1}\\
   \vec{\pvel}^{n+1} &= a\vec{\pvel}^n + \frac{\dt}{2m}\left(a\vec{F}^n + \vec{F} ^{n+1}\right) +  \frac{b}{m}\vec{\beta}^{n+1}

   
Where

.. math::
   
   b:&=\frac{1}{1+\frac{\xi\dt}{2}}\\
   a:&=b \left(1-\frac{\xi\dt}{2}\right)

Note that, since we need the force at :math:`t+\dt` to compute the velocities at :math:`t+\dt` the forces have to be a function of the positions only (and not velocities).
   
.. note:: The code for this module is located in the source code :code:`Integrator/VerletNVT/GronbechJensen.cuh`.

Usage
---------

Grønbech-Jensen [1]_ is available as an :ref:`Integrator`.


.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.




The following parameters are available:  
  * :code:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :code:`real friction` Friction, :math:`\xi`.
  * :cpp:`real dt` Time step.
  * :cpp:`real mass = -1` Mass of all the particles. If >0 all particles will have this mass, otherwise the mass for each particle in :ref:`ParticleData` will be used. If masses have not been set in :ref:`ParticleData` the default mass is 1 for all particles.  
  * :cpp:`bool initVelocities=true` Modify starting velocities to ensure the target temperature from the start. When :cpp:`false` the velocities of the particles are left untouched at initialization. The default is true and sets particle velocities following the botzmann distribution.
  * :code:`bool is2D = false` Set to true if the system is 2D  


.. code:: cpp
	  
  #include"uammd.cuh"
  #include"Integrator/VerletNVT.cuh"
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    using NVT = VerletNVT::GronbechJensen;
    NVT::Parameters params;
    params.temperature = 1.0;
    params.dt = 0.1;
    params.friction = 1.0;
    auto verlet = std::make_shared<NVT>(pd, params);
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

	  

Dissipative Particle Dynamics
==============================

One of the most popular techniques used to reintroduce some of the degrees of freedom lost with LD is Dissipative Particle Dynamics (DPD). This coarse graining technique can be used to go further in the spatio-temporal scale by choosing groups of fluid particles as the simulation unit, sitting inbetween microscopic (as in MD) and macroscopic (hydrodynamic) descriptions. In practice DPD is a Langevin approach where friction acts by pairs of particles and conserves momentum.

In the standard DPD, particles interact via a soft potential, modelling the interaction between two large groups of fluid particles.
In many instances DPD is used as a momentum-conserving thermostat, which thus permits to include hydrodynamics (contrary to a single Langevin approach). Local momemtum conservation results in the emergence of macroscopic hydrodynamic effects. These momentum conserving forces can then be tuned to reproduce not only thermodynamics, but also dynamical and rheological properties of diverse complex fluids.
The equations of motion in DPD have the same functional form as LD and can be in fact considered as a momentum-conserving generalization of LD. The equations of motion for DPD read,

.. math::

   m\vec{a} = \vec{F^c} + \vec{F^d} + \vec{F^r}.
   
Where the three forces are traditionally expressed as,

.. math::

   \vec{F^c}_{ij} &=\omega(r_{ij})\hat{\vec{\ppos}}_{ij}\\
    \vec{F^d}_{ij} &=-\xi\omega^2(r_{ij})(\vec{\pvel}_{ij}\cdot\vec{\ppos}_{ij})\hat{\vec{\ppos}}_{ij}\\
    \vec{F^r}_{ij} &=\sqrt{2\xi\kT}\omega(r_{ij})\widetilde{W}_{ij}\hat{\vec{\ppos}}_{ij}    

Where :math:`\vec{\pvel}_{ij} = \vec{\pvel}_j - \vec{\pvel}_i` is the relative velocity between particles :math:`i` and :math:`j`. Here :math:`\xi` represents a friction coefficient and is related to the random force strength via fluctuation-dissipation balance in a familiar way [2]_. In general :math:`\xi` can be considered to be a tensorial quantity and even derived from atomistic simulations using dynamic coarse graining theory. The factor :math:`\widetilde{W}_{ij}` is different from the one in LD in that it affects pairs of particles (instead of each individual one), it also represents a Gaussian random number with zero mean and unit standard deviation, but must be chosen independently for each pair while ensuring symmetry so that :math:`\widetilde{W}_{ij} = \widetilde{W}_{ji}`.
The weight function :math:`\omega(r)` is a soft repulsive force usually defined as

.. math::

   \omega(r) =  \begin{cases}
    \alpha\left(1-\dfrac{\ppos}{r_{c}}\right) & r<r_{c}\\
    0 & r\ge r_{c}
    \end{cases}
    
Where :math:`r_{c}` is a cut-off distance. The strength parameter, :math:`\alpha`, can in principle be different for each pair of particles, :math:`i` - :math:`j`, but for simplicity we will assume it is the same for every pair.

.. note:: The code is easily generalized for a different per-particle strength and/or friction.

Being an SDE where the forces depend on the velocities, numerical integration of the DPD equations is tricky. A simple modification can be made, sacrificing stability, by approximating the velocity to just first order in the Gronbech-Jensen update rule, so that the velocity depends only on the force for the current step. Unfortunately, this leads to artifacts in the transport properties and unacceptable temperature drifts. There are several strategies in the literature trying to overcome this, usually presented as modifications of the velocity Verlet algorithm.

All of these methods improve the accuracy of the predicted transport properties and response functions in exchange for increased computational cost. 
One popular approach is to simply use the energy-conserving velocity Verlet (see :ref:`Molecular Dynamics`)  with the DPD forces. This yields "poor" stability and presents certain artifacts due to the mistreatment of the derivative of the noise term incurred by treating the DPD equations as an ordinary differential equation instead of a proper SDE. However, it is often good enough and while it might require a smaller time step to recover measurables to an acceptable tolerance it is the fastest approach and trivial to implement in a code already providing the velocity Verlet algorithm.

This is the approach used in UAMMD, where DPD is encoded as a :ref:`Molecular Dynamics`  :ref:`Integrator` coupled with a :ref:`PairForces`  :ref:`Interactor` encoding the DPD forces.

.. note:: The force-computing code for this module is located in the source code :code:`Interactor/Potential/DPD.cuh`

Usage
----------

A DPD :ref:`Integrator` is created by coupling a :code:`VerletNVE`  :ref:`Molecular Dynamics`  :ref:`Integrator`  with a :code:`DPD`  :ref:`Potential` (the Potential is supplied to a :ref:`PairForces`  :ref:`Interactor` that can be then added to the Integrator).

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.


The following parameters are available for the DPD :ref:`Potential`:
  * :cpp:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :cpp:`real cutOff` The cut off, :math:`r_c`, for the weight function.
  * :cpp:`par.gamma`  The friction coefficient, :math:`\xi`.
  * :cpp:`real A`  The strength of the weight function, :math:`\alpha`.
  * :cpp:`real dt` The time step. Be sure to pass the same time step to DPD and the Integrator.

.. code:: cpp

  #include<uammd.cuh>
  #include<Integrator/VerletNVE.cuh>
  #include<Interactor/PairForces.cuh>
  #include<Interactor/Potential/DPD.cuh>
  using namespace uammd;
  //A function that creates and returns a DPD integrator
  auto createIntegratorDPD(UAMMD sim){
    Potential::DPD::Parameters par;
    par.temperature = sim.par.temperature;
    //The cut off for the weight function
    par.cutOff = sim.par.cutOff;
    //The friction coefficient
    par.gamma = sim.par.friction; 
    //The strength of the weight function
    par.A = sim.par.strength; 
    par.dt = sim.par.dt;  
    auto dpd = make_shared<Potential::DPD>(dpd);
    //From the example in PairForces
    auto interactor = createPairForcesWithPotential(sim, dpd);
    //From the example in the MD section
    // particle velocities should not be initialized
    // by VerletNVE (initVelocities=false)
    using NVE = VerletNVE;
    NVE::Parameters params;
    params.dt = par.dt;
    params.initVelocities=false;
    verlet = make_shared<NVE>(pd,  params);
    verlet->addInteractor(interactor);
    return verlet;
  }

.. note:: The :code:`UAMMD` structure in this example is taken from the :code:`example/` folders in the repository, containing, for convenience, an instance of :ref:`ParticleData` and a set of parameters


.. rubric:: References

.. [1] A simple and effective Verlet-type algorithm for simulating Langevin dynamics. Niels   Grønbech-Jensen  and  Oded   Farago 2013. https://doi.org/10.1080/00268976.2012.760055
.. [2] Statistical Mechanics of Dissipative Particle Dynamics. P Español and P Warren 1995. https://doi.org/10.1209/0295-5075/30/4/001
