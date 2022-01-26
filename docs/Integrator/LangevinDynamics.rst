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





.. rubric:: References

.. [1] A simple and effective Verlet-type algorithm for simulating Langevin dynamics. Niels   Grønbech-Jensen  and  Oded   Farago 2013. https://doi.org/10.1080/00268976.2012.760055
   
