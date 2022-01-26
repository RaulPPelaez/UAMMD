Brownian Dynamics
=================



When the viscous forces are much larger than the inertial forces, i.e. :math:`|\xi\vec{\pvel}| \gg |m\vec{a}|`, inertial terms becomes irrelevant at very short time scales.
Brownian Dynamics (BD)[2]_ takes advantage of such a time scale separation between the particle velocity fluctuations and its displacement and can be interpreted as the overdamped, or non-inertial, limit of :ref:`Langevin Dynamics`. The decorrelation time of the velocity, defined as :math:`\tau_l = m/\xi` is much faster than the time needed for a particle to move farther than its own size. BD represents the long time limit of the Langevin equation. This is a powerful property of BD, since sampling the probability distributions of the underlying stochastic processes (stemming from the rapid movement of the solute particles) does not require sampling their fast dynamics.

In BD, the coupling between the submerged particles and the solvent is instantaneous.
Furthermore, since the particle velocities decorrelate instantly, the only remaining relevant variables are the positions of the colloidal particles.


Neglecting hydrodynamic interactions (see :ref:`Brownian Hydrodynamics`) we will focus on the simple case of the mobility being non-zero only on the diagonal. In the case of a no-slip sphere of radius :math:`a` moving inside a fluid with viscosity :math:`\eta` the bare self-mobility is given by the well-known Stokes drag :math:`M = (6\pi\eta a)^{-1}`.

The Brownian dynamics equation of motion are

.. math::
   
  d\vec{\ppos} = M\vec{F}dt + \sqrt{2\kT M}d\vec{\noise},


Where:

  * :math:`\vec{\ppos}` - Particle positions (:math:`\vec{\ppos} = \{\vec{\ppos}_1, \dots, \vec{\ppos}_N\}`)
  * :math:`\vec{F}` - Particle forces
  * :math:`M = (6\pi \eta a)^{-1}` - Mobility -> :math:`M = D/\kT`. Here :math:`\eta` is the fluid viscosity and :math:`a` the hydrodynamic radius of the particles.
  * :math:`d\vec{\noise}`- Brownian noise vector (gaussian numbers with mean=0, std=1)



-----------------------------------------------------
Brownian Dynamics :ref:`integrators <Integrator>`
-----------------------------------------------------

There are several :ref:`Integrators <Integrator>` in UAMMD under the :cpp:`BD` namespace, which solve the BD equation above:
  
    
EulerMaruyama
---------------

The simplest algorithm, described in [6]_, advances the simulation with the following rule:

.. math::
   
  \vec{\ppos}^{n+1} = \vec{\ppos}^n + dt(M\vec{F}^n) + \sqrt{2*\kT M dt}d\vec{W}
  
  
This algorithm has a convergence scaling of 1/2 (:math:`dt^{0.5}`).  

MidPoint
------------

A two step explicit midpoint predictor-corrector scheme (described in [3]_). It has a convergence scaling of :math:`dt^4` at the expense of having twice the cost of a single step method, as it requires to evaluate forces twice per update. Noise has to be remembered as well but in practice it is just regenerated instead of stored.  
MidPoint updates the simulation with the following rule:  

.. math::

   \vec{\ppos}^{n+1/2} &= \vec{\ppos}^n + \half dt(M \vec{F}^n) + \sqrt{\kT M dt}d\vec{W}^n_1\\
   \vec{\ppos}^{n+1} &= \vec{\ppos}^n +  dt(M \vec{F}^{n+1/2}) + \sqrt{\kT M dt}(d\vec{W}^n_1 + d\vec{W}^n_2)

   
AdamsBashforth
---------------

This algorithm, described in [4]_, uses the forces from the previous step to improve the prediction for the next. It incurs the overhead of storing the previous forces but its computational cost is marginally larger than EulerMaruyama. This algorithm mixes a first order method for the noise with a second order method for the force. It yields better accuracy than EulerMaruyama, although this comes from experience since as of the time of writing no formal work has been done on its weak accuracy.  
AdamsBashforth updates the simulation with the following rule:   

.. math::
   
    \vec{\ppos}^{n+1} = \vec{\ppos}^n + dt(1.5M\vec{F}^n - 0.5 M\vec{F}^{n-1}) + \sqrt{2\kT M dt}d\vec{W}^n
  
Leimkuhler
------------

Described in [5]_ (see eq.45). While also a first order method it seems to yield better accuracy than AB and EM. I am not aware of any formal studies of its accuracy.  
The update rule is very similar to EM but uses noise from two steps (which are generated each time instead of stored):

.. math::

   \vec{\ppos}^{n+1} = \vec{\ppos}^n + dtM\vec{F}^{n} + \sqrt{2\kT M dt}\half(d\vec{W}^n + d\vec{W}^{n-1})

.. warning:: Note that, as stated in [5]_, while this solver seems to be better than the rest at sampling equilibrium configurations, it does not correctly solves the dynamics of the problem.

-----------------------------------------------------
Usage
-----------------------------------------------------

Use it as any other integrator module.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:  

  * :code:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :code:`real viscosity` Viscosity of the solvent.
  * :code:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles*)
  * :code:`real dt`  Time step
  * :code:`bool is2D = false` Set to true if the system is 2D  

\* If this parameter is not provided, the module will try to use the particle's radius as the hydrodynamic radius of each particle. In the latter case, if particle radii has not been set in :ref:`ParticleData` prior to the construction of the module an error will be thrown.  


-----------------------------------------------------
Example
-----------------------------------------------------

.. code:: cpp
	  
  #include"uammd.cuh"
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    //Choose the method
    using BD = BD::EulerMaruyama;
    //using BD = BD::MidPoint;
    //using BD = BD::AdamsBashforth;
    //using BD = BD::Leimkuhler;
    BD::Parameters par;
    par.temperature=1;
    par.viscosity=1;
    par.hydrodynamicRadius=1;
    par.dt=0.01;
    //Optionally you can place a shear matrix, dX = M*F*dt + sqrt(2*D*dt)*dW + K*R
    //par.K = {{1,2,3},{1,2,3},{1,2,3}};
    //Or, if you want to set just one row:
    //par.K[0] = {1,2,3};    
    ...
    auto bd = make_shared<BD>(pd, par);
    ...
    //Add any interactor
    bd->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    bd->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.



	  

****

.. rubric:: References:  

.. [2] An Introduction to Dynamics of Colloids. Dhont 1996; https://www.elsevier.com/books/an-introduction-to-dynamics-of-colloids/dhont/978-0-444-82009-9
.. [3] Temporal Integrators for Fluctuating Hydrodynamics. Delong et. al. (2013) Phys. Rev. E 87, 033302.  
.. [4] Brownian dynamics of confined suspensions of active microrollers. Balboa et. al. (2017) J. Chem. Phys. 146; https://doi.org/10.1063/1.4979494  
.. [5] The computation of averages from equilibrium and nonequilibrium Langevin molecular dynamics. Leimkuhler et. al. IMA J. Numerical Analysis 36, 1 (2016) https://doi.org/10.1093/imanum/dru056  
.. [6] An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations. Desmond J. Higham. (2001).  https://doi.org/10.1137/S0036144500378302  
