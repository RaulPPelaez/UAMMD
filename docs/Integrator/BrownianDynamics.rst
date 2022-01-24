Brownian Dynamics
=================
-----------------------------------------------------
Brownian Dynamics :ref:`integrators <Integrator>`
-----------------------------------------------------

There are several Integrators under the BD namespace, which solve the following differential equation:  
  
.. math::

   d\vec{\ppos} = M\vec{F}dt + \sqrt{2\kT M dt}d\vec{W}

Where:

  * :math:`\vec{\ppos}` - Particle positions (:math:`\vec{\ppos} = \{\vec{\ppos}_1, \dots, \vec{\ppos}_N\}`)
  * :math:`\vec{F}` - Particle forces
  * :math:`M = \frac{1}{6\pi \eta a}` - Mobility -> :math:`M = D/\kT`. Here :math:`\eta` is the fluid viscosity and :math:`a` the hydrodynamic radius of the particles.
  * :math:`d\vec{W}`- Brownian noise vector (gaussian numbers with mean=0, std=1)
    


Hydrodynamic interactions are not considered in this module. See :ref:`Brownian Hydrodynamics` for solvers with hydrodynamic interactions.


The following update rules are available:

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

     
Note that, as stated in [5]_, while this solver seems to be better than the rest at sampling equilibrium configurations, it does not correctly solves the dynamics of the problem.

-----------------------------------------------------
Usage
-----------------------------------------------------

Use it as any other integrator module.  
The following parameters are available:  

  * :code:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :code:`real viscosity` Viscosity of the solvent.
  * :code:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles)*
  * :code:`real dt`  Time step
  * :code:`bool is2D = false` Set to true if the system is 2D  

\* If this parameter is not provided, the module will try to use the particle's radius as the hydrodynamic radius of each particle. If particle radius has not been allocated prior construction of EulerMaruyama it will fail.  

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
    //Add any interactor[2]
    bd->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    bd->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

****

.. rubric:: References:  

.. [3] Temporal Integrators for Fluctuating Hydrodynamics. Delong et. al. (2013) Phys. Rev. E 87, 033302.  
.. [4] Brownian dynamics of confined suspensions of active microrollers. Balboa et. al. (2017) J. Chem. Phys. 146; https://doi.org/10.1063/1.4979494  
.. [5] The computation of averages from equilibrium and nonequilibrium Langevin molecular dynamics. Leimkuhler et. al. IMA J. Numerical Analysis 36, 1 (2016) https://doi.org/10.1093/imanum/dru056  
.. [6] An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations. Desmond J. Higham. (2001).  https://doi.org/10.1137/S0036144500378302  
