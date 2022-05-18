Fluctuating Hydrodynamics
==========================

We refer to fluctuating hydrodynamics as the Navier-Stokes equations coupled with particles in a :ref:`FCM` fashion.
The algorithms for most UAMMD solvers for fluctuating hydrodynamics are laid out in [1]_  [2]_ and [3]_.
In general, we can write the Navier-Stokes equation as:

.. math::

   \partial_t \rho &= -\nabla\cdot\vec{g}\\
   \partial_t\vec{g} &= -\nabla\cdot(\vec{g}\otimes\vec{v}) - \nabla\cdot\tens{\sigma} + \vec{f} + \nabla\cdot\tens{Z}

Where:
    :math:`\rho(\vec{r},t)`: Fluid density.
	  
    :math:`\vec{v}(\vec{r},t)`: Fluid velocity.
	  
    :math:`\vec{g}=\rho\vec{g}`: Fluid momentum.
	  
    :math:`\tens{\sigma} = \nabla\pi - \eta\nabla^2\vec{v} - (\xi+\eta/3)\nabla(\nabla\cdot\vec{v})`: Deterministic stress tensor

    :math:`\pi`: Pressure.

    :math:`\vec{f}`: Any external fluid forcing, typically :math:`\oper{S}\vec{F}`, the spreaded particle forces.
	  
    :math:`\vec{g}\otimes\vec{v}`: I call this the kinetic tensor.
	  
    :math:`\tens{Z}`: Fluctuating stress tensor.


Depending on the particular solver, some terms might be zero. For instance, if the solver is incompressible, the term :math:`\nabla\cdot\vec{v}` vanishes and the density is constant.

When inertia is disregarded the equations fall back to the Stokes equations, UAMMD offers solvers for this regime, referred to as :ref:`Brownian Hydrodynamics`.

In particular, UAMMD offers fluctuating hydrodynamics solvers in two regimes:
 * **Incompressible**: :ref:`ICM`.
 * **Compressible**: :ref:`ICM_Compressible`.


Some solvers use a staggered grid, it is worth introducing it here:

.. _Staggered grid:

About Staggered grids
~~~~~~~~~~~~~~~~~~~~~~
In a staggered grid each quantity kind (scalars, vector or tensor elements) is
defined on a different subgrid.
Scalar fields are defined in the cell centers, vector fields in cell faces and
tensor fields are defined at the centers and edges.

Let us denote a certain scalar field with :math:`\rho`, a vector field with :math:`\vec{v}`
(with components :math:`v^\alpha`) and a tensor field with :math:`\tens{E}` (with components
:math:`E^{\alpha\beta}` ).

Say :math:`\vec{i}=(i_x, i_y, i_z)` represents a cell in the grid, which is centered at
the position :math:`\vec{r}_i`. Then, the different fields, corresponding to cell
:math:`\vec{i}` would be defined at the following locations:

  - :math:`\rho_{\vec{i}} \rightarrow \vec{r}_{\vec{i}}`
  - :math:`\vec{v}^\alpha_{\vec{i}} \rightarrow \vec{r}_{\vec{i}} + h/2\vec{\alpha}`
  - :math:`\tens{E}^{\alpha\beta}_{\vec{i}} \rightarrow \vec{r}_{\vec{i}} + h/2\vec{\alpha} + h/2\vec{\beta}`

Where :math:`\vec{\alpha}` and :math:`\vec{\beta}` are the unit vectors in those directions and :math:`h` is the size of a cell.

This rules result in the values assigned to a cell sometimes being defined in
strange places. The sketch below represents all the values owning to a certain
cell, :math:`\vec{i}` (with center defined at ○). Unintuitively, some quantities assigned
to cell :math:`\vec{i}` lie in the neighbouring cells (represented below is also cell
:math:`\vec{i} + (1,0,0)`).

.. code::
   
                    <------h---->
	+-----⬒-----▽-----------+  
	|      	    |	       	|  
	|      	    |	       	|  
	|     ○	    ◨  	  △    	|
	| 	    |  	       	|  
	|      	    |		|  
	+-----------+-----------+  

Where each symbol represents:
  * ○: :math:`\rho` (Cell center, at :math:`\vec{r}_{\vec{i}}`)
  * ◨: :math:`v^x`
  * ⬒: :math:`v^y`
  * △: :math:`E^{xx}`
  * ▽: :math:`E^{xy}`


Naturally, this discretisation requires special handling of the discretized versions of the (differential) operators. See for instance :code:`ICM_Compressible/SpatialDiscretization.cuh` to see how UAMMD deals with them.

For instance, multiplying a scalar and a vector requires interpolating the
scalar at the position of the vector (Since the result, being a vector, must be
defined at the vector subgrids).

:math:`\vec{g} := \rho\vec{v} \rightarrow g^\alpha_{\vec{i}} = 0.5(p_{\vec{i}+\vec{\alpha}} + p_{\vec{i}})v^\alpha_{\vec{i}}`

For more information, check out [1]_, [3]_ or Raul's manuscript.

.. _ICM_Compressible:

Compressible Inertial Coupling Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the compressible inertial coupling method we employ a staggered grid for the spatial discretization of the Navier-Stokes equations.

Particles dynamics are integrated via a predictor-corrector Euler scheme (forces are only computed once). By default, the particle-fluid coupling is mediated via a three point Peskin kernel.

The algorithm is described in detail in Appendix A of [1]_ or in [3]_.
Check the files under ICM_Compressible for detailed information about the solver.

This solver is triply periodic, although walls and such could be included.

In order to evaluate the pressure we use a provided equation of state, by default :math:`\pi(\rho)=c_t^2\rho`.

Both of the Navier-Stokes equations can be written as a conservation equation
with the following form: :math:`U^c = AU^a + B(U^b + \Delta U(U^b, W^c))`

Where :math:`U` might be the density or the fluid velocity and :math:`(a,b,c)` are three different time points inside a time step (we use a third order Runge Kutta integrator).
In order to go from the time step :math:`n` to :math:`n+1` the solver must be called three times for the density and then the velocity:
      
  1. :math:`a=0`, :math:`b=n` and :math:`c=n+1/3`
  2. :math:`a=3/4`, :math:`b=1/4` and :math:`c=n+2/3`
  3. :math:`a=1/3`, :math:`b=2/3` and :math:`c=n+1`

The values of :math:`A` and :math:`B` allow to choose between different temporal discretizations.

The current implementation uses, for each subtime respectively:
  1.  :math:`A=0, B=1`
  2.  :math:`A=3/4, B=1/4`
  3.  :math:`A=1/3, B=2/3` 

In both cases, we can define :math:`\Delta U = -dt\nabla\cdot\tens{F} + dt\vec{f}`.

Where :math:`\tens{F}(U,W,t)` means one thing or another depending on the equation we are solving. :math:`\vec{f}` is only non-zero for the velocity.

:math:`W^c` represents the fluctuating stress tensor (:math:`\tens{Z}` above), which are defined as:

.. math::
   
   W^{n+1/3} &= W_A- \sqrt(3)W_B\\
   W^{n+2/3} &= W_A+ \sqrt(3)W_B\\
   W^{n+1} &= W_A
   
Where :math:`W_A` and :math:`W_B` are uncorrelated Gaussian random 3x3 tensors defined as:

.. math::

   \tens{W} = \sqrt{\frac{2\eta\kT}{h^3 dt}}\widetilde{\tens{W}} + \left(\sqrt{\frac{\xi\kT}{3h^3 dt}} - \frac{1}{3}\sqrt{\frac{2\eta\kT}{h^3dt}}\right)\text{Tr}\left(\widetilde{\tens{W}}\right)\mathbb{I}
   
Where :math:`\widetilde{\tens{W}} = \left(\tens{W}_v + \tens{W}_v^T\right)/\sqrt{2}` is a symmetric 3x3 tensor with

.. math::

   \left\langle \tens{W}_v^{\alpha\beta}(\vec{r}, t)\tens{W}_v^{\gamma\delta}(\vec{r}', t')\right\rangle = \delta_{\alpha\gamma}\delta_{\beta\delta}\delta_{\vec{r}\vec{r}'}\delta_{tt'}


The solver is described in more detail in Appendix A of [1]_. 

Other substepping schemes might be used with slight modifications to this code (see Florencio Balboa's Ph.D manuscript)

The overall algorithm, including the particles (which are included via the :ref:`Immersed Boundary Method`), can be summarized as:
   1. Take particles to mid step: :math:`\vec{q}^{n+1/2} = \vec{q}^n + \frac{dt}{2}\oper{J}^n\vec{v}^n`.
   2. Update the fluid densities and velocities using the Runge Kutta algorithm above to get :math:`\rho^{n+1}, \vec{v}^{n+1}`. Here we use :math:`\vec{f} = \oper{S}^{n+1/2}\vec{F}^{n+1/2}`.
   3. Update particle positions to next step: :math:`\vec{q}^{n+1} = \vec{q}^n + \frac{dt}{2}\oper{J}^{n+1/2}\left(\vec{v}^n+\vec{v}^{n+1}\right)`.

Usage
............

Use as the rest of the :ref:`Integrator` modules.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:  

  * :cpp:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :cpp:`real shearViscosity` Shear viscosity of the solvent.
  * :cpp:`real bulkViscosity` Bulk viscosity of the solvent.
  * :cpp:`real speedOfDound` The isothermal speed of sound is used in the default equation of state.
  * :cpp:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles).
  * :cpp:`real dt`  Time step.
  * :cpp:`real3 boxSize` The domain size.
  * :cpp:`int3 cellDim` Number of fluid cells, if set the hydrodynamicRadius is ignored.
  * :cpp:`uint seed` 0 (default) will take a value from the UAMMD generator
  * :cpp:`std::function<real(real3)> initialDensity`. A function to set the initial density, will be called for each point in the domain
  * :cpp:`std::function<real(real3)> initialVelocityX`. A function to set the initial X velocity, will be called for each point in the domain
  * :cpp:`std::function<real(real3)> initialVelocityY`. A function to set the initial Y velocity, will be called for each point in the domain
  * :cpp:`std::function<real(real3)> initialVelocityZ`. A function to set the initial Z velocity, will be called for each point in the domain
  
.. code:: c++
	  
	#include"Integrator/Hydro/ICM_Compressible.cuh"
	int main(){
	  //...
	  //Assume an instance of ParticleData exists
	  //auto pd = std::make_shared<ParticleData>(numberParticles);
	  //...
	  
	  using namespace ICM = Hydro::ICM_Compressible;
	  ICM::Parameters par;
	  par.shearViscosity = 1.0;
	  par.bulkViscosity = 1.0;
	  par.speedOfSound = 16; //For the equation of state
	  par.temperature = 0;
	  //par.hydrodynamicRadius = 1.0; //Particle hydrodynamic radius (used to determine the number of fluid cells)
	  par.cellDim = {32,32,32}; //Number of fluid cells, if set the hydrodynamicRadius is ignored
	  par.dt = 0.1;
	  par.boxSize = {32,32,32}; //Simulation domain
	  par.seed = 1234; 
	  //The initial fluid density and velocity can be customized:
	  par.initialDensity = [](real3 position){return 1.0;};
	  par.initialVelocityX = [](real3 position){return sin(2*M_PI*position.y);};
	  par.initialVelocityY = [](real3 position){return 1.0;};
	  par.initialVelocityZ = [](real3 position){return 1.0;};
	  
	  auto compressible = std::make_shared<ICM>(pd, par);
	  
	  //Now use it as any other integrator module
	  //compressible->addInteractor...
	  //compressible->forwardTime();
	  //...
	  return 0;
	}

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.

   
FAQ
......

1- I want to fiddle with the boundary conditions:
    -Check the function pbc_cells and fetchScalar in file ICM_Compressible/utils.cuh, which handles what happens when trying to access the information of a cell
    -You can also influence the solver itself (for instance to define special rules for the surfaces of the domain) in the functions of the file FluidSolver.cuh.

2- I want to chenge the spreading kernel:
    -Change the line "using Kernel" below to the type of your kernel. You might also have to change the initialization in the spreading and interpolation functions in ICM_Compressible.cu. You will also have to change the relation between the hydrodynamic radius and the number of fluid cells, do this in the ICM_Compressible constructor.

3- I want to add some special fluid forcing:
    -The function addFluidExternalForcing in ICM_Compressible.cu was created for this.

4- I want to change the equation of state:
    -Check the struct DensityToPressure in ICM_Compressible.cuh.



.. _ICM:

Incompressible Inertial Coupling Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo:: FILL



.. rubric:: References:  

.. [1] Inertial coupling for point particle fluctuating hydrodynamics. F. Balboa et. al. 2013
       
.. [2] STAGGERED SCHEMES FOR FLUCTUATING HYDRODYNAMICS. F. Balboa et. al. 2012
       
.. [3] Ph.D. manuscript. Florencio Balboa.

