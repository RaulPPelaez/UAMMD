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

    :math:`\vec{g}=\rho\vec{v}`: Fluid momentum.

    :math:`\nabla\cdot \tens{\sigma} = \nabla\pi - \eta\nabla^2\vec{v} - (\xi+\eta/3)\nabla(\nabla\cdot\vec{v})`: Deterministic stress tensor

    :math:`\pi`: Pressure.

    :math:`\vec{f}`: Any external fluid forcing, typically :math:`\oper{S}\vec{F}`, the spreaded particle forces.

    :math:`\vec{g}\otimes\vec{v}`: I call this the kinetic tensor.

    :math:`\tens{Z}`: Fluctuating stress tensor.


Depending on the particular solver, some terms might be zero. For instance, if the solver is incompressible, the term :math:`\nabla\cdot\vec{v}` vanishes and the density is constant.

When inertia is disregarded the equations fall back to the Stokes equations, UAMMD offers solvers for this regime, referred to as :ref:`Brownian Hydrodynamics`. Not that in any case we disregard particle inertia (i.e the particles follow the local fluid exactly).

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
(with components :math:`v^\alpha`) and a tensor field with :math:`\bm{\Sigma}` (with components
:math:`\Sigma^{\alpha\beta}` ).

Say :math:`i` represents a cell in the grid, which is centered at
the position :math:`\vec{r}_i`. Then, the different fields, corresponding to cell
:math:`i` would be defined at the following locations:

  - :math:`\rho_{i} \rightarrow \vec{r}_{\vec{i}}`
  - :math:`\vec{v}^\alpha_{i+\alpha/2} \rightarrow \vec{r}_{i} + h/2\vec{\alpha}`
  - :math:`\bm{\Sigma}^{\alpha\beta}_{i+\alpha/2 + \beta/2} \rightarrow \vec{r}_{i} + h/2\vec{\alpha} + h/2\vec{\beta}`

Where :math:`\vec{\alpha}` and :math:`\vec{\beta}` are the unit vectors in those directions and :math:`h` is the size of a cell. From now on a superindex denotes a coordinate and a subindex denotes a physical position in space. To be more explicit the component :math:`x` of the velocity, :math:`v^x_ {i+x/2}` of cell :math:`i` is defined at the position :math:`\vec{r}_i + \hat{\vec{x}}h/2`, which is to be understood as "the location of the center of cell :math:`i` plus :math:`h/2` in the :math:`x` direction".

These rules result in the values assigned to a cell sometimes being defined in
strange places. The sketch below represents all the values owning to a certain
cell, :math:`i` (with center defined at ○). Unintuitively, some quantities assigned
to cell :math:`i` lie in the neighbouring cells (represented below is also the cell to its right).

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
  * ○: :math:`\rho_i` (Cell center, at :math:`\vec{r}_{i}`)
  * ◨: :math:`v^x_{i+x/2}`
  * ⬒: :math:`v^y_{i+y/2}`
  * △: :math:`\bm{\Sigma}^{xx}_{i + x}`
  * ▽: :math:`\bm{\Sigma}^{xy}_{i + x/2 + y/2},\bm{\Sigma}^{yx}_{i + x/2 + y/2}`


Naturally, this discretisation requires special handling of the discretized versions of the (differential) operators. See for instance :code:`ICM_Compressible/SpatialDiscretization.cuh` to see how UAMMD deals with them.

For instance, multiplying a scalar and a vector requires interpolating the
scalar at the position of the vector (Since the result, being a vector, must be
defined at the vector subgrids). One example of this is computing the momentum:

:math:`g^\alpha_{i+alpha/2} = \rho_{i+alpha/2}v^\alpha_{i+alpha/2} = \frac{1}{2}(\rho_i + \rho_{i+1})v^\alpha_{i+alpha/2}`

The differential operators are discretized via finite differences.
For instance, the gradient of an scalar is a vector. Lets say we want to compute the :math:`\alpha` component of the density gradient at cell :math:`i`:

.. math::

  (\nabla\rho)_{i+\alpha/2}^\alpha := \partial_\alpha\rho_i = \frac{1}{h}(\rho_{i+\alpha} - \rho_i)

The result is defined at the location :math:`i + x/2`. The rest of the operators follow a similar pattern, where in order to map from one space to another (like when going from scalars to vector in the above example) we have to make sure that the result is defined at the right location.


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
  2. :math:`a=b+3/4`, :math:`b=n+1/4` and :math:`c=n+2/3`
  3. :math:`a=b+1/3`, :math:`b=n+2/3` and :math:`c=n+1`

The values of :math:`A` and :math:`B` allow to choose between different temporal discretizations.

The current implementation uses, for each subtime respectively:
  1.  :math:`A=0, B=1`
  2.  :math:`A=3/4, B=1/4`
  3.  :math:`A=1/3, B=2/3`

In both cases, we can define :math:`\Delta U = -dt\nabla\cdot\tens{F} + dt\vec{f}`.

Where :math:`\tens{F}(U,W,t)` means one thing or another depending on the equation we are solving. :math:`\vec{f}` is only non-zero for the velocity.

:math:`W^c` represents the fluctuating stress tensor (:math:`\tens{Z}` above), which are defined as:

.. math::

   W^{n+1/3} &= W_A- \sqrt{3}W_B\\
   W^{n+2/3} &= W_A+ \sqrt{3}W_B\\
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

Boundary conditions via ghost cells
.......................................

The boundary conditions are implemented using ghost cells. Since none of the operators require searching for a value beyond first neighbours we can use a single layer of ghost cells.

.. figure:: ../img/ghostcells.*
	    :width: 50%
	    :align: center
	    :alt: A representation of the ghost cell layer.

	    A fluid discretized at the white cells is surrounded by a single layer of ghost cells (green). In order to apply periodic boundary conditions we must carefully fill the ghost cells. For instance, in order for the cell :math:`(0,1)` to access the information of the cell located to its left it is necessary for the ghost cell located there to store the information of the cell :math:`(2,1)`. Similarly the ghost cell at the top left corner must store the information in the cell :math:`(2,2)`.



.. note:: In order to ensure a single layer is enough we store the density, :math:`\rho`, the fluid velocity :math:`\vec{v}` and the momentum, :math:`\vec{g}=\rho\vec{v}`, across the whole domain. While storing the momentum everywhere is redundant doing this simplifies the implementation and facilitates the customization of the boundary conditions via the ghost cells.


It is useful to lay out a situation in which not storing the momentum explicitly can be problematic when using a single layer of ghost cells.
In particular, there is an issue when trying to evaluate the divergence of the kinetic tensor,

.. math::

   \nabla\cdot\tens{K} := \nabla\cdot(\vec{g} \otimes \vec{v}),

at the border of the domain. Tensor divergence is defined elementwise (such that the result is a vector) as

.. math::

  \left(\nabla\cdot \tens{K}\right)^\alpha_{i+\alpha/2} = \left(\sum_\beta \partial_\beta \tens{K}^{\alpha\beta}_{i+\alpha/2 + \beta/2}\right)_{i+\alpha/2}.

For instance, the :math:`x` component will be defined at the same position as the velocity :math:`v^x_{i+x/2}` in the staggered grid sketch, that is :math:`\vec{r}_i + \hat{\vec{x}}h/2`.
Lets focus on the :math:`\beta = \alpha` component:

.. math::

   \left(\partial_\beta \tens{K}^{\beta\beta}_{i+\beta}\right)_{i+\beta/2} = 1/h (\tens{K}^{\beta\beta}_{i+\beta} - \tens{K}^{\beta\beta}_{i})

And now let us evaluate

.. math::

   \tens{K}^{\beta\beta}_{i+\beta} = \frac{1}{2}(g^\beta_{i+\beta/2} + g^\beta_{i+3/2\beta})\frac{1}{2}(v^\beta_{i+\beta/2} + v^\beta_{i+3/2\beta})

In order to compute one element of a tensor (comping from two vectors) we interpolate the components at the same place. In this case the location :math:`i+\beta` (where :math:`\rho_{i+x}` is located in the staggered grid when :math:`\beta=x`).

Finally, let us compute :math:`g^\beta_{i+3/2\beta}`:

.. math::

   g^\beta_{i+3/2\beta} = \rho_{i+3/2\beta}v^\beta_{i+3/2\beta}=\frac{1}{2}(\rho_{i+2\beta} + \rho_{i+\beta})v^\beta_{i+3/2\beta}

Think about the rightmost cell of the domain, if we do not store the momentum the algorithm will try to fetch :math:`\rho_{i+2\beta}`, which lies one cell to the right of the ghost layer and is of course invalid. One solution is to store the momentum separately, so the element :math:`g^\beta_{i+3/2\beta}` is accessible in a ghost cell.


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

.. _ICMWalls:

Walls
......

By default the compressible ICM Integrator is triply periodic. The class :code:`ICM_Compressible_impl` has a template parameter called Walls that can be customized to handle the ghost layers in the Z direction. In particular allowing to place walls at the domain limits.

.. note:: The :ref:`Integrator` called :code:`ICM_Compressible` is an alias to :code:`ICM_Compressible_impl<Hydro::icm_compressible::DefaultWalls>`.

The wall-handling class must abide to the following rules:

.. cpp:class:: ICMZWalls

   A class with arbitrary name that will be used by :code:`ICM_Compressible_impl` to handle the ghost cells in the Z direction.
   Note that this class can be :ref:`ParameterUpdatable` (for instance to model a moving wall).

   .. cpp:function:: ICMZWalls()

      Must provide a default constructor due to a technical restriction.

   .. cpp:function:: __host__ __device__ bool isEnabled();

      If the domain is periodic in Z, this must return false. Otherwise return true.

   .. cpp:function:: __device__ void applyBoundaryConditionZBottom(FluidPointers fluid, int3 ghostCell, int3 n);

      This function applies the boundary conditions at the bottom z wall for the fluid for a given ghost cell.

   .. cpp:function:: __device__ void applyBoundaryConditionZTop(FluidPointers fluid, int3 ghostCell, int3 n);

      This function applies the boundary conditions at the top z wall for the fluid for a given ghost cell.


For an usage example you can check the file :code:`test/Hydro/ICM_Compressible/walltest.cu`, which encodes a moving wall at the bottom.

The default wall class in :code:`ICM_Compressible.cuh` is not enabled and can serve as an example as well.

An additional parameter is present in :code:`ICM_Compressible_impl` to allow to provide an instance of the Wall handler:
  * :cpp:`std::shared_ptr<Walls> walls`. If present, this object will be used to handle the ghost layers in the Z direction.

FAQ
......

1- I want to fiddle with the boundary conditions:
    -Check the file :code:`ICM_Compressible/GhostCells.cuh`, which handles the filling of the ghost layer. You might also want to check the :code:`ICM_Compressible/Fluctuations.cuh`, which among other things handles the ghost layer for the fluctuations. Finally, if particles are involved, you will probably need to modify the spreading kernel (see below).
    
    -Walls can be placed in the Z direction and customized via a template parameter (or by modifying the default wall class in :code:`ICM_Compressible.cuh`). See :ref:`Walls<ICMWalls>` above.
    
    -You can also influence the solver itself (for instance to define special rules for the surfaces of the domain) in the functions of the file :code:`ICM_Compressible/FluidSolver.cuh`.

2- I want to change the spreading kernel:
    -Change the line "using Kernel" below to the type of your kernel. You might also have to change the initialization in the spreading and interpolation functions in ICM_Compressible.cu. You will also have to change the relation between the hydrodynamic radius and the number of fluid cells, do this in the ICM_Compressible constructor.

3- I want to add some special fluid forcing:
    -The function addFluidExternalForcing in ICM_Compressible.cu was created for this.

4- I want to change the equation of state:
    -Check the struct DensityToPressure in ICM_Compressible.cuh.



.. _ICM:

Incompressible Inertial Coupling Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In the incompressible scheme density is constant and the divergence of the velocity is null, simplifying the equations to

.. math::

   \rho\partial_t{\vec{\fvel}} +\rho\nabla\cdot (\vec{\fvel}\otimes\vec{\fvel})  + \nabla \pi &= \eta \nabla^2\vec{\fvel} + \vec{f} + \nabla\cdot \mathcal{Z}\\
    \nabla\cdot\vec{\fvel} &= 0


This scheme uses the same staggered grid spatial discretization as the compressible scheme and solves the equations in a triply periodic environment.

We can rewrite the incompressible Navier-Stokes equation above as

.. math::

  \dot{\vec{\fvel}} = \rho^{-1} \oper{P}\left(\vec{\mathfrak{f}} + \tilde{\vec{f}}\right) = \rho^{-1} \oper{P}\vec{f}^*.

Where we have introduced a new fluid forcing,

.. math::

  \vec{\mathfrak{f}} = -\rho\nabla\cdot (\vec{\fvel}\otimes\vec{\fvel}) + \eta\nabla^2\vec{\fvel},

that includes the advective and diffusive terms to simplify the notation.

The projection operator, :math:`\oper{P}`, is formally defined as

.. math::

  \oper{P}  :=  \mathbb{I} - \nabla\nabla^{-2}\nabla.

Finally, the external fluid forcing :math:`\tilde{\vec{f}}` is defined as

.. math::

   \tilde{\vec{f}} = \vec{f} + \nabla\cdot\tens{Z}


We apply the projection operator in Fourier space, as we did in, for instance the :ref:`FCM`. Since we now have to solve the temporal variation of the velocity and we have non-linear terms, the diffusive and advective terms will be evaluated in real space. In the ICM, the divergence of the noise is also evaluated in real space.

We use a second-order accurate predictor-corrector scheme for temporal discretization. We can discretize the coupled fluid-particle equations as

.. math::

    &\vec{\ppos}^{n+\half} = \vec{\ppos}^n + \frac{\dt}{2}\oper{J}^n\vec{\fvel}^n,\\
    &\rho\frac{\vec{\fvel}^{n+1} - \vec{\fvel}^n}{\dt} = \oper{P}\left(\vec{\mathfrak{f}}^{n+\half} + \tilde{\vec{f}}^{n+\half} \right),\\
    &\vec{\ppos}^{n+1} = \vec{\ppos}^n + \frac{\dt}{2}\oper{J}^{n+\half}\left(\vec{\fvel}^{n+1} + \vec{\fvel}^{n}\right).

Which requires evaluating the non-linear fluid forcing terms at mid step (i.e advection and diffusion).
The convective term is discretized using a second order explicit Adams-Bashforth method (Eq. 35 in [4]_ ),

.. math::

  \nabla\cdot (\vec{\fvel}\otimes\vec{\fvel})^{n+\half} = \frac{3}{2} \nabla\cdot (\vec{\fvel}\otimes\vec{\fvel})^n - \half \nabla\cdot (\vec{\fvel}\otimes\vec{\fvel})^{n-1}.

Advection is therefore stored each step to be reused in the next.
The diffusive term is similarly discretized to second-order by

.. math::

  \nabla^2\vec{\fvel}^{n+\half} = \half\nabla^2\left(\vec{\fvel}^{n+1} + \vec{\fvel}^{n}\right).

Replacing both equations and solving for the velocity at time :math:`n+1` leads to the full form of the velocity solve, depending only on the velocity from previous time steps

.. math::

    &\vec{\fvel}^{n+1} = \tilde{\oper{P}}\vec{g}^n =\tilde{\oper{P}}\Big[    \left(\frac{\rho}{\dt}\mathbb{I} + \frac{\eta}{2}\nabla^2\right)\vec{\fvel}^n- \\
    & \frac{3\dt}{2} \nabla\cdot (\vec{\fvel}\otimes\vec{\fvel})^n - \frac{\dt}{2} \nabla\cdot (\vec{\fvel}\otimes\vec{\fvel})^{n-1}+\\
    &\oper{S}\vec{F}^{n+\half} + \nabla\cdot\mathcal{Z}^n \Big],

where the modified projection operator is defined as

.. math::

  \tilde{\oper{P}} :=\left(\frac{\rho}{\dt}\mathbb{I} - \frac{\eta}{2}\nabla^2\right)^{-1}\oper{P}

and is applied in Fourier space.

The full algorithm can be summarized as follows:
  * Take particle positions to time :math:`n+\half`: :math:`\vec{\ppos}^{n+\half} = \vec{\ppos}^n + \frac{\dt}{2}\oper{J}^n\vec{\fvel}^n`.
  * Spread forces on particles to the staggered grid: :math:`\oper{S}\vec{F}^{n+\half}`.
  * Compute and store advection: :math:`\nabla\cdot (\vec{\fvel}\otimes\vec{\fvel})^n`.
  * Compute the rest of the terms in :math:`\vec{f}^*`, using the advective term just computed in addition to the one stored in the previous step.
  * Take :math:`\vec{f}^*` to Fourier space and apply :math:`\tilde{\oper{P}}`: :math:`\fou{\vec{\fvel}}^{n+1} = \fou{\tilde{\oper{P}}}\fou{\vec{f}^*}`.
  * Take :math:`\fou{\vec{\fvel}}^{n+1}` back to real space.
  * Evaluate particle positions at :math:`n+1` by interpolating: :math:`\vec{\ppos}^{n+1} = \vec{\ppos}^n + \frac{\dt}{2}\oper{J}^{n+\half}\left(\vec{\fvel}^{n+1} + \vec{\fvel}^{n}\right)`.

We use the discrete form of the differential operators for a staggered grid (see :ref:`Staggered grid`).



Usage
.......

Usage of the ICM :ref:`Integrator` requires a list of the familiar parameters for hydrodynamics thus far plus the fluid density, which is constant.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:

  * :cpp:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :cpp:`real viscosity` Shear viscosity of the solvent.
  * :cpp:`real density` Density of the solvent.
  * :cpp:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles).
  * :cpp:`real dt`  Time step.
  * :cpp:`Box box` The domain size.
  * :cpp:`int3 cells` Number of fluid cells, if set the hydrodynamicRadius is ignored.
  * :cpp:`uint seed` 0 (default) will take a value from the UAMMD generator
  * :cpp:`bool sumThermalDrift = false` Thermal drift has a neglegible contribution in ICM (and formally null), but can still be computed via random finite differences if desired.
  * :cpp:`bool removeTotalMomentum = true` Set the total fluid momentum to zero in each step


.. code:: cpp

  #include<uammd.cuh>
  #include<Integrator/Hydro/ICM.cuh>
  int main(){
    //...
    //Assume an instance of ParticleData exists
    //auto pd = std::make_shared<ParticleData>(numberParticles);
    //...
    Hydro::ICM::Parameters par;
    par.temperature = 1.0;
    par.viscosity = 1.0;
    par.density = 1.0;
    par.hydrodynamicRadius = 1;
    par.dt = 0.01;
    par.box = Box({32, 32, 32});
    auto icm = std::make_shared<Hydro::ICM>(pd, par);
    //Now use it as any other integrator module
    //icm->addInteractor...
    //icm->forwardTime();
    //...
    return 0;
  }


Here, :code:`pd` is a :ref:`ParticleData` instance.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.


.. rubric:: References:

.. [1] Inertial coupling for point particle fluctuating hydrodynamics. F. Balboa et. al. 2013

.. [2] STAGGERED SCHEMES FOR FLUCTUATING HYDRODYNAMICS. F. Balboa et. al. 2012

.. [3] Ph.D. manuscript. Florencio Balboa.

.. [4] Inertial coupling method for particles in an incompressible fluctuating fluid. F. Balboa et. al. 2014.
