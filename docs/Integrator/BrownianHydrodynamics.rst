Brownian Hydrodynamics
==========================

In Brownian Hydrodynamics (BDHI) we reintroduce the hydrodynamic interactions neglected in :ref:`Brownian Dynamics`.

An stochastic differential equation for the time evolution of the positions of a collection of colloidal particles with positions :math:`\vec{q} =\{\vec{q}_1,\vec{q}_2,\dots, \vec{q}_N\}` from the Smoluchowski description of the two-body level marginal probability [1]_ as

.. math::

   d\vec{\ppos} = \tens{M}\vec{F}dt + \sqrt{2\kT\tens{M}}d\vec{\noise} + \kT\vec{\partial}_{\vec{\ppos}}\cdot\tens{M}dt

Where :math:`\vec{F}` are the forces acting on the particles (in UAMMD :ref:`Interactors <Interactor>` are used to provide forces) and the (so-called) Brownian motion, :math:`d\vec{\noise}`, is a collection of Wienner increments with zero mean and

.. math::

   \left\langle d\noise_{i}^\alpha(t)d\noise_{j}^\beta(t') \right\rangle = dt\delta(t-t')\delta_{ij}\delta_{\alpha\beta}.

The pairwise mobility tensor, :math:`\tens{M}`, encodes in its elements the hydrodynamic transfer of the force at one point :math:`\vec{q}_i`, to the displacement at another point :math:`\vec{q}_j`. It thus determines the correlations between two particle displacements.
The mobility tensor is then related to the diffusion coefficients via the Einstein relation,

.. math::

   \tens{D}(\vec{\ppos}) = \kT \tens{M}(\vec{\ppos}).

The Einstein relation coupled with the fluctuation dissipation balance (stating that, in the absence of forces, :math:`\left\langle d\vec{\ppos}\otimes d\vec{\ppos}\right\rangle = 2\tens{D} dt`) hints that mobility tensor should be symmetric and positive semi-definite, so that we can define a tensor :math:`\tens{B}:=\sqrt{\tens{M}}` such that

.. math::
  \tens{B}\tens{B}^T := \tens{M}.

If the system is translationally invariant (isotropic) :math:`\tens{M}(\vec{\ppos}, \vec{\ppos}') = \tens{M}(\vec{\ppos}-\vec{\ppos}')`. To model hydrodynamic couplings, :math:`\tens{M}` is usually taken to be the Rotne-Prager-Yamakawa tensor (describing up to the second hydrodynamic reflection) [1]_. Including further reflections explodes the complexity of the formulation and it is only useful in situations in which particles are really close (e.g. high densities) [1]_, in which case the lubrication approximation is preferable.

.. note:: There are situations in which the system is not isotropic. Such is the case if a wall is present, where the mobility becomes dependent on the distance to the wall, :math:`\tens{M}=\tens{M}(\vec{q}-\vec{q}', z, z')`.

The last term in the BDHI equation of motion, known as the thermal drift term, can be non zero in cases where rigid particle constraints are imposed [2]_ or due to the presence of boundaries [3]_  [4]_. However, in isotropic and irrotational hydrodynamic fields, this term is usually zero and can thus be omitted from the description.

.. hint:: In UAMMD, the :ref:`Quasi 2D` hydrodynamics module is the only solver that includes the thermal drift term explicitly.


One of the usual choices for the mobility tensor is the Rotne-Prager-Yamakawa (RPY) tensor [5]_ , describing the hydrodynamic interaction of two spheres of radius :math:`a`. The free-space RPY mobility tensor can be written in real space as


.. _RPY:

.. math::

  \tens{M}^{\textrm{RPY}}(\vec{r} = \vec{q}_i-\vec{q}_j) = M_0\left\{
  \begin{aligned}
    &\left( \frac{3a}{4r} + \frac{a^3}{2r^3} \right)\mathbb{I} + \left(\frac{3a}{4r} - \frac{3a^3}{2r^3}\right)\frac{\vec{r}\otimes\vec{r}}{r^2}  & r > 2a\\
    &\left(1 - \frac{9r}{32a} \right)\mathbb{I} + \left( \frac{3r}{32a} \right)\frac{\vec{r}\otimes\vec{r}}{r^2} & r \le 2a
  \end{aligned}\right.

The self mobility, :math:`M_0 := \tens{M}(0) = (6\pi\eta a)^{-1}` is, by no coincidence, equal to the drag for a sphere moving through a Stokesian fluid as given by the Stokes-Einstein relation [1]_.

.. note:: Most UAMMD hydrodynamic modules work at the RPY level, although the pseudo-spectral, immersed-boundary-like, algorithms offered by UAMMD (see for instance the :ref:`Force Coupling Method`) regularize the mobility slightly differently in the near field.

.. hint:: The distinction between BD and BDHI in this documentation (and therefore UAMMD) is a bit arbitrary. We call BD to the particular case of BDHI in which we neglect hydrodynamic interactions. This distinction is, however, useful for us due to the much more complex nature of the algorithms employed to solve BDHI (as opposed to the quite simple ones that our so-called BD allows). In the literature, both BDHI and BD are typically referred to as simply "BD".

Given the, in general, long-ranged nature of the hydrodynamic interaction numerical integration of the BDHI equations is not trivial (and quite the field on its own). The main difficulties are related to evaluating the :math:`\tens{M}\vec{F}` product (a :math:`O(N^2)` operation in principle) and the square root of the mobility for the thermal fluctuations (an even more complex operation). Different algorithms exists depending on the boundary conditions and the geometry of the domain. UAMMD offers hydrodynamic solvers (in the form of :ref:`Integrators`) for open, triply periodic and doubly periodic boundary conditions.

It is worth summarizing them here:
 * **Open boundaries**: :ref:`Cholesky`, :ref:`Lanczos`.
 * **Triply periodic**: :ref:`FCM`, :ref:`PSE`, :ref:`FIB`.
 * **Doubly periodic**: :ref:`Quasi2D`, :ref:`DPStokes`.

Let us start with the open boundary solvers.

.. _Cholesky:

Open boundary BDHI solvers
---------------------------

Cholesky
~~~~~~~~~~

The classic strategy for computing the square root of the mobility, originally proposed by Ermak [6]_, is by direct Cholesky factorization. This operation requires :math:`O(N^3)` operations, rendering this algorithm unsuitable for large numbers of particles (above :math:`10^4`). Additionally, it has :math:`O(N^2)` storage requirements, since the full mobility matrix has to be stored.
However, the sheer raw power of the GPU can make this a valid option. In UAMMD the Cholesky factorization is accomplished via a single library call to NVIDIA's cuSolver function `potrf <https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-potrf>`_.
On the other hand, since the mobility matrix has to be stored anyway, the rest of the algorithm can be coded via a few function calls to a linear algebra library. In particular, the Cholesky module uses the matrix-vector multiplications in the `cuBLAS <https://docs.nvidia.com/cuda/cublas/index.html>`_ library. Taking into account the symmetric form of the mobility matrix, only the upper half needs to be computed and stored, cuBLAS (and most linear algebra libraries) provide subroutines that leverage this. In this regard, there is not much possibility for optimization.

Usage
********

In UAMMD, BDHI algorithms are separated between temporal integration schemes and strategies for computing the deterministic and stochastic displacements. Both pieces are joined to form an :ref:`Integrator` that can be used as usual.
Here is an example of the Euler-Maruyama integration scheme being specialized with the Cholesky decomposition algorithm for the fluctuations.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:

  * :code:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :code:`real viscosity` Viscosity of the solvent.
  * :code:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles*)
  * :code:`real dt`  Time step

\* If this parameter is not provided, the module will try to use the particle's radius as the hydrodynamic radius of each particle. In the latter case, if particle radii has not been set in :ref:`ParticleData` prior to the construction of the module an error will be thrown.

.. code:: c++

  #include"uammd.cuh"
  #include<Integrator/BDHI/BDHI_EulerMaruyama.cuh>
  #include<Integrator/BDHI/BDHI_Cholesky.cuh>
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    //A strategy is mixed with an integration scheme
    using BDHI = BDHI::EulerMaruyama<BDHI::Cholesky>;
    BDHI::Parameters par;
    par.temperature = 1.0;
    par.viscosity = 1.0;
    //For Cholesky the radius is optional.
    //If not selected, the module will use the individual
    //  radius of each particle.
    //par.hydrodynamicRadius = 1.0;
    par.dt = 0.01;
    auto bdhi = std::make_shared<BDHI>(sim.pd, par);
    ...
    auto bdhi = make_shared<BDHI>(pd, par);
    ...
    //Add any interactor
    bdhi->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    bdhi->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: Being an open boundary solver, Cholesky does not require a simulation box as a parameter.

.. note:: Cholesky uses a generalized form of the RPY tensor that accounts for differently sized particles, if an hydrodynamic radius is not provided, the radius in :ref:`ParticleData` will be used for each particle.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.

.. _Lanczos:

Lanczos
~~~~~~~~

Fixman proposed a method based on Chebyshev polynomials [7]_ to compute the square root of the mobility. This method requires approximating the extremal eigenvalues of the mobility. Many strategies can be employed to find out these eigenvalues, with complexities ranging from :math:`O(N^3)` (thus beating the purpose) to :math:`O(N^{2.25})`. More recently, a family of iterative algorithms based on Krylov subspace decompositions (using the Lanczos algorithm) have emerged [8]_ showcasing algorithmic complexities in the order :math:`O(kN^2)`, being :math:`k` the number of required iterations (which is usually around the order of :math:`10` depending on the desired tolerance). In \uammd the technique developed in [8]_ is implemented.

.. note:: The Lanczos iterative algorithm for fast computation of :math:`\sqrt{\tens{M}}\vec{v}` (being :math:`\tens{M}` an arbitrary matrix and :math:`\vec{v}` an arbitrary vector) is also available as a separate repository here https://github.com/RaulPPelaez/LanczosAlgorithm

Another benefit of this method over Cholesky is that it is not required to store the full mobility matrix in order to compute the fluctuations. The product of the mobility tensor by a vector (the forces in the deterministic term and a random noise in the fluctuating one) can be computed by recomputing the necessary terms. This will be particularly useful later, when most elements in the mobility tensor become zero, reducing the complexity of the computation for both terms. In particular, UAMMD's implementation of the Lanczos iterative method is templated for any object capable of providing the product of any given vector with the mobility matrix. In the current instance we use the :ref:`NBody` algorithm coupled with a :ref:`Transverser` because the mobility is a dense matrix.


Usage
******

Using the Lanczos strategy in UAMMD is similar to using :ref:`Cholesky`. With the difference that now, being an iterative algorithm, a tolerance can be selected.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:

  * :cpp:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :cpp:`real viscosity` Viscosity of the solvent.
  * :cpp:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles*)
  * :cpp:`real dt`  Time step
  * :cpp:`real tolerance` Tolerance for the Lanczos iterative solver.

\* If this parameter is not provided, the module will try to use the particle's radius as the hydrodynamic radius of each particle. In the latter case, if particle radii has not been set in :ref:`ParticleData` prior to the construction of the module an error will be thrown.


.. code:: c++

  #include"uammd.cuh"
  #include<Integrator/BDHI/BDHI_EulerMaruyama.cuh>
  #include<Integrator/BDHI/BDHI_Cholesky.cuh>
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    //A strategy is mixed with an integration scheme
    using BDHI = BDHI::EulerMaruyama<BDHI::Lanczos>;
    BDHI::Parameters par;
    par.temperature = 1.0;
    par.viscosity = 1.0;
    //For Lanczos the radius is optional.
    //If not selected, the module will use the individual
    //  radius of each particle.
    //par.hydrodynamicRadius = 1.0;
    par.dt = 0.01;
    //The tolerance for the stochastic term computation
    par.tolerance = 1e-3;
    ...
    auto bdhi = make_shared<BDHI>(pd, par);
    ...
    //Add any interactor
    bdhi->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    bdhi->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: Being an open boundary solver, Lanczos does not require a simulation box as a parameter. Additionally, since this is an (approximate) iterative solver, a tolerance is also required.

.. note:: Lanczos uses a generalized form of the RPY tensor that accounts for differently sized particles, if an hydrodynamic radius is not provided, the radius in :ref:`ParticleData` will be used for each particle.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.

Triply periodic BDHI solvers
-----------------------------

UAMMD's triply periodic solvers are based on solving the fluctuating steady Stokes equation for a fluid coupled with a group of particles (as opposed to the BDHI dynamical equation above). We wont go into much detail here, a more in depth description of the mathematical machinery behind these methods is provided, for instance, in [10]_ or [11]_

.. sidebar::

   .. note:: Neglecting convection is valid for small Reynolds number hydrodynamics, i.e, :math:`\text{Re} = \frac{\eta v}{\rho L} \ll 1` with :math:`L` the smallest characteristic length of the system (e.g. particle radius). Moreover, we assume that the Schmidt number is very large, :math:`S_c = \eta/(\rho D_0) \gg 1`, where :math:`\rho`is the fluid density and :math:`D_0 = \kT/(6\pi\eta a)` is the typical diffusion coefficient of a submerged particle, which implies that fluid momentum propagates much faster than particle diffusion. For :math:`S_c\gg 1` the transient term :math:`\rho\partial_t\vec{v}` can be neglected, which is a sane approximation (even for proteins in water).

If we take the overdamped limit of Navier-Stokes equation, where the momentum of the fluid can be eliminated as a fast variable (allowing to neglect the transient term :math:`\partial_t \vec{\fvel}` as well as the convection) we get the so-called Stokes equations

.. math::
    \nabla \pi - \eta \nabla^2\vec{\fvel} &=  \tilde{\vec{f}},\\
    \nabla\cdot\vec{\fvel} &= 0.

Where :math:`\vec{\fvel}(\vec{\fpos}, t)` represents the velocity field of the fluid, :math:`\pi` the pressure and :math:`\eta` its viscosity.

:math:`\tilde{\vec{f}} := \vec{f} + \nabla\cdot\mathcal{Z}` includes the external forces, :math:`\vec{f}`, (some localized force density acting on the fluid (which can arise from the presence of submerged particles)) and the thermal noise, which includes a fluctuating stress tensor, :math:`\mathcal{Z}(\vec{\fpos}, t)`, which must comply with the fluctuation-dissipation balance according to the following statistical properties

.. math::

   &  \langle \mathcal Z_{ij}\rangle = 0\\
   &  \langle \mathcal Z_{ik}(\vec{\fpos},t)\mathcal Z_{jm}(\vec{\fpos}',t')\rangle = 2\kT\eta(\delta_{ij}\delta_{km} + \delta_{im}\delta_{kj})\delta(\vec{\fpos}-\vec{\fpos}')\delta(t-t')

Where :math:`i,j,k,m` represent the different coordinates.

We can eliminate the pressure from the description by using the projection method. Let's take the divergence of the first equation

.. math::
  \eta\nabla^2\vec{\fvel} = \nabla\left[\nabla^{-2}(\nabla\cdot\tilde{\vec{f}})\right] - \tilde{\vec{f}} = -\oper{P} \tilde{\vec{f}}

Where the projection operator, :math:`\oper{P}`, is formally defined as

.. math::
  \oper{P}  :=  \mathbb{I} - \nabla\nabla^{-2}\nabla.

:math:`\oper{P}` projects onto the space of divergence-free velocity. :math:`\mathbb{I}` represents the identity.
In the particular case of an unbounded domain with fluid at rest at infinity, all the differential operators in :math:`\oper{P}` commute in Fourier space, so that

.. math::

   \fou{\oper{P}}(\vec{k}) = \mathbb{I} - \frac{\vec{k}\otimes\vec{k}}{k^2}

Where :math:`\vec{k}` are the wave numbers.
Finally, we can identify

.. math::

   \oper{L} := -\nabla^{-2}\oper{P}


as the Stokes solution operator to arrive at

.. math::

  \vec{\fvel} = \eta^{-1}\oper{L}\tilde{\vec{f}}

The Green's function, :math:`\tens{G}`, of this equation in the case of an unbounded domain can be written in Fourier space as

.. math::

   \eta^{-1}\oper{L}(\vec{k})\rightarrow \fou{\tens{G}}(\vec{k}) := \eta^{-1}k^{-2}\fou{\oper{P}}(\vec{k})

The inverse transform of this Green's function can be computed analytically to get

.. math::

   \tens{O}(\vec{r}) := \frac{1}{8\pi\eta r}\left(\mathbb{I} - \frac{\vec{r}\otimes\vec{r}}{r^2}\right)

This solution is known as the Oseen tensor, the response of a three dimensional unbounded fluid at rest at infinity to a delta forcing.

The Green formalism laid out here constitutes a mechanism to translate forces into velocities in the fluid. In order to couple this with a group of submerged particles we make use of the :ref:`Immersed Boundary Method` (IBM).

The IBM teaches us that we can transform the forces acting on a group of particles into a force density of the fluid by making use of the spreading operator, :math:`\oper{S}`, as

.. math::

  \vec{f}(\vec{\fpos}) = \oper{S}(\vec{\fpos})\vec{F} = \sum_i\delta_a(\vec{\ppos}-\vec{\fpos}_i)\vec{F}_i,

where :math:`\vec{F} := \{\vec{F}_1,\dots,\vec{F}_N\}` are the forces acting on the particles and  :math:`\delta_a(\vec{\fpos})` is a distribution of compact support (usually a smooth smeared delta function, such as a Gaussian).

On the other hand, we can evaluate the velocity of a submerged particle by averaging its local fluid velocity (imposing a no-slip condition so that the particle follows the fluid exactly). We do this via the use of the interpolation operator, :math:`\oper{J} = \oper{S}^*`, as

.. math::

     \vec{\pvel}_i= \oper{J}_{\vec{\ppos}_i}\vec{\fvel} =\int{\delta_a(\vec{\ppos}_i - \vec{\fpos})\vec{\fvel}(\vec{\fpos})d\vec{\fpos}},


where :math:`\vec{\pvel}_i` is the velocity of particle :math:`i`.

Putting it all together, we can write the equation for the particle dynamics as

.. math::

   \frac{d\vec{q}_i}{dt} = \vec{u}_i = \eta^{-1}\oper{J}_{\vec{\ppos}_i}\oper{L}(\oper{S}\vec{F} + \nabla\cdot\mathcal Z).

Which can be shown to be equivalent to the BDHI equations of motion for the particles by defining

.. math::

   \tens{M} = \eta^{-1}\oper{J}\oper{L}\oper{S},

or without the operator notation, the element mobility between particle :math:`i` and :math:`j` as

.. math::

     \tens{M}_{ij} = \eta^{-1}\iint{\delta_a(\vec{q}_j-\vec{r})\oper{L}(\vec{r}, \vec{r}')\delta_a(\vec{q}_i -\vec{r}')d\vec{r}d\vec{r}'}.

.. note:: The :ref:`RPY` tensor arises from evaluating the double convolution of the Oseen tensor with a delta function integrated over the surface of two spheres centered at :math:`\vec{q}_i` and :math:`\vec{q}_j`.


Finally, its "square root" can be defined as


.. math::

   \tens{M}^{1/2} = \eta^{-1/2}\oper{J}\oper{L}\nabla\cdot.



.. _FCM:

Force Coupling Method
~~~~~~~~~~~~~~~~~~~~~~

The :ref:`FCM` [9]_ is an Immersed-Boundary-like Eulerian-Lagrangian pseudo-spectral method initially devised for the computation of the hydrodynamic displacements of a colloidal suspension in a triply periodic environment. The FCM framework makes use of the Green formalism laid out in the previous section and constitutes the basis of all the other hydrodynamic modules in UAMMD.

.. hint:: The FCM is so generic as a Green formalism solver that UAMMD uses it also for electrostatics (see :ref:`SpectralEwaldPoisson`).

UAMMD's open boundary hydrodynamic methods use the explicit form of an open boundary mobility (the :ref:`RPY` one), not taking into account the periodic images of the system in any way. Furthermore, the computational complexity of these methods is restrictive. Luckily, we can manage to do it in :math:`O(N)` operations if we consider periodic boundary conditions. In particular by solving the Stokes equation equation directly in Fourier space via the Force Coupling Method [9]_. In doing so, we get the added benefit (and disadvantage) of not imposing a specific mobility tensor, which will arise naturally according to the convolution between the Green's function and the spreading kernel.


We discretize and solve the velocity of the fluid on a regular grid with size :math:`h` (which can vary in each direction), with a number of cells in each size :math:`N_c = L/h`.
We use the FFT to discretize the Fourier transform, this requires us to evaluate the properties of the fluid in a grid. This grid must be fine enough to correctly describe the smeared delta function (in the spreading and interpolation operators, a Gaussian in the case of the original FCM). On the other hand, the Gaussian kernel has an infinite range and to make the overall spreading/interpolation have a constant cost for each particle (independent of the size of the domain) it is necessary to truncate it at a certain distance, :math:`r_c`. UAMMD automatically chooses these parameters to ensure errors stay below a certain provided tolerance.

Without going into much detail, the FCM can be summarized into the following steps:
  * Spread particle forces to the grid: :math:`\vec{f} = \oper{S}\vec{F}`
  * Transform fluid forcing to Fourier space: :math:`\fou{\vec{f}} = \mathfrak{F}\oper{S}\vec{F}`
  * Multiply by the Green's function to get :math:`\eta^{-1}\fou{\tens{G}}\mathfrak{F}\oper{S}\vec{F}`
  * Sum the stochastic forcing in Fourier space: :math:`\fou{\vec{\fvel}} = \eta^{-1}\fou{\tens{G}}(\mathfrak{F}\oper{S}\vec{F} + \vec{k}\fou{\mathcal{Z}})`
  * Transform back to real space: :math:`\vec{\fvel} = \eta^{-1}\mathfrak{F}^{-1}\fou{\tens{G}}(\mathfrak{F}\oper{S}\vec{F} + \vec{k}\fou{\mathcal{Z}})`
  * Interpolate grid velocities to particle positions: :math:`\vec{\pvel} = \oper{J}\vec{\fvel}`

Here :math:`\mathfrak{F}` represents the Fourier transform operator.

Once the particle velocities are computed, the dynamics can be integrated using, for instance, the Euler-Maruyama scheme devised for :ref:`BD`. The update rule in the case of Euler-Maruyama is

.. math::

  \vec{\ppos}^{n+1} = \vec{\ppos}^n + \vec{\pvel}^n\dt,

where the particle velocities already include the stochastic displacements.



Usage
*******

Use as the rest of the :ref:`Integrator` modules.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:

  * :cpp:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas. Can be 0.
  * :cpp:`real viscosity` Viscosity of the solvent.
  * :cpp:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles)
  * :cpp:`int3 cells`  Number of grid cells in each direction. This parameter can be set instead of the hydrodynamic radius and will force FCM to construct the grid of this size.
  * :cpp:`real dt`  Time step
  * :cpp:`real tolerance` Overall tolerance of the solver (affects the grid size and kernel support).
  * :cpp:`Box box` A :cpp:class:`Box` with the domain size information.


.. code:: c++

  #include"uammd.cuh"
  #include<Integrator/BDHI/BDHI_EulerMaruyama.cuh>
  #include<Integrator/BDHI/BDHI_FCM.cuh>
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    //A strategy is mixed with an integration scheme
    using FCM = BDHI::EulerMaruyama<BDHI::FCM>;
    FCM::Parameters par;
    par.temperature = 1.0;
    par.viscosity = 1.0;
    par.hydrodynamicRadius = 1.0;
    //par.cells = {128, 128, 128}; //You can specify the grid size instead.
    par.dt = 0.01;
    par.tolerance = 1e-3;
    par.box = Box({128, 128, 128});
    auto bdhi = std::make_shared<FCM>(pd, par);
    ...
    //Add any interactor
    bdhi->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    bdhi->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: Being a triply periodic solver, FCM requires a simulation box as a parameter. Additionally, since this is a non-exact solver (with spatial discretization errors), a tolerance is also required.

.. warning:: Contrary to the open boundary methods, in FCM all particles must have the same hydrodynamic radius.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.

.. hint:: Although by default the Gaussian kernel is used for spreading/interpolation, the code includes a lot of other alternatives (such as Peskin kernels) that can be selected in the source file "Integrator/BDHI/BDHI_FCM.cuh"

.. note:: Although this is undocumented at the moment, the FCM module can also deal with torques/angular displacements.


.. _PSE:

Positively Split Ewald
~~~~~~~~~~~~~~~~~~~~~~~

An Ewald split version of the Force Coupling Method [12]_. Splits the computation between a far and near field contributions. The far field reuses the FCM machinery described above and the near field makes use of the Lanczos algorithm.


.. todo:: Fill

Usage
******

Use as the rest of the :ref:`Integrator` modules.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:

  * :cpp:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :cpp:`real viscosity` Viscosity of the solvent.
  * :cpp:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles)
  * :cpp:`real dt`  Time step
  * :cpp:`real tolerance` Overall tolerance of the algorithm (FCM in the far field and Lanczos iterative solver in the near field).
  * :cpp:`Box box` A :cpp:class:`Box` with the domain size information.
  * :cpp:`real split` The splitting parameter of the PSE algorithm in units of inverse of length. This parameter only affects performance and must be manually tuned in a case by case basis to find the optimal (usually between 0.1/hydrodynamicRadius-1/hydrodynamicRadius).

.. code:: c++

  #include"uammd.cuh"
  #include<Integrator/BDHI/BDHI_EulerMaruyama.cuh>
  #include<Integrator/BDHI/BDHI_PSE.cuh>
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    //A strategy is mixed with an integration scheme
    using PSE = BDHI::EulerMaruyama<BDHI::PSE>;
    PSE::Parameters par;
    par.temperature = 1.0;
    par.viscosity = 1.0;
    par.hydrodynamicRadius = 1.0;
    par.dt = 0.01;
    par.tolerance = 1e-3;
    par.split = 0.5;
    par.box = Box({128, 128, 128});
    auto bdhi = std::make_shared<PSE>(pd, par);
    ...
    //Add any interactor
    bdhi->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    bdhi->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: Being a triply periodic solver, PSE requires a simulation box as a parameter. Additionally, since this is a non-exact solver (with spatial discretization errors), a tolerance is also required.

.. warning:: Contrary to the open boundary methods, in PSE all particles must have the same hydrodynamic radius.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.


.. _FIB:

Fluctuating Immersed Boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identical to :ref:`FCM`, but using a :ref:`staggered grid<Staggered grid>` [10]_.

.. todo:: fill

Usage
*******


Use as the rest of the :ref:`Integrator` modules.

.. sidebar::

   .. warning:: Note that the temperature is provided in units of energy.

The following parameters are available:

  * :cpp:`real temperature` Temperature of the solvent in units of energy. This is :math:`\kT` in the formulas.
  * :cpp:`real viscosity` Viscosity of the solvent.
  * :cpp:`real hydrodynamicRadius` Hydrodynamic radius of the particles (same for all particles)
  * :cpp:`real dt`  Time step
  * :cpp:`real tolerance` Overall tolerance of the solver (affects the grid size and kernel support).
  * :cpp:`Box box` A :cpp:class:`Box` with the domain size information.


.. code:: c++

  #include"uammd.cuh"
  #include<Integrator/BDHI/BDHI_EulerMaruyama.cuh>
  #include<Integrator/BDHI/BDHI_FIB.cuh>
  using namespace uammd;
  int main(){
    //Assume an instance of ParticleData, called "pd", is available
    ...
    //A strategy is mixed with an integration scheme
    using FIB = BDHI::EulerMaruyama<BDHI::FIB>;
    FIB::Parameters par;
    par.temperature = 1.0;
    par.viscosity = 1.0;
    par.hydrodynamicRadius = 1.0;
    par.dt = 0.01;
    par.tolerance = 1e-3;
    par.box = Box({128, 128, 128});
    auto bdhi = std::make_shared<FIB>(pd, par);
    ...
    //Add any interactor
    bdhi->addInteractor(myInteractor);
    ...
    //Take simulation to the next step
    bdhi->forwardTime();
    ...
    return 0;
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: Being a triply periodic solver, FIB requires a simulation box as a parameter. Additionally, since this is a non-exact solver (with spatial discretization errors), a tolerance is also required.

.. warning:: Contrary to the open boundary methods, in FIB all particles must have the same hydrodynamic radius.

.. note:: As usual, any :ref:`Interactor` can be added to this :ref:`Integrator`, as long as it is able to compute forces.



Doubly periodic BDHI solvers
------------------------------


.. _Quasi2D:

Quasi two-dimensional
~~~~~~~~~~~~~~~~~~~~~~~


.. todo:: fill

Usage
********

.. todo:: fill


.. _DPStokes:

Doubly Periodic Stokes (DPStokes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo:: fill

Usage
********

.. todo:: fill



.. rubric:: References:

.. [1] An Introduction to Dynamics of Colloids. Dhont, J.K.G. 1996. https://www.elsevier.com/books/an-introduction-to-dynamics-of-colloids/dhont/978-0-444-82009-9

.. [2] A generalised drift-correcting time integration scheme for Brownian suspensions of rigid particles with arbitrary shape. Timothy A Westwood and Blaise Delmotte and Eric E Keaveny 2021.

.. [3] Hydrodynamic fluctuations in quasi-two dimensional diffusion. Raul P. Pelaez et al. 2018. https://doi.org/10.1088/1742-5468/aac2fb

.. [4] Hydrodynamics of Suspensions of Passive and Active Rigid Particles: A Rigid Multiblob Approach. Florencio Balboa et. al 2016. https://doi.org/10.2140/camcos.2016.11.217

.. [5] Variational Treatment of Hydrodynamic Interaction in Polymers. Rotne,Jens  and Prager,Stephen 1969. https://doi.org/10.1063/1.1670977

.. [6] Brownian dynamics with hydrodynamic interactions. Ermak,Donald L.  and McCammon,J. A. 1978. https://doi.org/10.1063/1.436761

.. [7] Construction of Langevin forces in the simulation of hydrodynamic interaction. Fixman, Marshall 1986. https://doi.org/10.1021/ma00158a043

.. [8] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations. Ando,Tadashi et. al. 2012.   https://doi.org/10.1063/1.4742347

.. [9] Fluctuating force-coupling method for simulations of colloidal suspensions. Keaveny 2014. https://doi.org/10.1016/j.jcp.2014.03.013

.. [10] Brownian dynamics without Green's functions.  Delong et. al. 2014. https://doi.org/10.1063/1.4869866

.. [11] Complex fluids in the GPU era. Raul P. Pelaez tesis manuscript 2022. https://github.com/RaulPPelaez/tesis/raw/main/manuscript.pdf

.. [12] Rapid sampling of stochastic displacements in Brownian dynamics simulations. Fiore et. al. 2017. https://doi.org/10.1063/1.4978242
