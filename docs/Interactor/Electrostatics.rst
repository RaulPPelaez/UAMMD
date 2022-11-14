.. _SpectralEwaldPoisson:

Electrostatics
====================

UAMMD implements fast spectral solvers for the Poisson equation with Gaussian sources of charge with arbitrary widths centered at the particles' locations. Two solvers are available with triply or doubly periodic boundary conditions.

In general, we want to solve the Poisson equation in a periodic domain in the presence of a charge density :math:`f(\vec{\fpos}=(x,y,z))`,

.. math::
   
 \varepsilon\Delta\phi=-f.
 
Here :math:`\varepsilon` represents the permittivity of the medium and :math:`f` accounts for :math:`N` Gaussian charges of strength :math:`Q_i` located at :math:`\vec{\ppos}_i`,

.. math::
   
  f(\vec{\fpos})= \oper{S}(\vec{\fpos})\vec{Q} = \sum_iQ_i\delta_a(||\vec{\fpos}-\vec{\ppos}_i||).

Let us denote with the vector containing all charges as :math:`\vec{Q} = \{Q_1,\dots,Q_N\}`.
We will use the spreading operator, :math:`\oper{S}`, (see :ref:`IBM`) to transform the particles charges to a smooth charge density field. We use the Gaussian kernel,

.. math::
   
  \delta_a(\vec{r})=\frac{1}{\left(2\pi a^2\right)^{3/2}}\exp{\left(\frac{-r^2}{2a^2}\right)},

Identifying :math:`a` as the width the charges (notice that the case :math:`a\rightarrow 0` corresponds to point charges).
Once the Poisson equation is solved we have the value of the potential in every point in space and can be evaluated at the charges locations via the interpolation operator (see :ref:`IBM`).

.. math::
   
  \phi_{\vec{\ppos}_i} = \oper{J}_{\vec{\ppos}_i}\phi = \int Q_i\delta_a(\vec{\ppos}_i - \vec{\fpos})\phi(\vec{\fpos})d\vec{\fpos}.

Here :math:`\oper{J}` represents the interpolation operator, that averages a quantity defined in space to a charge's location.
The electrostatic energy can be computed as

.. math::
   
  U =  \frac{1}{2}\sum_i{\phi_{\vec{\ppos}_i}}.


In a similar way we compute the electrostatic force, :math:`\vec{F}_i = -Q_i\nabla_i{\phi}`, acting on each charge from the electric field

.. math::
   
  \vec{E} = -\nabla{\phi}.

Interpolating again we get

.. math::

   \vec{E}_i = \oper{J}_{\vec{\ppos}_i}\vec{E}(\vec{\fpos}).

Thus, the electrostatic force acting on particle :math:`i` is

.. math::
   
   \vec{F}_i = Q_i\oper{J}_{\vec{\ppos}_i}\vec{E}.


Given that the Gaussian kernel has, in principle, an infinite support evaluating the charge density at every point in space, as well as computing the averages of the electric potential and field at the particles' locations can be highly inefficient. In practice we overcome this limitation by truncating the Gaussian kernel at a certain distance according to a desired tolerance.


Triply periodic boundary conditions
--------------------------------------

In this instance we solve the Poisson equation with in a periodic boundary (in the three dimensions).
Our approach is similar to the one presented at [1]_. However, the authors of [1]_ make use of the so-called Fast Gaussian Gridding (FGG) to accelerate spreading/interpolation while UAMMD's its :ref:`Immersed Boundary framework <IBM>`. The electrostatic modules in UAMMD make use of Ewald splitting framework following closely from the mathematical machinery laid out in [2]_.
One notable thing about the following algorithm is that it can be seen as merely a reinterpretation of terms in the non-fluctuating version of the :ref:`FCM` for hydrodynamics.

We solve the Poisson equation in Fourier space by convolution with the Poisson's Greens function,

.. math::
   
 \hat\phi(\vec{k}) = \frac{\hat f(\vec{k})}{\varepsilon k^2}.

We can reuse the methodological machinery devised for the :ref:`FCM`.

The electric field can be derived from the potential in fourier space via *ik* differentiation,

.. math::
   
  \hat{\vec{E}} = i\vec{k}\hat{\phi}.


This equation can be discretized using a 3D FFT in a grid with spacing fine enough to resolve the Gaussian charges.

The whole algorithm, going from particle charges to forces, can be summarized as follows
 * Spread charges to the grid, :math:`f=\oper{S}\vec{Q}`.
 * Fourier transform :math:`\hat{f} = \mathfrak{F}f`.
 * Multiply by the Poisson's Greens function to obtain the potential, :math:`\fou{\phi} = \frac{\hat{f}}{\varepsilon k^2}`.
 * Compute field via *ik* differentiation, :math:`\fou{\vec{E}} = i\vec{k}\fou\phi`.
 * Transform potential and field back to real space :math:`\phi = \mathfrak{F}^{-1}\fou\phi`; :math:`\vec{E} = \mathfrak{F}^{-1}\fou{\vec{E}}`.
 * Interpolate energy and/or force to charge locations, :math:`\phi_i = \oper{J}\phi`; :math:`\vec{E}_i = \oper{J}_{\vec{\ppos}_i}\vec{E}`.


The grid size is coupled to the Gaussian charge width, hindering the simulation of systems with narrow charges or large domains. UAMMD's implementation has an Ewald splitting mode to overcome this limitation.

Without getting into too much detail (see [2]_), we can write the potential as

.. math::
   
 \phi=(\phi - \gamma^{1/2}\star\psi) + \gamma^{1/2}\star\psi = \phi^{\near} + \phi^{\far},

where :math:`\star` represents convolution and the intermediate solution :math:`\psi` satisfies

.. math::

   \varepsilon\Delta\psi=-f\star\gamma^{1/2}.
   
The splitting function :math:`\gamma` is defined as

.. math::
   
 \gamma^{1/2} = \frac{8\xi^3}{(2\pi)^{3/2}}\exp\left(-2r^2\xi^2\right).

Here the splitting parameter, :math:`\xi`, is an arbitrary factor that is chosen to optimize performance. 
Given that the Laplacian commutes with the convolution we can divide the problem in two separate parts, denoted as near and far field  

.. math::
   
 &\varepsilon\Delta\phi^{\far}=-f\star\gamma,\\
 &\varepsilon\Delta\phi^{\near}=-f\star(1-\gamma).
 
The convolution of two Gaussians is also a Gaussian, so in the case of the far field the RHS results in wider Gaussian sources that can be interpreted as smeared versions of the original ones. The far field RHS thus decays exponentially in Fourier space and is solved as in the non Ewald split case.  
On the other hand the near field resulting charges are sharply peaked and more compactly supported than the originals, furthermore integrating to zero in 3D.  
The near field Green's function is computed analytically in real space and evaluated for each pair of particles inside a given radius (that is controlled by the desired tolerance). The electric field is computed by analytically differentiating and evaluating this Green's function.  
For a given tolerance, the splitting parameters controls the load that each part of the algorithm takes. In each case there will be an optimal split that gives the best performance.  


Usage
~~~~~~

The triply periodic Poisson solver is available as an :ref:`Interactor` called :cpp:any:`Poisson`.

The following parameters are available:  
  * :cpp:`Box box` Simulation domain (must be triply periodic).
  * :cpp:`real epsilon` Permittivity.
  * :cpp:`real gw` Gaussian width of the charges (all charges have the same width).
  * :cpp:`real tolerance` Overall tolerance of the algorithm.
  * :cpp:`real split = 0` The splitting parameter, :math:`\xi`, for the Ewald mode. If it is equal to 0 the non-Ewald split mode is used.


.. code:: c++
	  
  #include<uammd.cuh>
  #include<Interactor/SpectralEwaldPoisson.cuh>
  using namespace uammd;
  //Creates and returns a triply periodic Poisson solver Interactor
  auto createTPPoissonInteractor(std::shared_ptr<ParticleData> pd){
    Poisson::Parameters par;
    par.box = Box({128, 128, 128});
    //Permittivity
    par.epsilon = 1.0;
    //Gaussian width of the sources
    par.gw = 1.0; 
    //Overall tolerance of the algorithm
    par.tolerance = 1e-4;
    //If a splitting parameter is passed
    // the code will run in Ewald split mode
    //Otherwise, the non Ewald version will be used
    //par.split = 1.0;
    return std::make_shared<Poisson>(pd, par);
  }

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: A :ref:`ParticleGroup` can be provided instead of a :ref:`ParticleData` for the module to act only on a subset of particles.
	  
.. note:: As usual, this :ref:`Interactor` can be added to an :ref:`Integrator`.

.. note:: The tolerance is the maximum relative error allowed in the potential for two charges. The potential for L->inf is extrapolated and compared with the analytical solution. Also in Ewald split mode the relative error between two different splits is less than the tolerance. See test/Potential/Poisson  


Doubly periodic boundary conditions
-------------------------------------

We want to solve the Poisson equation with the following set of boundary conditions for the potential

.. math::
   
   &\phi(x,y,z\rightarrow 0^+)=\phi(x,y,z\rightarrow 0^-)\\
   &\phi(x,y,z\rightarrow H^-)=\phi(x,y,z\rightarrow H^+).
   
And for the electric field 

.. math::
   &\varepsilon_0 \frac{\partial \phi}{\partial z}(x,y,z\rightarrow 0^+)-\varepsilon_b \frac{\partial \phi}{\partial z}(x,y,z\rightarrow 0^-)=-\sigma_b(x,y)\label{eq:dppoissonbcs3}\\
   &\varepsilon_0 \frac{\partial \phi}{\partial z}(x,y,z\rightarrow H^-)-\varepsilon_t \frac{\partial \phi}{\partial z}(x,y,z\rightarrow H^+)=\sigma_t(x,y)
   
We introduce, via these BCs, the possibility of having arbitrary surface charges at the walls, :math:`\sigma_b` and :math:`\sigma_t` for the bottom and top respectively. Additionally, we can set different permittivities inside the slab (:math:`\varepsilon_0`) above (:math:`\varepsilon_t`) and below (:math:`\varepsilon_b`) it.

Finally, we assume that the domain is overall electroneutral,

.. math::
   
  \sum_{k=1}^N{Q_k} + \int_0^{L_{xy}}{\int_0^{L_{xy}}{(\sigma_b(x,y) + \sigma_t(x,y))dx dy}} = 0.

We impose that the sources do not overlap the boundaries in the :math:`z` direction, :math:`f(z>H \text{ or } z<0) = 0`, so that the charge density integrates to one inside the slab. Given that the Gaussian is not compactly supported we truncate it at :math:`n_\sigma a \ge 4 a` to overcome this, ensuring that the integral is at least :math:`99.9\%` of the charge :math:`Q`.

The approach to solve the set of equations above is wildly different from the triply periodic case, a complete description of the algorithm can be found in [3]_. In short, we use a grid-based solver as in the triply periodic case and make use of Ewald splitting, the main difference now is that we work in a Fourier-Chebyshev space instead of just Fourier.


Usage
~~~~~~~~

The creation of the Doubly Periodic Poisson Interactor is similar to that of the triply periodic case. With the exception that now the box size is communicated separately in the parallel and perpendicular directions and the permittivity can be different inside and outside the domain. Besides the parameters in the source code example below, additional ones are available to fine-tune several internal precision parameters (such as support, upsampling or overall tolerance). By default, the module will provide an overall tolerance of around 4 digits, which is the study case in the original work describing the doubly periodic algorithm [3]_. Additionally, a special functor can be provided specifying the surface charges.
In all instances, the surface charge will enforce overall electroneutrality inside the domain. For instance, if a single positive charge of strength :math:`Q` is located inside the domain, each wall will be assigned a constant charge of :math:`-Q/2`.

The following parameters are available:
  * :cpp:`real Lxy` Simulation domain size in the plane.
  * :cpp:`real H` Domain height (:math:`z\in [-H/2, H/2]`).
  * :cpp:`DPPoissonSlab::Permitivity perm` Permittivity in the three domains, contains a top, bottom and inside members.
  * :cpp:`real gw` Gaussian width of the charges (all charges have the same width).
  * :cpp:`real split = 0` The splitting parameter, :math:`\xi`, for the Ewald mode. If it is equal to 0 the non-Ewald split mode is used.
  * :cpp:`std::shared_ptr<SurfaceChargeDispatch> surfaceCharge` An object providing the surface charge, see below.
Additionally, some optional/advanced parameters are available:
  * :cpp:`int Nxy` Instead of the splitting parameter the number of cells for the far field can be specified.
  * :cpp:`int support` Number of support cells for the Gaussian kernel.
  * :cpp:`real numberStandardDeviations`  :math:`n_\sigma` above, number of standard deviations to truncate the Gaussian kernel at.
  * :cpp:`real tolerance` Controls the cut off distance of the near field Green's function.

.. code:: c++
	  
  #include<Interactor/DoublyPeriodic/DPPoissonSlab.cuh>
  using namespace uammd;
	  
  auto createDPPoissonInteractor(std::shared_ptr<ParticleData> pd){  
    DPPoissonSlab::Parameters par;
    par.Lxy = 128;
    par.H = 10; //Domain height
    DPPoissonSlab::Permitivity perm;
    perm.inside = 1.0;
    perm.top = 1.0;
    perm.bottom = 1.0;
    par.permitivity = perm;
    par.gw = 1.0; //Width of the Gaussian sources
    par.split = gw*0.1; //Splitting parameter
    auto poisson = make_shared<DPPoissonSlab>(pd, par);
    return poisson;
  }

.. hint:: The doubly periodic electrostatic :ref:`Interactor` does not accept an overall tolerance parameter. The accuracy is defaulted to provide a relative error around 1e-3. Advanced users can refer to [3]_ to tune the advanced parameters to achieve more accuracy.

Here, :code:`pd` is a :ref:`ParticleData` instance.

.. hint:: A :ref:`ParticleGroup` can be provided instead of a :ref:`ParticleData` for the module to act only on a subset of particles.
	  
.. note:: As usual, this :ref:`Interactor` can be added to an :ref:`Integrator`.

.. note:: A set of examples showcasing this implementation can be found at https://github.com/stochasticHydroTools/DPPoissonTests , which can be used to reproduce the results in [3]_.

Providing the surface charges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The surface charge parameter in the DPPoisson module must inherit from the type :cpp:any:`DPPoissonSlab_ns::SurfaceChargeDispatch`.

.. cpp:class:: DPPoissonSlab_ns::SurfaceChargeDispatch

   .. cpp:function:: virtual real top(real x, real y);

      Must return the surface charge at position x,y on the top wall, :math:`\sigma_t`.

   .. cpp:function:: virtual real bottom(real x, real y);

      Must return the surface charge at position x,y on the bottom wall, :math:`\sigma_b`.


Example
/////////////

A constant surface charge dispatcher.

.. code:: c++
	  
   #include<Interactor/DoublyPeriodic/DPPoissonSlab.cuh>
   using namespace uammd;
   
   struct ConstantSurfaceCharge: public DPPoissonSlab_ns::SurfaceChargeDispatch{
     real top(real x, real y) override{ return 1.0;}
     real bottom(real x, real y) override{ return -1.0;}
   };

      
****

.. rubric:: References:  

.. [1] Spectral accuracy in fast Ewald-based methods for particle simulations. Dag Lindbo and Anna-Karin Tornberg 2011. https://doi.org/10.1016/j.jcp.2011.08.022
.. [2] The Ewald sums for singly, doubly and triply periodic electrostatic systems. Tornberg, Anna-Karin 2015. https://doi.org/10.1007/s10444-015-9422-3       
.. [3] A fast spectral method for electrostatics in doubly periodic slit channels. Ondrej Maxian, Raul P. Pelaez et.al. 2021. https://doi.org/10.1063/5.0044677
