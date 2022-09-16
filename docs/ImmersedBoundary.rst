.. _IBM:

Immersed Boundary Method
========================


The Immersed Boundary Method (IBM) [Peskin2002]_ is a mathematical framework for simulation of fluid-structure interaction. In the IBM the forces acting on a certain marker (particle) are distributed (spread) to the nearby fluid grid points via some smeared delta function (see figure below). Furthermore, we can apply the inverse operation, called interpolation, that gathers the surrounding fluid velocity field into the marker positions. The IBM offers us a way to discretize the spreading and interpolation operators via some sophisticated :math:`\delta_a` kernels (smeared delta functions of typical width :math:`a`) often referred to as Peskin kernels.
Furthermore, the spreading and interpolation algorithms will also be applicable to other situations outside its intended purpose. For instance, to spread charges and interpolate electric fields when solving the Poisson equation (see :ref:`Electrostatics`). In general, we can use the spreading and interpolation algorithms here for transforming between a Lagrangian (particles) and an Eulerian (grid) description.

The spreading operator, :math:`\oper{S}`, transforms a quantity, :math:`\vec{F}=\{\vec{F}_1(\vec{\ppos}_1),\dots,\vec{F}_N(\vec{\ppos}_N)\}`, defined at the markers (particles) positions into a field, :math:`\vec{f}`, defined on a grid

.. math::

  \vec{f}(\vec{\fpos}) = \oper{S}(\vec{\fpos})\vec{F} = \sum_i\delta_a(\vec{\ppos}_i-\vec{\fpos})\vec{F}_i,

The interpolation operator (:math:`\oper{J} = \oper{S}^*`) averages a field, :math:`\vec{\fvel}`, defined on a grid at the position of a marker, :math:`\vec{\ppos}_i`. We discretize the interpolation operator as

.. math::
   
  \vec{\pvel}_i = \oper{J}_{\vec{\ppos}_i}\vec{\fvel}(\vec{\fpos}) = \int_V \delta_a(\vec{\ppos}_i - \vec{\fpos})\vec{\fvel}(\vec{\fpos}) d\vec{\fpos} \approx \sum_j{\delta_a(\vec{\ppos}_i - \vec{\fpos}_j)\vec{\fvel}(\vec{\fpos}_j)dV(j)}


Where the sum goes over all the cells, :math:`j`, in the grid, with centers (or nodes) at :math:`\vec{\fpos}_j`. The function :math:`dV(j)` are the quadrature weights (i.e the volume of the cell). For a regular grid with cell size :math:`h`, the quadrature weights are simply :math:`dV(j) := h^3`.



.. figure:: img/ibm.*
	    :width: 50%
	    :align: center

	    A representation of the Immersed Boundary. The blue circle represents a particle (with the green cross marking its center). Some quantity (i.e. the force) acting on it can be spread to the grid points inside its radius of action (red crosses). Alternatively, some quantity defined at the green crosses can be interpolated to the particle (via the interpolation operator).

   
   
A thorough description of the whole IBM can be found at [Peskin2002]_, including the equations of motion for an arbitrary structure submerged in an incompressible fluid. However, we will only make use of the properties devised for the kernels.

In particular, Peskin kernels must abide by a series of postulates that intend to maximize computational efficiency (which translates to closer support) while minimizing the discretization effect of the grid (such as translational invariance).

For the sake of simplicity, the first postulate consists in assuming the kernel can be separated as

.. math::

   \delta_a(\vec{r}=(x,y,z)) =\frac{1}{h^3}\phi\left(\frac{x}{h}\right)\phi\left(\frac{y}{h}\right)\phi\left(\frac{z}{h}\right)

This allows to state the postulates regarding the one-dimensional function, :math:`\phi`. Additionally, this form yields :math:`\delta_a\rightarrow\delta` as :math:`h\rightarrow 0`.
The second postulate is that :math:`\phi(r)` must be continuous for :math:`r\in\mathbb R`, avoiding jumps in the quantities spread to, or interpolated from, the grid. The close support postulate says that :math:`\phi(r>r_c) = 0`, being :math:`r_c` a cut off radius. This is our main means for seeking computational efficiency, since reducing the support of the kernel by one cell reduces dramatically the required operations. In particular, if the kernel has a support of :math:`n_s` cells in each direction, spreading or interpolating requires visiting :math:`n_s^3` nearby cells, so a support of :math:`n_s=5` requires :math:`125` cells while a support of :math:`3` requires just :math:`27`. Note that the support, :math:`n_s` must be large enough to include all cells within :math:`r_c` of the point to spread. In the case of a regular grid, this can be achieved by choosing :math:`n_s \ge 2r_c/h+1`.

The last basic postulate is required for the kernel to conserve the communicated quantities and it is simply a discrete expression of the fact that the kernel must integrate to unity.

.. math::
   
   \sum_j \phi(r-j) = 1 \textrm{ for } r\in\mathbb R

Where :math:`j` are the centers or the cells inside the support.
The next postulate intends to enforce the translational invariance of the distributed quantities as much as possible.

.. math::
   
   \sum_j\left(\phi(r-j)\right)^2 = C \textrm{ for any } r\in\mathbb R

Where :math:`C` is some constant to be determined. This is a weaker version of the condition for exact grid translational invariance

.. math::
   
  \sum_j\phi(r_1-j)\phi(r_2-j) = \Phi(r_1-r_2)

Which states that the coupling between any two points must be a function of their distance. However, it can be shown that satisfying this condition is incompatible with a compact support [Peskin2002]_. The above equation attempts to guarantee some degree of translational invariance by imposing a condition on the point with maximum coupling, :math:`r_1 = r_2`.

Finally, we can impose conditions on the conservation of the first :math:`n` moments to get increasingly higher order accuracy interpolants (at the expense of wider support)

.. math::
   
  \sum_j(r-j)^n\phi(r-j) = K_n

Where :math:`K_n` are some constants.
By solving the system of equations given by these conditions, different kernels can be found. 

   
Already defined kernels
~~~~~~~~~~~~~~~~~~~~~~~~

UAMMD exposes several kernels at :code:`src/misc/IBM_kernels.cuh`.

3-point Peskin kernel
------------------------

In particular, enforcing only the condition for the first moment (with :math:`K_1=0`) we arrive at the so-called 3-point Peskin kernel.

.. math::
   
  \phi_{p_3}(|r|) =  \left\{
  \begin{aligned}
    & \frac{1}{3}\left( 1 + \sqrt{1-3r^2}\right)& r < 0.5\\
    & \frac{1}{6}\left(5-3r-\sqrt{1-3(1-r)^2}\right)& r < 1.5\\
    & 0 & r>1.5 
  \end{aligned}\right.

Where the argument :math:`|r|` represents the fact that the above expression must be evaluated for the absolute value of the separation (since the kernel is symmetrical). The distance is in units of the cell size, :math:`h`.

.. cpp:class:: IBM_kernels::Peskin::threePoint

	       .. cpp:function:: threePoint(real h);
				 
				 The constructor requires the cell size.
	       

4-point Peskin kernel
----------------------

We can add a more restrictive condition on the integration to unity postulate

.. math::
   
  \sum_{j \textrm{ even}} \phi(r-j)  =  \sum_{j \textrm{ odd}} \phi(r-j) = \half

Which smooths the contributions of the kernel when using a central difference discretization for the gradient operator.
Solving for :math:`\phi` with this extra condition yields the classic 4-point Peskin kernel

.. math::
   
  \phi_{p_4}(|r|) =  \left\{
  \begin{aligned}
    & \frac{1}{8}\left( 3 - 2r + \sqrt{1+4r(1-r)}\right)& r < 1\\
    & \frac{1}{8}\left(5-2r-\sqrt{-7+12r-4r^2}\right)& r < 2\\
    & 0 & r>2
  \end{aligned}\right.

The main advantage of this kernel is that it interpolates linear functions exactly, and smooth functions are interpolated to second order accuracy. The distance, :math:`r` is in units of the cell size, :math:`h`.


.. cpp:class:: IBM_kernels::Peskin::fourPoint

	       .. cpp:function:: fourPoint(real h);
				 
				 The constructor requires the cell size.
	       


6-point Peskin kernel
-----------------------

Recently, a new 6-point kernel has been developed that satisfies the moment conditions up to :math:`n=3` for a special choice of :math:`K_2` [Bao2016]_. Additionally, it also satisfies the even-odd condition, it is three times differentiable and offers a really good translational invariance compared to similarly supported kernels.

This kernel sets :math:`K_1= K_3 = 0` and :math:`K_2 = \frac{59}{60} - \frac{\sqrt{29}}{20}`.

Solving for :math:`\phi` using these conditions, and defining the following

.. math::

    &\alpha = 28\\
    &\beta(r) = \frac{9}{4} - \frac{3}{2} (K_2 + r^2) + (\frac{22}{3}-7K_2)r - \frac{7}{3})r^3\\
    &\gamma(r) = -\frac{11}{32}r^2 + \frac{3}{32}(2K_2+r^2)r^2 +
    \frac{1}{72}\left((3K_2-1)r+r^3\right)^2 +\\
    &\qquad+\frac{1}{18}\left((4-3K_2)r -r^3\right)^2\\
    &\chi(r) = \frac{1}{2\alpha}\left( -\beta(r) + \textrm{sgn}(\frac{3}{2} - K_2)\sqrt{\beta(r)^2 - 4\alpha\gamma(r)}\right)

We get the expression for the 6-point kernel

.. math::
   
  \phi_{p_6}(|r|) =  \left\{
    \begin{aligned}
      & 2\chi(r) + \frac{5}{8} + \frac{1}{4}(K_2 + r^2)& r < 1\\[8pt]
      & -3\chi(r-1) + \frac{1}{4} - \frac{1}{6}\left((4-3K_2) + (r-1)^2\right)(r-1) & r < 2\\[8pt]
      & \chi(r-2) - \frac{1}{16} + \frac{1}{8}\left(K+(r-2)^2\right) - \\
      &\qquad-\frac{1}{12}\left((3K_2-1) - (r-2)^2\right)(r-2)& r<3\\[8pt]
      &0 &r>3
  \end{aligned}\right.

The distance, :math:`r` is in units of the cell size, :math:`h`.
Given its complexity it is advisable to tabulate :math:`\phi_{p_6}`. Other Peskin-like kernels can be found by enforcing other conditions, see for example [Yang2009]_.


.. cpp:class:: IBM_kernels::GaussianFlexible::sixPoint

	       .. cpp:function:: sixPoint(real h);
				 
				 The constructor requires the cell size.
	       

Barnett-Magland (BM) kernel
-----------------------------

A new kernel, called "exponential of the semicircle"(ES) and here referred to as BM, has been recently developed to improve the efficiency of non-uniform FFT methods [Barnett2019]_.
This kernel has a simple mathematical expression

.. math::
   
  \phi_{BM}(r,\{\beta, w\}) = \left\{
  \begin{aligned}
    &\frac{1}{S}\exp\left[\beta(\sqrt{1-(r/w)^2}-1)\right] & |r|/w\le 1\\
    & 0 & \textrm{otherwise}
  \end{aligned}\right.

Where :math:`\beta` and :math:`w` are parameters related to the shape and support (width) of the kernel. The parameter :math:`S(\beta, w)` is the necessary normalization to ensure that the BM kernel integrates to unity. Since it does not have an analytic integral, this factor must be computed numerically. One advantage of BM kernel is that it decays faster than a Gaussian in Fourier space, which is beneficial in spectral methods.

One disadvantage of the kernels above is that we do not know their analytical Fourier transform (in the case of the BM kernel this stems from it not having an analytical integral).


.. cpp:class:: IBM_kernels::BarnettMagland

	       .. cpp:function:: BarnettMagland(real w, real beta);
				 
				 The constructor requires :math:`w` and :math:`\beta`.


.. note:: Contrary to the Peskin kernels, the BM kernel does not provide a support distance by default. The kernel class must be inherited in order to define the :cpp:any:`getSupport` and :cpp:any:`getMaxSupport` methods.
	       

Gaussian kernel
-----------------

Finally, we can include here for completeness the Gaussian kernel, which can be defined as

.. math::
   
  \phi_G(r,\{\sigma\}) = \frac{1}{(2\pi\sigma)^{3/2}}\exp\left(\frac{-r^2}{2\sigma^2}\right)

Where :math:`\sigma` is the width of the Gaussian.

.. cpp:class:: IBM_kernels::Gaussian

	       .. cpp:function:: Gaussian(real width);
				 
				 The constructor requires the width :math:`\sigma` of the Gaussian.


.. note:: Contrary to the Peskin kernels, the Gaussian kernel does not provide a support distance by default. The kernel class must be inherited in order to define the :cpp:any:`getSupport` and :cpp:any:`getMaxSupport` methods.
	       


Defining a new kernel
~~~~~~~~~~~~~~~~~~~~~~~~~

Any kernel must adhere to the following interface

.. cpp:class:: IBMKernel

	       A conceptual interface class for IBM spread/interpolation kernels.
	       The return type of the phi functions: :cpp:`KernelReturnType` will be auto deduced and can be anything, as long as the :code:`WeightCompute` argument passed to :code:`spread` and :code:`gather` can handle it.
	       Note that the return type will be, more often than not, a simple :code:`real` number.
	       But there are situations where a more complex return type might be needed, for instance, to spread the value: :math:`(r_x^2 + r_y^2 + r_z^2)\phi(r_x)\phi(r_y)\phi(r_z)` we will need the return type to be a :code:`real2` storing :cpp:`r` and :cpp:`phi` in each direction.

   .. cpp:function:: __device__ int3 getSupport(real3 pos, int3 cell);

      Returns the number of support cells for a marker with position :cpp:`pos` lying inside cell :cpp:`cell`. Note that this function might just return the same number regardless of the position.

   .. cpp:function:: int3 getMaxSupport();

      Return the maximum support required by the kernel.
      
   .. cpp:function:: __device__ KernelReturnType phiX(real r, real3 pos);

      Computes the kernel at a distance :cpp:`r` in the X direction. The value of the kernel can depend on the position of the marker, given at :cpp:`pos`.
		     
   .. cpp:function:: __device__ KernelReturnType phiY(real r, real3 pos);

      Computes the kernel at a distance :cpp:`r` in the Y direction. The value of the kernel can depend on the position of the marker, given at :cpp:`pos`.
		     
   .. cpp:function:: __device__ KernelReturnType phiZ(real r, real3 pos);

      Computes the kernel at a distance :cpp:`r` in the Z direction. The value of the kernel can depend on the position of the marker, given at :cpp:`pos`.

   .. cpp:function:: __device__ KernelReturnType phi(real r, real3 pos);

      Instead of having a different function per direction (:cpp:any:`phiX`, :cpp:any:`phiY` and :cpp:any:`phiZ`) this single function can be defined instead (so that :cpp:`phiX = phiY = phiZ`).


Example
-----------

.. code:: cpp

  //A simple Gaussian kernel compatible with the IBM module.
  class Gaussian{
    const real prefactor;
    const real tau;
    const int support;
  public:
    Gaussian(real width, int support):
      prefactor(pow(2.0*M_PI*width*width, -0.5)),
      tau(-0.5/(width*width)),
      support(support){}

   int3 getMaxSupport(){
     return {support, support, support};
   }
   
    __device__ int3 getSupport(real3 pos, int3 cell){
      return getMaxSupport();
    }
  
    __device__ real phi(real r, real3 pos) const{
      return prefactor*exp(tau*r*r);
    }
  };



Usage in UAMMD
~~~~~~~~~~~~~~

The :cpp:any:`IBM` class is used to communicate between marker (particle) and grid data.

.. cpp:class:: template<class Kernel, class Grid, class Index3D> IBM
	       
   The IBM class is templated to be generic for any kernel, grid type (for instance regular vs others), and layout of the grid data (for instance row vs column major).
   By default, IBM will use UAMMD's :cpp:any:`Grid` type (a regular grid) and expect a linear indexing of the grid data (so that the data for a cell with coordinates :cpp:`(i,j,k)` is expected to be stored in :cpp:`gridData[(j+k*n.y)*n.x + i]`). See :ref:`Advanced functionality` on how to modify these behaviors.
	       
   .. cpp:function:: IBM(std::shared_ptr<Kernel> kernel, Grid grid);

      The basic constructor of the IBM module requires an instance of a :cpp:any:`Kernel` (see :ref:`Already defined kernels`) and a :cpp:any:`Grid` instance (which can be the :ref:`regular grid <Grid>` provided by UAMMD or any other class adhering to the Grid concept).
      The :cpp:any:`Grid` object will provide information like the dimensions of the grid and domain, or the distance between an arbitrary point and the center of a certain grid point.

   .. cpp:function::     template<class PosIterator,\
			 class QuantityIterator,\
			 class GridDataIterator>\
			 void spread(PosIterator q, QuantityIterator f,\
			 GridDataIterator gridData,\
			 int numberParticles, cudaStream_t st = 0);

      The basic overload of the spread function takes the position of the markers, the values defined at each marker's positions and the grid data (stored according to :cpp:any:`Index3D`).
      This function will add to each cell, :math:`c_j`, in :cpp:`gridData` the result of :math:`c_j += \sum_i\text{WeightCompute}(\delta_a(\vec{\ppos}_i-\vec{\fpos}_{c_j}), \vec{f}_i)`.
      Here :math:`\text{WeightCompute}` defaults to multiplication, see :ref:`Advanced functionality`. 
      The types of the different quantities are irrelevant as long as the required arithmetics are defined for them (for instance, the weight compute must be able to process the type of the marker data and return a type for which the :cpp:`GridDataIterator::value_type::operator+()` and :cpp:`GridDataIterator::value_type::operator=()` exists.

   .. cpp:function::     template<class PosIterator,\
			 class QuantityIterator,\
			 class GridDataIterator>\
			 void gather(PosIterator q, QuantityIterator v,\
			 GridDataIterator gridData,\
			 int numberParticles, cudaStream_t st = 0);

      The basic overload of the interpolation function takes the position of the markers, the values defined at each marker's positions and the grid data (stored according to :cpp:any:`Index3D`).
      This function will add to each particles value, :math:`v_i`, the result of :math:`v_i += \sum_j\text{WeightCompute}(\delta_a(\vec{\ppos}_i-\vec{\fpos}_{c_j}), \vec{v}_{c_j})*\text{QuadratureWeight}_j`, where :math:`\vec{c}_j` is the value stored for cell :math:`j`.
      Here :math:`\text{WeightCompute}` defaults to multiplication. The :math:`\text{QuadratureWeight}` of each cell defaults to :math:`h^3`, the volume of a grid cell provided by the :cpp:any:`Grid` object. See :ref:`Advanced functionality` for more information.
      The types of the different quantities are irrelevant as long as the required arithmetics are defined for them (for instance, the weight compute must be able to process the type of the grid data and return a type for which the :cpp:`GridDataIterator::value_type::operator+()` and :cpp:`GridDataIterator::value_type::operator=()` exists.


Example
-----------

Spreading and interpolating using most of the default behavior in the IBM module.

.. code:: cpp

  #include<uammd.cuh>
  #include<misc/IBM.cuh>
  #include<misc/IBM_kernels.cuh>
  using namespace uammd;
  int main(){
    using Kernel = IBM_kernels::Peskin::threePoint;
    //using Kernel = IBM_kernels::Peskin::fourPoint;
    //using Kernel = IBM_kernels::GaussianFlexible::sixPoint;
    real L = 128;
    Grid grid(Box({L,L,L}), make_int3(L, L, L));
    int3 n = grid.cellDim;
    //Initialize some arbitrary per particle data
    thrust::device_vector<real> markerData(numberParticles);  
    thrust::fill(markerData.begin(), markerData.end(), 1.0);
    
    thrust::device_vector<real3> markerPositions(numberParticles);
    { //Initialize some arbitrary positions
      std::mt19937 gen(sys->rng().next());
      std::uniform_real_distribution<real> dist(-0.5, 0.5);
      auto rng = [&](){return dist(gen);};
      std::generate(markerPositions.begin(), markerPositions.end(), [&](){ return make_real3(rng(), rng(), rng())*L;});
    }
    //Allocate the output grid data
    thrust::device_vector<real> gridData(n.x*n.y*n.z);
    thrust::fill(gridData.begin(), gridData.end(), 0);
    
    auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
    IBM<Kernel> ibm(kernel, grid);
    auto pos_ptr = thrust::raw_pointer_cast(markerPositions.data());
    auto markerData_ptr = thrust::raw_pointer_cast(markerData.data());
    auto gridData_ptr = thrust::raw_pointer_cast(gridData.data());
    ibm.spread(pos_ptr, markerData_ptr, gridData_ptr, numberParticles);

    //We can now go back to the particles, performing the inverse operation (interpolation).
    thrust::fill(markerData.begin(), markerData.end(), 0);
    ibm.gather(pos_ptr, markerData_ptr, gridData_ptr, numberParticles);
    return 0;
  }
  

.. note:: The value types of the marker and grid data are irrelevant, as long as the arithmetics of the types are well defined. We could, for instance, have both the grid and marker data be of type :cpp:any:`real3` to spread/interp 3 values at once.
	  
.. note:: By default, the IBM module expects the grid data to be stored in a row-major format. The data for a cell with coordinates :cpp:`(i,j,k)` is expected to be stored in :cpp:`gridData[(j+k*n.y)*n.x + i]`. This default indexing can be modified, see :ref:`Advanced functionality` below.


Advanced functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~

The IBM module offers a deep level of customization, besides the functionality described above (mainly the selection of an arbitrary kernel and grid/marker data types) we can:
 * Modify the memory layout of the grid data.
 * Modify the :cpp:any:`Grid` type (a regular grid by default).
 * Modify how the multiplication of the kernel and a marker value is performed when spreading (or the kernel and a grid value when interpolating).
 * Modify the quadrature weights of each grid point in the gather operation.


Changing the default indexing of the grid data
-------------------------------------------------

By default, the IBM module looks for the data of cell :cpp:`(i,j,k)` at :cpp:`gridData[(j+k*n.y)*n.x + i]`. The :cpp:any:`IBM` class exposes an optional template parameter that allows to modify this assumption.

.. cpp:function:: IBM::IBM(std::shared_ptr<Kernel> kernel, Grid grid, Index3D index);

		  A constructor that allows to specialize the module with a different indexing for the grid data. The default type for :cpp:any:`Index3D` is :cpp:any:`IBM_ns::LinearIndex3D`.

		  
.. cpp:class:: Index3D

	       A conceptual functor that provides the index in the grid data arrays in :cpp:any:`IBM` given its coordinates.

   .. cpp:function::  __device__ int operator()(int i, int j, int k);

	The parenthesis operator must take the 3D coordinates of a cell and return the index of the value in the grid data arrays. The default :cpp:any:`Index3D` type is constructed with the grid dimensions, :cpp:`nx,ny,nz`, and its parenthesis operator returns :cpp:`(j+k*n.y)*n.x + i`.
      

Example
***********

Using the IBM module with a non-default indexing of the grid data arrays.

.. code:: cpp


    struct LinearIndex3D{
      LinearIndex3D(int nx, int ny, int nz):nx(nx), ny(ny), nz(nz){}

      __device__ int operator()(int i, int j, int k) const{
	return i + nx*(j+ny*k);
      }

    private:
      const int nx, ny, nz;
    };

    int main(){
      //Grid grid; Assume a grid instance is available
      using Kernel = IBM_kernels::Peskin::threePoint;
      auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
      int3 n = grid.cellDim;
      LinearIndex3D cell2index(n.x, n.y, n.z);
      IBM<Kernel, Grid, LinearIndex3D> ibm(kernel, grid, cell2index);
      //Now ibm.spread() and ibm.gather() will look for the data of cell (i,j,k) at cell2index(i,j,k).
      return 0;
    }


Using a non-default Grid
-----------------------------

Any Grid type adhering to the :cpp:any:`Grid` concept can be used to specialize :cpp:any:`IBM`.

.. code:: cpp

    #include<uammd.cuh>
    #include<misc/IBM.cuh>
    #include<misc/IBM_kernels.cuh>
    #include<utils/Grid.cuh>
    using namespace uammd;

    //A grid that simply inherits everything from UAMMD's default regular grid
    struct MyGrid : public Grid{
     
      MyGrid(Box box, int3 in_cellDim): Grid(box, in_cellDim){}
    };
     
    int main(){
      //MyGrid grid; Assume a grid instance is available
      using Kernel = IBM_kernels::Peskin::threePoint;
      auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
      int3 n = grid.cellDim;
      IBM<Kernel, MyGrid> ibm(kernel, grid);
      //Now ibm.spread() and ibm.gather() will use the rules in MyGrid.
      return 0;
    }

.. note:: The Doubly Periodic modules make use of the IBM module with a Chebyshev grid, defined at :code:`src/mis/ChebyshevUtils.cuh`.


Modify how the multiplication of the kernel and a marker value is performed when spreading (or the kernel and a grid value when interpolating).
---------------------------------------------------------------------------------------------------------------------------------------------------


.. cpp:function::     template<class PosIterator,\
		      class ResultIterator,\
		      class GridQuantityIterator,\
		      class WeightCompute>\
		      void spread(PosIterator pos,\
		                  ResultIterator Jq,\
				  GridQuantityIterator gridData,\
				  WeightCompute wc,\
				  int numberParticles, cudaStream_t st = 0)

		      This overload of the :cpp:any:`spread` function allows to specialize the operation with a non-default :cpp:any:`WeightComputation`

		      
.. cpp:class:: WeightComputation


   .. cpp:function::  template<class T, class T2> __device__ auto operator()(T value, thrust::tuple<T2, T2, T2> kernel);

		      Takes a value (defined at a markers position in the case of spreading or at a grid point in the case of interpolation) and a kernel evaluation. Returns the value that will be added to the other description (a value at a grid point in the case of spreading and at a markers position in the case of interpolation). The default weight computation will return :cpp:`value*thrust::get<0>(kernel)*thrust::get<1>(kernel)*thrust::get<2>(kernel);`.
		      The type :cpp:`T2` will be determined by the return type of the kernel.


Example
********


.. code:: cpp

  #include<uammd.cuh>
  #include<misc/IBM.cuh>
  #include<misc/IBM_kernels.cuh>
  using namespace uammd;

  struct MyWeightCompute{
  
    inline __device__ real operator()(real value, thrust::tuple<real, real, real> kernel) const{
  	return value*thrust::get<0>(kernel)*thrust::get<1>(kernel)*thrust::get<2>(kernel);
    }

  };

  int main(){
    using Kernel = IBM_kernels::Peskin::threePoint;
    real L = 128;
    Grid grid(Box({L,L,L}), make_int3(L, L, L));
    int3 n = grid.cellDim;
    //Initialize some arbitrary per particle data
    thrust::device_vector<real> markerData(numberParticles);  
    thrust::fill(markerData.begin(), markerData.end(), 1.0);
    
    thrust::device_vector<real3> markerPositions(numberParticles);
    { //Initialize some arbitrary positions
      std::mt19937 gen(sys->rng().next());
      std::uniform_real_distribution<real> dist(-0.5, 0.5);
      auto rng = [&](){return dist(gen);};
      std::generate(markerPositions.begin(), markerPositions.end(), [&](){ return make_real3(rng(), rng(), rng())*L;});
    }
    //Allocate the output grid data
    thrust::device_vector<real> gridData(n.x*n.y*n.z);
    thrust::fill(gridData.begin(), gridData.end(), 0);
    
    auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
    IBM<Kernel> ibm(kernel, grid);
    auto pos_ptr = thrust::raw_pointer_cast(markerPositions.data());
    auto markerData_ptr = thrust::raw_pointer_cast(markerData.data());
    auto gridData_ptr = thrust::raw_pointer_cast(gridData.data());
    MyWeightCompute weightCompute;
    ibm.spread(pos_ptr, markerData_ptr, gridData_ptr, weightCompute, numberParticles);

    return 0;
  }

		      
		      
Modifying the quadrature weights of each grid point in the gather operation
-------------------------------------------------------------------------------

.. cpp:function::     template<class PosIterator,\
		      class ResultIterator,\
		      class GridQuantityIterator,\
		      class QuadratureWeights,\
		      class WeightCompute>\
		      void gather(PosIterator pos,\
		                  ResultIterator Jq,\
				  GridQuantityIterator gridData,\
				  QuadratureWeights qw,\
				  WeightCompute wc,\
				  int numberParticles, cudaStream_t st = 0)

				  
   This overload of the :cpp:any:`gather` function allows to specialize the operation with a non-default :cpp:any:`QuadratureWeight`. Note that if this functionality is required, a :cpp:any:`WeightComputation` must also be provided, which can just be the default one: :cpp:`IBM_ns::DefaultWeightComputation`.


.. cpp:class:: QuadratureWeight


   .. cpp:function::  __device__ real operator()(int3 cell, const Grid &grid);

		      Takes the coordinates of a cell and a grid, must return the quadrature weights of the cell. The default implementation returns the volume of the cell via :cpp:`grid.getCellVolume(cell);`.


Example
*********

.. code:: cpp

  #include<uammd.cuh>
  #include<misc/IBM.cuh>
  #include<misc/IBM_kernels.cuh>
  using namespace uammd;

  struct MyQuadratureWeights{
    __host__ __device__ real operator()(int3 cellj, const Grid &grid){
      return grid.getCellVolume(cellj);
    }
  };

  int main(){
    using Kernel = IBM_kernels::Peskin::threePoint;
    real L = 128;
    Grid grid(Box({L,L,L}), make_int3(L, L, L));
    int3 n = grid.cellDim;
    //Initialize some arbitrary per particle data
    thrust::device_vector<real> markerData(numberParticles);  
    thrust::fill(markerData.begin(), markerData.end(), 0.0);
    
    thrust::device_vector<real3> markerPositions(numberParticles);
    { //Initialize some arbitrary positions
      std::mt19937 gen(sys->rng().next());
      std::uniform_real_distribution<real> dist(-0.5, 0.5);
      auto rng = [&](){return dist(gen);};
      std::generate(markerPositions.begin(), markerPositions.end(), [&](){ return make_real3(rng(), rng(), rng())*L;});
    }
    //Allocate the output grid data
    thrust::device_vector<real> gridData(n.x*n.y*n.z);
    thrust::fill(gridData.begin(), gridData.end(), 1.0);
    
    auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
    IBM<Kernel> ibm(kernel, grid);
    auto pos_ptr = thrust::raw_pointer_cast(markerPositions.data());
    auto markerData_ptr = thrust::raw_pointer_cast(markerData.data());
    auto gridData_ptr = thrust::raw_pointer_cast(gridData.data());
    IBM_ns::DefaultWeightCompute weightCompute;
    MyQuadratureWeights quadratureWeights;
    ibm.gather(pos_ptr, markerData_ptr, gridData_ptr,
	  quadratureWeights, weightCompute,
	  numberParticles);

    return 0;
  }




****

.. rubric:: References:  

.. [Peskin2002] The immersed boundary method. Peskin, Charles S. 2002. Acta Numerica 11. https://doi.org/10.1017/S0962492902000077
.. [Bao2016]  A Gaussian-like immersed-boundary kernel with three continuous derivatives and improved translational invariance. Yuanxun Bao and Jason Kaye and Charles S. Peskin 2016. Journal of Computational Physics 316.  https://www.sciencedirect.com/science/article/pii/S0021999116300663
.. [Barnett2019] A Parallel Nonuniform Fast Fourier Transform Library Based on an "Exponential of Semicircle" Kernel. Barnett, Alexander H. and Magland, Jeremy and af Klinteberg, Ludvig 2019. SIAM Journal on Scientific Computing 41. https://doi.org/10.1137/18M120885X
.. [Yang2009] A smoothing technique for discrete delta functions with application to immersed boundary method in moving boundary simulations. Xiaolei Yang et. al. 2009.  Journal of Computational Physics 228. https://doi.org/10.1016/j.jcp.2009.07.023

	     
