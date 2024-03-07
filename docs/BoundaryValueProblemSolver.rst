Boundary Value Problem (BVP) Solver
===================================


Some of the solvers in UAMMD (in particular the doubly periodic ones) require solving a particular one dimensional boundary value problem in a batched manner, which can be written in general as

.. math::

  (\partial_z^2-k^2) y(z) = f(z),

where the solver must be applied for a group of different values of :math:`k` (which correspond to wavenumbers in the current usages of the solver).

With the boundary conditions:

.. math::  

  (\alpha_0\partial_z + \alpha_1) y(H) &= \alpha_2\\
  (\beta_0\partial_z + \beta_1) y(-H) &= \beta_2

Being :math:`\alpha_n,\beta_n` some arbitrary parameters. These parameters can take any value (including zero) as long as the resulting BVP remains well defined. For instance, making  :math:`\alpha_1,\beta_1` equal to zero at the same time results in a system with no unique solution.

Both :math:`y(z)` and :math:`f(z)` can be real or complex-valued as well as :math:`\alpha_2,\beta_2,k` and any other parameter.
.. hint::

   Initialization of the batched BVP solver requires inverting a dense matrix for each value of :math:`k`, which can become quite expensive. The solver tries to mitigate this cost by inverting these matrices in parallel, but experience suggests that letting it use more than 4 cores is counter-productive.


The BVP solver requires :math:`O(n_z^2)` operations at initialization and around :math:`O(5n_z)` operations at each subsequent call to solve.
In the batched version each problem is independent of the rest, so the cost is multiplied times the number of subproblems.

The BVP solver works in `Chebyshev space <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_, so its input and output are Chebyshev coefficients (see the :ref:`Fast Chebyshev Transform` module for utilities to transform between real/complex space and Chebyshev coefficients).

More information about how the BVP solver in UAMMD works can be found in `Raul's manuscript <https://raw.githubusercontent.com/RaulPPelaez/tesis/main/manuscript.pdf>`_.

Usage
------

Every function and class in the BVP solver source code lies under the :cpp:`uammd::BVP` namespace.
The BVP solver library exposes two main template classes , designed to manage both real and complex-valued numbers. To ensure consistent behavior and avoid unexpected results, it is recommended to use the complex number types defined in the Thrust library, specifically thrust::complex<real>, when working with complex numbers.

.. cpp:class:: template<class T> BatchedBVPHandler<T>

   Used to initialize and hold the information for a group of boundary value problems. Each subproblem can have different parameters (mainly :math:`\alpha_n,\beta_n,k` in the above equations.
   This class cannot be used in device code, its only used for initialization. See :cpp:any:`BatchedBVPGPUSolver`.

   .. cpp:function::  template<class KIterator, class BatchedTopBC, class BatchedBottomBC> BatchedBVPHandler(const KIterator &klist, BatchedTopBC top, BatchedBottomBC bot, int numberSystems, real H, int nz);

		  Initializes all requested boundary value problems.
		  
		  :param numberSystems: Number of subproblems.
		  :param klist: An iterator containing the value of :math:`k` for each subproblem.
		  :param top: A :cpp:any:`BoundaryCondition` iterator for the top BCs (provides :math:`\alpha_0,\alpha_1`).
		  :param bot: A :cpp:any:`BoundaryCondition` iterator for the bottom BCs (provides :math:`\beta_0,\beta_1`).
		  :param H: The location of the boundary conditions (goes from -H to H).
		  :param nz: Number of elements of a subproblem (all subproblems must have the same size).

   .. cpp:function::  template<class T> BatchedBVPGPUSolver_impl<T> getGPUSolver();

      Provides an instance of the solver to be used in the GPU.

.. cpp:class:: template<class T> BatchedBVPGPUSolver<T>

   While :cpp:any:`BatchedBVPHandler<T>` is used to initialize and store the different subproblems, this class is used to actually solve the subproblems in a CUDA kernel.
   This class has no public constructors, the only way to get an instance to it is via :cpp:any:`BatchedBVPHandler<T>::getGPUSolver`.

   .. cpp:function:: template<class T, class FnIterator, class AnIterator, class CnIterator>   __device__ void solve(int instance,		    const FnIterator& fn,			    T alpha_2, T beta_2,			    AnIterator& an,			    CnIterator& cn);

		     Solves one subproblem in the list.
		     
		     :param instance: The index of the subproblem (given by the order of k values when initializing).
		     :param fn: Iterator with the Chebyshev coefficients of the right hand side of the BVP equation.
		     :param alpha_2: Right hand side of the top BC.
		     :param beta_2: Right hand side of the bottom BC.
		     :param an: Output. An iterator to the Chebyshev coefficients of the first derivative of the solution.
		     :param cn: Output. An iterator to the Chebyshev coefficients of the solution.
   



Initialization requires an iterator to a special type of functor that provides the information for the Boundary conditions. This class must comply with the following interface:

.. cpp:class:: BoundaryCondition

   Returns the parameters of the equations for the boundary conditions of the BVP.
	       
   .. cpp:function:: T getFirstIntegralFactor();
		     
      Returns :math:`\alpha_0` or :math:`\beta_0`, depending on which BC this class represents.

   .. cpp:function:: T getFirstIntegralFactor();

      Returns :math:`\alpha_1` or :math:`\beta_1`, depending on which BC this class represents.

Aliases for Real and Complex Types
----------------------------------

To facilitate the use of the BVP solver with real and complex numbers, the following aliases are defined:

.. code-block:: cpp

    using BatchedBVPHandlerReal     = BatchedBVPHandler_impl<real>;
    using BatchedBVPHandlerrComplex = BatchedBVPHandler_impl<thrust::complex<real>>;

    using BatchedBVPGPUSolverReal    = BatchedBVPGPUSolver_impl<real>;
    using BatchedBVPGPUSolverComplex = BatchedBVPGPUSolver_impl<thrust::complex<real>>;

These aliases allow for a more intuitive and type-safe way to work with the BVP solver for different numerical types.

Example
++++++++

In this example we will solve a family of equations given by:

.. math::

  (\partial_z^2-k^2) y_n = 0,

where :math:`y_n` are the Chebyshev coefficients of the solution.

With the boundary conditions:

.. math::  

  (\partial_z - 1) y(H) &= 0\\
  (\partial_z + 1) y(-H) &= 0


We will solve this problem for a series of integer values of k going from 0 to the number of systems (just an example for simplicity).
Also for simplicity the alpha and beta parameters in the left hand side of the BCs are identical, but note that the iterator for the solver handler can hold different BCs for different subproblems.

.. code:: c++

  #include"misc/BVPSolver.cuh"
  #include<algorithm>
  
  using namespace uammd;
  using namespace uammd::BVP;
  
  struct TopBoundaryCondition{
   real getFirstIntegralFactor(){return 1;}
   real getSecondIntegralFactor(){return -1;}
  }
  
  struct BottomBoundaryCondition{
   real getFirstIntegralFactor(){return 1;}
   real getSecondIntegralFactor(){return 1;}
  }

  __global__ void solveBVP(BatchedBVPGPUSolver solver,
    real* rightHandSize, real* firstDerivative, real* solution, int nz, int numberSystems){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id>=numberSystems) return;
    real alpha = 0;
    real beta = 0;
    //Lets store each problem contigously (note that an strided storage would be more efficient, and can be obtained
    // by using a thrust::permutation_iterator, for instance). See current usages of the module in UAMMD
    real* fd = firstDerivative + id*nz;
    real* sol = solution + id*nz;
    real* rhs = rightHandSide + id*nz;
    solver.solve(id, rhs, alpha, beta, fd, sol);
  }
  
  int main(){
    std::vector<real> klist(numberSystems);
    std::iota(klist.begin(), klist.end()); //Fill with range 0:numberSystems
    real H = 1;
    auto topbc = thrust::make_constant_iterator(TopBoundaryConditions());
    auto botbc = thrust::make_constant_iterator(BottomBoundaryConditions());
    int nz = 32;
    auto bvp = std::make_shared<BVP::BatchedBVPHandler>(klist, topbc, botbc,
                                                        numberSystems, H, nz);
    auto gpu_solver = bvp.getGPUSolver();
    //Storage for the first derivative of the solution for each problem
    thrust::device_vector<real> firstDerivative(numberSystems*nz);
    //Storage for the Cheb coefficients of the solution
    auto solution = firstDerivative;
    auto rightHandSide = firstDerivative;
    thrust::fill(rightHandSide.begin(), rightHandSide.end(), 0);
    solveBVP<<<numberSystems/128+1, 128>>>(gpu_solver,
                                           rightHandSide.data().get(),
	                                   firstDerivative.data().get(), solution.data().get(),
                                           nz, numberSystems);
    return 0;
  }
