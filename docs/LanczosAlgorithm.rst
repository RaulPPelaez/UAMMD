Lanczos Algorithm
=================

The Lanczos algorithm is an iterative method for finding the approximate solutions of large, sparse linear systems of equations. It is often used as a preconditioner for other iterative methods, such as the conjugate gradient method. The algorithm works by constructing a sequence of vectors, called the Lanczos basis, which span the Krylov subspace associated with the linear system. The approximate solutions are then found by minimizing the residual error in this subspace. The algorithm is known for its good convergence properties and its ability to handle large, ill-conditioned systems.

Many times in UAMMD computing the product of the square root of a martix and a vector (:math:`\sqrt{\tens{M}}\vec{v}`) is needed. The Lanczos algorithm can be leveraged for that.

.. note::

   Even so the term Lanczos algorithm is more generic (as stated above) the module in UAMMD that computes the square root of a matrix times a vector is referred to as Lanczos algorithm. This is due to the fact that I did not knew better hen I implemented the method described in the reference [1]_


.. cpp:class:: lanczos::Solver

   The class that performs the product of the square root of a matrix and a vector (:math:`\sqrt{\tens{M}}\vec{v}`).
   The Lanczos algorithm requires to compute the dot product of the matrix and the vector several times.
   The complexity of the algorithm depends on the cost of performing the dot product of the Matrix and a vector (which must be provided as a functor to the run function).
   If M has size NxN and the cost of the dot product is O(M). The total cost of the algorithm is O(m·M). Where m << N.
   If M·v performs a dense M-V product, the cost of the algorithm would be O(m·N^2).
   
   .. cpp:function:: Solver();

      The constructor requires no arguments.

   .. cpp:function:: int run(MatrixDot &dot, real *out, const real* vector, real tolerance, int N, cudaStream_t st = 0);

      :param dot: A valid :cpp:any:`MatrixDot` object encoding the dot product of the target matrix and some arbitrary vector.
      :param out: The output, filled with the result of the square root of the target matrix and the given vector
      :param vector: The vector to be multiplied by the square root of the matrix.
      :param tolerance: The algorithm will run as many iterations as needed for the error to be below this threshold.
      :param N: The size of the vector.
      :param st: An optional cuda stream.
      :returns: The number of iterations required to achieve the desired tolerance.

   .. cpp:function:: real runIterations(MatrixDot &dot, real *out, const real* vector, int numberIterations, int N);

      An alternative version of the :cpp:any:`run` function. Instead of requiring a tolerance and returning a number of iterations to convergence, this version requires a number of iterations and returns the residual after those iterations.
      
   .. cpp:function:: void setIterationHardLimit(int newLimit);

      If the problem is ill-formed the Lanczos algorithm might not converge and try to run infinite iterations.
      The upper limit for the number of iterations before failing can be set with this function.

   .. cpp:function:: int getLastRunRequiredSteps();

      Return the number of iterations required for convergence for the last time :cpp:any:`run` was called.

      
.. cpp:class:: lanczos::MatrixDot

   This pure virtual class can be inherited to provide :cpp:any:`lanczos::Solver` with a way to compute a matrix vector product.

   .. cpp:function:: void setSize(int newsize);

      Sets the size of the vector. Simply sets the member variable :cpp:any:`m_size` in case the operator() needs it.
   
   .. cpp:function:: virtual void operator()(real* v, real *mv) = 0;
		  
      :param v: An arbitrary input vector.
      :param mv: Must be overwritten with the result of multiplying the target matrix by the vector v.

   .. cpp:member:: int m_size;

      Objects inheriting MatrixDot will have available this member, containing the size of the vector and matrix.
      

.. hint:: Inheriting from MatrixDot is not actually needed, any callable object with the correct () operator will do.
	  


Examples
---------

Example of a MatrixDot functor:

.. code:: c++
   
  //Encodes the matrix dot product of the diagonal matrix M_{ij} = 2\delta_{ij}
  struct IdentityMatrixDot: public lanczos::MatrixDot{
  
    void operator()(real* v, real*mv) override{
      auto cit = thrust::make_constant_iterator<real>(2.0);
      thrust::transform(thrust::cuda::par,
  		      cit, cit + m_size,
  		      v,
  		      mv,
  		      thrust::multiplies<real>());
    }
  
  };

Computing the product of the square root of the matrix defined above and a vector filled with 1.

.. code:: c++

  real tol = 1e-7;
  int size = 128;
  auto lanczos = std::make_shared<lanczos::Solver>();
  IdentityMatrixDot dot;
  thrust::device_vector<real> Mv(size);
  thrust::fill(Mv.begin(), Mv.end(), real());
  thrust::device_vector<real> v(size);
  thrust::fill(v.begin(), v.end(), 1);
  lanczos->run(dot, Mv.data().get(), v.data().get(), tol, size);
  //Now Mv contains sqrt(2) (within 1e-7 tolerance) in all its elements.

  
.. rubric:: References::

..   [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations. T. Ando et.al. JCP 2012.
