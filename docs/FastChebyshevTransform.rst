Fast Chebyshev Transform
=========================

Some solvers benefit from working in Chebyshev space. In fact, most current modules leveraging the Chebyshev space employ Fourier space in the rest of directions in what we call Fourier-Chebyshev space.

.. hint::
   The doubly periodic electrostatics and Stokes modules solve their corresponding 3D equations in Fourier space in the plane directions and in Chebyshev space in the perpendicular direction (Z).

  
This apparently strange relation between the two spaces comes from the fact that the numerical machinery already in place for the Fourier transform (the FFT) can be tricked into performing a Chebyshev transformation. 

The Chebyshev transform is defined via the series expansion of a certain function :math:`f(z)` into `Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_. In particular we can write

.. math::

   f(z) = \sum_n f_n T_n(z)

Where :math:`f_n` are referred to as Chebyshev coefficients of :math:`f(z)` and :math:`T_n(z)` is the n'th Chebyshev polynomial.

Theory
------

Chebyshev polynomials show several interesting properties for numerical solvers. The most important one in the UAMMD solvers is the fact that

.. math::
   
   T_n(z_m = \cos(m\pi/(N-1))) = \cos(\pi nm/(N-1))

Meaning that if we evaluate the function at the Chebyshev roots, the Chebyshev decomposition becomes a cosine series.

Let us see how to leverage this by inspecting the Chebyshev transform of a certain function :math:`u(x_n)` into its Chebyshev coefficients :math:`a_m`, defined as

.. math::

   u(x_n) = u_n = \sum_{m=0}^Na_mT_m(x_n).
   
For the direct transform we have:

.. math::

   a_m = \frac{p_m}{N-1}\left[\frac{1}{2}\left(u_0 + u_{N-1}(-1)^m\right) + \sum_{n=1}^{N-2}u_n T_m(x_n)\right]
   
where

.. math::
   
  p_m =
      \begin{cases}
        1 &\text{ if }  m=0,N-1\\
        2 &\text{ otherwise}\\
      \end{cases}

It is also worth mentioning that :math:`T_n(-x) = (-1)^nT_n(x)`

.. hint::
   
  Note that we can also define :math:`T_n(x_m = -\cos(m\pi/(N-1))) = (-1)^n\cos(\pi nm/(N-1))`,
  which leads to
  
  .. math::
  
     a_m = \frac{p_m}{N-1}\left[\frac{1}{2}\left(u_0(-1)^m + u_{N-1}\right) + \sum_{n=1}^{N-2}u_n T_m(x_n)\right]
  

On the other hand, the discrete cosine transform, as defined by:

.. math::

   \fou{f}_k = \frac{1}{2}(f_0 + (-1)^kf_{N-1}) + \sum_{n=1}^{N-2}f(x_n)\cos\left(\frac{\pi nk}{N-1}\right)
   
for :math:`k=0,\dots,N-1`, with

.. math::
   
   x_m = \cos(m\pi/(N-1))
   
is exactly equivalent to a DFT of size :math:`2(N-1)` real numbers with even symmetry (periodic extended).
This means that a signal given by :math:`abcde` is transformed into :math:`abcdedcb`, where we have left out the first mode at the end and the last one only appears once.

This eq. also happens to be identical to our definition of Chebyshev transform in except for the factor in :math:`p_m`.

So all we have to do is to mirror our signal such that

.. math::
   
 y_n = \begin{cases} 
 f_n, &\text{ for } n=0,\dots,N-1\\
 f_{2N-2-n}, &\text{ for } n=N,\dots,2N-2
 \end{cases}
 
then perform a 1D FFT on that new signal. Finally we need to reescale by the :math:`p_m/(2N-2)` factor.
The inverse transform is similar, we scale by :math:`(2N-2)/p_m`, then compute the periodic extension and then perform the iFFT.
Additionally, note that if we have a 3D field we can Fourier transform it in the plane and Chebyshev transform it in Z by using a single 3D FFT.

Functions for the FCT in UAMMD
---------------------------------

UAMMD exposes several functions in the :code:`misc/Chebyshev/FastChebyshevTransform.cuh` source code for Chebyshev and Fourier-Chebyshev signals.
All functions lie under the :code:`uammd::chebyshev` namespace.

.. warning:: All the functions create and destroy a cufft plan.
	     
.. cpp:function::  template<class Container> auto fourierChebyshevTransform3DCufft(Container i_fx, int3 n);

      From a 3D field of complex values with Z values evaluated at Chebyshev roots (z=cos(pi*k/(n.z-1)) returns the Chebyshev coefficients in Z for each wavenumber of the input in Fourier space in the plane directions.
      Container must hold GPU accessible memory.
      Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k).
      Returns a new vector, leaving the input unmodified.

.. cpp:function:: template<class Container>  auto chebyshevTransform1DCufft(Container fx);

      From a 1D field of complex values evaluated at Chebyshev roots (z=cos(pi*k/(n.z-1)) returns their Chebyshev coefficients.
      Container must hold GPU accessible memory.
      Returns a new vector, leaving the input unmodified.


.. cpp:function::    template<class Container>   auto inverseFourierChebyshevTransform3DCufft(Container fn, int3 n);


    From the complex Chebyshev coeffients of a series of signals (each signal assigned to a 2D wave number in Fourier space) returns the (complex valued) inverse transform in the plane directions and the values of the signal in Z evaluated at the Chebyshev roots (z=cos(pi*k/(n.z-1)).
    Container must hold GPU accessible memory.
    Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k), being i,j wavenumbers and k Chebyshev coeffients
    Returns a new vector, leaving the input unmodified.

.. cpp:function:: template<class Container>   auto inverseChebyshevTransform1DCufft(Container fn, int nz);

    From a group of complex Chebyshev coefficients returns the corresponding signal evaluated at the Chebyshev roots (z=cos(pi*k/(n.z-1))
    Container must hold GPU accessible memory.
    Returns a new vector, leaving the input unmodified.

.. cpp:function:: template<class Container>  auto chebyshevTransform3DCufft(Container fx, int3 n);

    From a group of n.x*n.y batched complex valued signals, each of them evaluated at the Chevyshev roots (z=cos(pi*k/(n.z-1)), returns the Chebyshev coefficients for each signal.
    Container must hold GPU accessible memory.
    Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k), being i,j different signals and k z elements
    Returns a new vector, leaving the input unmodified.


.. cpp:function:: template<class Container> auto inverseChebyshevTransform3DCufft(Container fn, int3 n);

    From a group of n.x*n.y batched signals containing complex Chebyshev coefficients, returns, for each signal, the inverse Chebyshev transform (the values of the function evaluated at the Chebyshev roots).
    Container must hold GPU accessible memory.
    Assumes the element (i,j,k) is located at element id=i+n.x*(j+n.y*k), being i,j different signals and k Chebyshev coefficients
    Returns a new vector, leaving the input unmodified.
