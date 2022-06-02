Vector types
============

To ease writing code independent of the precision (:cpp:`float` or :cpp:`double`) UAMMD offers a series of vector types.

.. cpp:class:: real

	       An alias to either :cpp:`float` or :cpp:`double` depending on the compilation mode (see :ref:`Compiling UAMMD`)


.. cpp:class:: real2

	       A POD structure packing two real numbers (has two public members called x and y)



.. cpp:class:: real3

	       A POD structure packing three real numbers (has three public members called x, y and z)


.. cpp:class:: real4

	       A POD structure packing four real numbers (has three public members called x, y, z and w)


All the vector types have algebraic operator overloads (+,-,*,/, etc) acting elementwise. 
Many mathematical functions are also overloaded for UAMMD's vector types (sqrt, cos, sin,...), always acting elementwise.
Furthermore, several vector operations are also available (such as dot, cross, norm,...).

Example
----------


.. code:: c++

   real4 a = {1,2,3,4};
   real4 b = {1,2,3,4};

   real4 c = a*b; //Stores {1,4,9,16}

   real4 d = sqrt(c); //Stores {1,2,3,4}
   
   
