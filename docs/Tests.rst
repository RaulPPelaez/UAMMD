Tests
=====

The folder "test" in the main UAMMD repository contains unit tests. 

Tests are written using `GTest <https://google.github.io/googletest/primer.html>`_. To compile and run the tests use the standard steps:

.. code:: bash

	  mkdir build && cd build
	  cmake ..
	  #Solve any dependency issues raised by CMake
	  make test
	  

.. warning:: Some tests are stochastic in nature and can thus fail intermittently. Typically rerunning these tests solves the issue.

.. warning:: Some tests can take a lot of time and require a beefy GPU.



.. note:: Some tests are not automatic and not adapted to GTest, instead requiring to manually run a script from within the test folder. Mostly these consist on a bash script accompanied by a readme and an UAMMD code reproducing a certain known solution. The test script will typically generate messages stating the result of the tests in addition to some figures that can be inspected to evaluate the correctness of the solver.
