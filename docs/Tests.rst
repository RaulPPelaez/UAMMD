Tests
=====

The folder "test" in the main UAMMD repository contains test scripts for the different solvers. Mostly these consist on a bash script accompanied by a readme and an UAMMD code reproducing a certain known solution. The test script will typically generate messages stating the result of the tests in addition to some figures that can be inspected to evaluate the correctness of the solver.

In order to run them go into the folder of a certain test and run :code:`bash test.bash`. Note that these tests can typically take a lot of time, as averaging is required. Check the test script for a certain solver for more information and customization. You may have to tweak the Makefiles. Finally, test scripts usually include the possibility of using many GPUs.

Recent Unit Tests are written using `GTest <https://google.github.io/googletest/primer.html>`_. In these instances a CMakeLists.txt file is present. To compile and run the tests use the standard steps:

.. code:: bash

	  mkdir build && cd build
	  cmake ..
	  #Solve any dependency issues raised by CMake
	  make test

Note that some tests are stochastic in nature and can thus fail intermittently. Typically rerunning these tests solves the issue.
