Tests
=====

The folder "test" in the main UAMMD repository contains test scripts for the different solvers. Mostly these consist on a bash script accompanied by a readme and an UAMMD code reproducing a certain known solution. The test script will typically generate messages stating the result of the tests in addition to some figures that can be inspected to evaluate the correctness of the solver.

In order to run them simply go into the folder of a certain test and run :code:`bash test.bash`. Note that these tests can typically take a lot of time, as averaging is required. Check the test script for a certain solver for more information and customization. You may have to tweak the Makefiles. Finally, test scripts usually include the possibility of using many GPUs.
