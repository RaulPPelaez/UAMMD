# Raul P. Pelaez 2021. Python wrapper example
# The python bindings in the file python_wrapper.cu are compiled into a library called "uammd"
# This library exposes an UAMMD PairForces Interactor specialized for a LJ potential
# In this example we will see how to use this library from the python side.
# If you want to learn how to expose any UAMMD functionality to python, see python_wrapper.cu
# You will notice that the provided Makefile compiles a python library that prints messages.
# You can easily make the library silent by setting LOGLEVEL to 0 in the Makefile.
import uammd
import numpy as np

# First set up some required parameters
L = 32
box = uammd.Box(L, L, L)
par = uammd.Parameters(sigma=1.0, epsilon=1.0, cutOff=2.5, box=box)

numberParticles = 2
# This object holds all the necessary UAMMD state, this call also initializes the CUDA environment.
lj = uammd.LJ(par, numberParticles)

precision = np.float32  # It is important to use the library precision

# Lets place two particles at a distance sigma from each other
positions = np.zeros((numberParticles, 3), precision)
positions[1][0] = par.sigma

# Forces must be initialized to zero, since the sumForce function below adds to it.
forces = np.zeros((numberParticles, 3), precision)

# Given a list of positions this function will fill "forces" with the lj forces between them
lj.sumForce(positions, forces)

print("The force between one particle at [0 0 0] and another one at [sigma 0 0] is F=[-+24, 0,0]")
print("Contents of position vector")
print(positions)
print("Contents of force vector:")
print(forces)
