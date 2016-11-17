# peticodiac

In this project, called Peticodiac, we use many-core devices (such as GPUs) to accelerate the primitive operations of the general simplex procedure. The host program is implemented in C++, and the device kernels are implemented in either OpenCL or CUDA.

## Primitive Operations
The primitive operations of the general simplex procedure include:

1. *check bounds*: finds "broken" variables by checking for bounds violations of the basic variables

2. *find suitable*: finds a suitable non-basic variable whose assignment may be tweaked in order to correct the violation

3. *pivot*: swaps the basic (broken) and non-basic (suitable) variables, updates the tableau, and updates the assignment of the broken variable

4. *update assignment*: computes the new assignment of all the basic variables

## User Guide
If you are using MacOS X, install the GNU gcc6 compiler first.
```
brew install homebrew/versions/gcc6
```

Execute the makefile, and execute the peticodiac program
```
Usage:
./peticodiac NUM_OF_VARIABLES NUM_OF_CONSTRAINTS SOLVER_TYPE
NUM_OF_VARIABLES: specify the number of initial non-basic variables to be create
NUM_OF_CONSTRAINTS: specify the number of initial basic variables to be create
SOLVER_TYPE: specify the solver type
             1: CPU_EAGER
             2: CPU_LAZY
             3: CUDA

Example:
# Determine fesibility for a linear equation with 3 non-basic variables
# and 1 constraint using the CPU-Eager solver
# x0 + x1 + x2 + s0 = 0
# l <= s0 <= u, where l and u are randomly generated
./peticodiac 3 1 1
```

## Work-in-progress

This project is a current work-in-progress. In the near future, additional information will be provided for building and using the application, as well as providing benchmarks and links to other useful resources.
