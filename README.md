# peticodiac

In this project, called Peticodiac, we use many-core devices (such as GPUs) to accelerate the primitive operations of the general simplex procedure. The host program is implemented in C++, and the device kernels are implemented in either OpenCL or CUDA.

## Primitive Operations
The primitive operations of the general simplex procedure include:

1. *checkBounds*: finds "broken" variables by checking for bounds violations of the basic variables
 
2. *findSuitable*: finds a suitable non-basic variable whose assignment may be tweaked in order to correct the violation
 
3. *pivot*: swaps the basic (broken) and non-basic (suitable) variables, updates the tableau, and updates the assignment of the broken variable

4. *updateAssignment*: computes the new assignment of all the basic variables

## Work-in-progress

This project is a current work-in-progress. In the near future, additional information will be provided for building and using the application, as well as providing benchmarks and links to other useful resources.
