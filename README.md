# efsim

efsim (Embedded Fractures SIMulator) is a code base for simulating flow and transport in porous media, where fractures are modeled as lower-dimensional surfaces embedded in the surrounding media. 

This code base constitutes the computer implementations developed as a part of the PhD project "Reservoir Simulation in Heterogeneous and Fractured Reservoirs", conducted at NTNU, Trondheim, in the period August 2013 - March 2018.

**Disclaimer**: This code base is distributed in the hope that it will be useful, but without *any* warranty.

## Introduction
The code base contains implementations to solve single-phase flow coupled to advective transport of a concentration in heterogeneous and possibly fractured porous media. The solver is sequential with three main steps:

 1. The flow problem (Darcy's equation) is solved by a classical continuous Galerkin finite element method
 2. Locally conservative fluxes are computed by a postprocessing step
 3. The transport problem (advection equation) is solved by a low-order finite volume method with upwinding (or equivalently the lowest order discontinuous Galerkin finite element method)

There are two main application programs:
* **spsim**: Slightly compressible single-phase flow coupled to advective transport in heterogeneous porous media (2/3D)
* **efsim**: Incompressible single-phase flow coupled to advective transport in heterogeneous and fractured porous media (2D)

More details are given in the references listed below.
* LH Odsæter, MF Wheeler, T Kvamsdal, and MG Larson. [Postprocessing of non-conservative flux for compatibility with transport in heterogeneous media](https://www.sciencedirect.com/science/article/pii/S0045782516303565). *Computer Methods in Applied Mechanics and Engineering* 315:799-830 (2017)
* LH Odsæter, T Kvamsdal, MG Larson. [A simple embedded discrete fracture-matrix model for a coupled flow and transport problem in porous media](https://arxiv.org/abs/1803.03423). arXiv:1803.03423 (2018)

Please cite one or both if you are using this code.


## Build instructions
efsim uses CMake and is only verified to work in Unix-environments.

### Prerequisites
efsim is based on the open source finite element library [deal.II](https://www.dealii.org/), which is well documented.
* **deal.II**, version 8.4.1. [Installation instructions](https://www.dealii.org/8.4.1/index.html)
* **trilinos**, version 11.4.1. [Installation instructions](https://www.dealii.org/8.4.1/index.html)

### Build

    mkdir build
    cd build
    cmake -DDEAL_II_DIR=/path/to/deal.II ../
    make
Optionally, 

    make test
will run a series of integration and benchmark tests to check if the code runs correctly.

## Usage

After a successful installation, the build directory should now contain a number of binaries. The two main application programs are `spsim` and `efsim`, see  [Introduction](#Introduction) above. These are run with the commands

    ./spsim input_spsim.prm
or

    ./efsim input_efsim.prm
Examples of input files are found in the base directory `input_files`.  For documentation of the input files, run

    ./spsim -help
    ./efsim -help

**Known pitfall**: The programs assume that the directory they are called from contains a subdirectory `output`, where result files are stored. An exception `dealii::StandardExceptions::ExcIO` is called if not.

## Main classes
The main classes are briefly described below.

|Class name|Usage  |
|--|--|
| PressureSolverBase        | Base class for pressure solvers
| EllipticPressureSolver    | FEM solver for the elliptic (stationary) pressure problem, i.e., incompressible flow |
| ParabolicPressureSolver   | FEM solver for the parabolic (dynamics) pressure problem, i.e., slightly compressible flow. Implicit Euler in time |
| VelocityData              | Stores velocity/flux data |
| PostProcessBase           | Base class for postprocessors to calculate locally conservative fluxes |
| PostProcessMM             | Minimum modification postprocessing |
| PostProcessGS             | Gauss-Seidel postprocessing (deprecated) |
| TransportSolver           | Finite volume solver for advective transport problem. Implicit Euler in time |
| CoupledFlowTransport      | Main driver class for `spsim` |
| EmbeddedSurface           | Representation of fractures as lower-dimensional surfaces |

Problem specific functions are found in `ProblemFunctions.h`