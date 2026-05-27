# challenge3 - Initial Idea and project structure
OpenFOAM style implementation: the idea is to run the solver from any location on PCs where the software is installed. Typical run folder must contain: a file BC.json, with boundary conditions on top, bottom, left and right boundaries; a file setup.json containing the function f definition, number of nodes, number of processors and solution method (jacobi or schwartz); run.sh (optional) script to run the software with the given settings.
The output must be a results folder with the VTKs of the solution and a .log file with useful informations for benchmarking. 

Code structure:

laplace-solver/
├── include/
│   ├── BoundaryCondition.hpp
│   ├── Grid.hpp
│   ├── Solver.hpp
│   ├── JacobiSolver.hpp
│   ├── SchwarzSolver.hpp
│   ├── VTKExporter.hpp
│   ├── Logger.hpp
│   └── Utils.hpp
├── src/
│   ├── BoundaryCondition.cpp
│   ├── Grid.cpp
│   ├── Solver.cpp
│   ├── JacobiSolver.cpp
│   ├── SchwarzSolver.cpp
│   ├── VTKExporter.cpp
│   ├── Logger.cpp
│   ├── Utils.cpp
│   └── main.cpp
├── test/
│   └── (benchmark cases, after)
├── BC.json
├── setup.json
├── run.sh
├── Makefile
├── .gitignore
└── README.md

Solver is abstract class, JacobiSolver and SchwartzSolver derived classes.
Utils contains several free support functions (es, json reader, parser setup).
