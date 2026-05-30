/**
 * @file JacobiSolver.hpp
 * @brief MPI+OpenMP Jacobi iterative solver for the Laplace equation.
 */

#pragma once

#include "Solver.hpp"

/**
 * @brief Jacobi solver with hybrid MPI/OpenMP parallelism.
 */
class JacobiSolver : public Solver
{
public:
    /**
     * @brief Construct a JacobiSolver.
     * @param grid    Distributed grid.
     * @param bc      Boundary conditions.
     * @param params  Tolerance and max-iter parameters.
     */
    JacobiSolver(Grid& grid, const BoundaryCondition& bc, const SolverParams& params);

    /**
     * @brief Run the Jacobi iteration until convergence or max iterations.
     *
     * Algorithm:
     * -# exchangeGhostRows()  [MPI_Sendrecv]
     * -# Update U_new in parallel  [\#pragma omp parallel for]
     * -# Compute local error  [omp reduction]
     * -# MPI_Allreduce to get global error
     * -# Swap U and U_new
     * -# Re-apply non-Dirichlet BCs
     * -# Check convergence
     */
    void solve() override;

private:
    /**
     * @brief Perform a single Jacobi sweep: fill Unew_ from U_.
     *
     * Only interior nodes of the owned rows are updated.
     * The ghost rows of U_ must already be up-to-date.
     *
     * @return  Local squared-norm of the increment (before dividing by h).
     */
    double sweep();
};
