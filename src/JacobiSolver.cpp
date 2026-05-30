/**
 * @file JacobiSolver.cpp
 * @brief Implementation of the MPI+OpenMP Jacobi iterative solver.
 */

#include "../include/JacobiSolver.hpp"

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <iostream>


JacobiSolver::JacobiSolver(Grid&                    grid,
                             const BoundaryCondition& bc,
                             const SolverParams&      params)
    : Solver(grid, bc, params)
{}


double JacobiSolver::sweep()
{
    const int    n   = grid_.n();
    const int    lr  = grid_.localRows();
    const double h2  = grid_.h() * grid_.h();
    const double inv4 = 0.25;

    RowMatrix& U    = grid_.U();
    RowMatrix& Unew = grid_.Unew();
    const RowMatrix& F = grid_.F();

    double localErr2 = 0.0;


#pragma omp parallel for reduction(+:localErr2) schedule(static)
    for (int li = 1; li <= lr; ++li)           // li: local row index (1-based)
    {
        for (int j = 1; j < n - 1; ++j)       // j: column (interior only)
        {
            // 5-point stencil:
            //   neighbours from U (previous iteration)
            const double newVal = inv4 * (
                U(li - 1, j) +   // south neighbour (ghost row if li==1)
                U(li + 1, j) +   // north neighbour (ghost row if li==lr)
                U(li, j - 1) +
                U(li, j + 1) +
                F(li - 1, j) * h2
            );

            const double diff = newVal - U(li, j);
            localErr2 += diff * diff;
            Unew(li, j) = newVal;
        }
    }

    return localErr2;
}



void JacobiSolver::solve()
{
    const double h = grid_.h();
    const double tol = params_.tol;
    const int maxIt = params_.maxIter;
    const int rank  = grid_.rank();

    double t0 = MPI_Wtime();

    for (int k = 0; k < maxIt; ++k)
    {
        // 1. Communicate ghost rows with neighbouring ranks.
        grid_.exchangeGhostRows();

        // 2. Perform one Jacobi sweep; collect local squared error.
        double localErr2 = sweep();

        // 3. Reduce the global squared error across all ranks.
        double globalErr2 = 0.0;
        MPI_Allreduce(&localErr2, &globalErr2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Convergence criterion: sqrt(h * sum_ij (U_new - U)^2) < tol
        const double err = std::sqrt(h * globalErr2);

        // 4. Accept the new solution (swap buffers) and re-apply BCs.
        grid_.swapBuffers();
        grid_.applyBoundaryConditions(bc_);

        // 5. Check convergence.
        if (err < tol)
        {
            iterations_ = k + 1;
            error_ = err;
            break;
        }

        // Update iteration count.
        iterations_ = k + 1;
        error_  = err;

        // Optional: periodic progress print on rank 0.
        if (rank == 0 && (k + 1) % 5000 == 0)
        {
            std::cout << "  [Jacobi] iter " << k + 1
                      << "  err = " << err << "\n";
        }
    }

    solveTime_ = MPI_Wtime() - t0;
}