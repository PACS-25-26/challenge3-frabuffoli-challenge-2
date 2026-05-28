/**
 * @file SchwarzSolver.cpp
 * @brief Block-Jacobi (one-level Schwarz) solver implementation.
 */

#include "../include/SchwarzSolver.hpp"

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <vector>


SchwarzSolver::SchwarzSolver(Grid&                    grid,
                              const BoundaryCondition& bc,
                              const SolverParams&      params)
    : Solver(grid, bc, params)
{
    // Interior columns: j = 1 .. n-2  (exclude left/right boundary walls)
    // Boundary values are computed by applyBoundaryConditions after each iteration,
    // supporting both Dirichlet and Neumann/Robin BC types.
    interiorCols_ = grid_.n() - 2;

    // Interior rows owned by this rank.
    // All localRows_ rows are interior (boundary rows belong to ghost slots).
    localInteriorRows_ = grid_.localRows();

    rhs_.resize(localInteriorRows_ * interiorCols_);

    buildAndFactorMatrix();
}

void SchwarzSolver::buildAndFactorMatrix()
{
    const int nDof = localInteriorRows_ * interiorCols_;
    const double h2inv = 1.0 / (grid_.h() * grid_.h());

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(5 * nDof);

    for (int li = 1; li <= localInteriorRows_; ++li)
    {
        for (int cj = 1; cj <= interiorCols_; ++cj)
        {
            const int idx = (li - 1) * interiorCols_ + (cj - 1);

            trips.emplace_back(idx, idx, 4.0 * h2inv);

            if (li > 1)
            {
                const int idxS = (li - 2) * interiorCols_ + (cj - 1);
                trips.emplace_back(idx, idxS, -h2inv);
            }

            if (li < localInteriorRows_)
            {
                const int idxN = li * interiorCols_ + (cj - 1);
                trips.emplace_back(idx, idxN, -h2inv);
            }

            if (cj > 1)
            {
                const int idxW = (li - 1) * interiorCols_ + (cj - 2);
                trips.emplace_back(idx, idxW, -h2inv);
            }

            if (cj < interiorCols_)
            {
                const int idxE = (li - 1) * interiorCols_ + cj;
                trips.emplace_back(idx, idxE, -h2inv);
            }
        }
    }

    A_.resize(nDof, nDof);
    A_.setFromTriplets(trips.begin(), trips.end());
    A_.makeCompressed();

    lu_.compute(A_);
    if (lu_.info() != Eigen::Success)
        throw std::runtime_error("SchwarzSolver: SparseLU factorisation failed");
}

SchwarzSolver::VecXd SchwarzSolver::assembleRHS() const
{
    const int    n      = grid_.n();
    const double h2inv  = 1.0 / (grid_.h() * grid_.h());
    const RowMatrix& U = grid_.U();
    const RowMatrix& F = grid_.F();

    VecXd rhs(localInteriorRows_ * interiorCols_);
    rhs.setZero();

    for (int li = 1; li <= localInteriorRows_; ++li)
    {
        for (int cj = 1; cj <= interiorCols_; ++cj)
        {
            const int j   = cj;
            const int idx = (li - 1) * interiorCols_ + (cj - 1);

            rhs(idx) = F(li - 1, j);

            // Ghost-row contributions (top/bottom boundaries)
            if (li == 1)
                rhs(idx) += h2inv * U(0, j);

            if (li == localInteriorRows_)
                rhs(idx) += h2inv * U(localInteriorRows_ + 1, j);

            // Left boundary contribution (only for first interior column j=1)
            if (cj == 1)
                rhs(idx) += h2inv * U(li, 0);

            // Right boundary contribution (only for last interior column j=n-2)
            if (cj == interiorCols_)
                rhs(idx) += h2inv * U(li, n - 1);
        }
    }

    return rhs;
}


void SchwarzSolver::scatterSolution(const VecXd& sol)
{
    RowMatrix& U = grid_.U();

    for (int li = 1; li <= localInteriorRows_; ++li)
    {
        for (int cj = 1; cj <= interiorCols_; ++cj)
        {
            const int j   = cj;
            const int idx = (li - 1) * interiorCols_ + (cj - 1);
            U(li, j) = sol(idx);
        }
    }
}


void SchwarzSolver::solve()
{
    const double h     = grid_.h();
    const double tol   = params_.tol;
    const int    maxIt = params_.maxIter;
    const int    rank  = grid_.rank();

    double t0 = MPI_Wtime();

    for (int k = 0; k < maxIt; ++k)
    {
        // 1. Exchange ghost rows with neighbours.
        grid_.exchangeGhostRows();

        // 2. Re-apply boundary conditions BEFORE assembling RHS.
        grid_.applyBoundaryConditions(bc_);

        // 3. Assemble local RHS.
        VecXd rhs = assembleRHS();

        // 3. Solve local sub-system (reuse factorisation).
        VecXd sol = lu_.solve(rhs);
        if (lu_.info() != Eigen::Success)
            throw std::runtime_error("SchwarzSolver: SparseLU solve failed");

        // 4. Compute local increment error before updating U_.
        //    We compare sol against the current interior values of U_.
        double localErr2 = 0.0;
        {
            const RowMatrix& U = grid_.U();

            #pragma omp parallel for reduction(+:localErr2) schedule(static)
            for (int li = 1; li <= localInteriorRows_; ++li)
            {
                for (int cj = 1; cj <= interiorCols_; ++cj)
                {
                    const int j   = cj;
                    const int idx = (li - 1) * interiorCols_ + (cj - 1);
                    const double diff = sol(idx) - U(li, j);
                    localErr2 += diff * diff;
                }
            }
        }

        // 5. Scatter solution into U_.
        scatterSolution(sol);

        // 6. Re-apply boundary conditions AFTER scattering solution.
        grid_.applyBoundaryConditions(bc_);

        // 6. Global convergence check.
        double globalErr2 = 0.0;
        MPI_Allreduce(&localErr2, &globalErr2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        const double err = std::sqrt(h * globalErr2);

        iterations_ = k + 1;
        error_      = err;

        if (err < tol)
            break;

        if (rank == 0 && (k + 1) % 500 == 0)
            std::cout << "  [Schwarz] iter " << k + 1
                      << "  err = " << err << "\n";
    }

    solveTime_ = MPI_Wtime() - t0;
}
