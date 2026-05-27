/**
 * @file SchwarzSolver.hpp
 * @brief Block-Jacobi (one-level Schwarz) solver for the Laplace equation.
 */

#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include "Solver.hpp"

/**
 * @brief Block-Jacobi / one-level additive Schwarz solver.
 */
class SchwarzSolver : public Solver
{
public:
    /**
     * @brief Construct a SchwarzSolver.
     * @param grid    Distributed grid.
     * @param bc      Boundary conditions.
     * @param params  Tolerance and max-iter.
     */
    SchwarzSolver(Grid& grid, const BoundaryCondition& bc, const SolverParams& params);

    /**
     * @brief Run the Schwarz iteration until convergence or max iterations.
     *
     * Algorithm (outer loop):
     * -# exchangeGhostRows()  [MPI_Sendrecv]
     * -# buildRHS()  – assemble RHS from ghost rows and forcing term
     * -# lu_.solve(rhs_) – apply the pre-factored local LU
     * -# Scatter solution back into grid.U()
     * -# Compute local error and MPI_Allreduce
     * -# Check convergence
     */
    void solve() override;

private:
    using SpMat  = Eigen::SparseMatrix<double>; // Sparse matrix type
    using VecXd  = Eigen::VectorXd;             // Dense vector type
    using SpLU   = Eigen::SparseLU<SpMat>;      // LU factorisation type

    SpMat A_;   // Local stiffness matrix (interior nodes only)
    VecXd rhs_; // Local RHS vector
    SpLU  lu_;  // Pre-factored LU decomposition of A_

    /** @brief Number of locally owned interior rows. */
    int localInteriorRows_;

    /** @brief Number of interior columns (= n-2). */
    int interiorCols_;

    /**
     * @brief Build and factorise the local stiffness matrix A_.
     */
    void buildAndFactorMatrix();

    /**
     * @brief Assemble the local RHS from the current ghost rows and F_.
     *
     * @return  Assembled RHS vector of length localInteriorRows_ x interiorCols_.
     */
    VecXd assembleRHS() const;

    /**
     * @brief Scatter the solution vector back into grid.U().
     * @param sol  Solution vector from lu_.solve().
     */
    void scatterSolution(const VecXd& sol);
};
