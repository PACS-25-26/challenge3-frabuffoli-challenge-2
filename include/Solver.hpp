/**
 * @file Solver.hpp
 * @brief Abstract base class for iterative solvers.
 */

#pragma once

#include <memory>
#include <string>
#include "Grid.hpp"
#include "BoundaryCondition.hpp"

/**
 * @brief Parameters common to all solvers, read from setup.json.
 */
struct SolverParams
{
    double tol      = 1e-6;   // Convergence tolerance
    int    maxIter  = 1000; // Maximum number of iterations
};

/**
 * @brief Abstract interface for iterative solvers.
 */
class Solver
{
public:
    /**
     * @brief Construct a solver.
     * @param grid    Reference to the distributed grid.
     * @param bc      Boundary conditions.
     * @param params  Convergence parameters.
     */
    Solver(Grid& grid, const BoundaryCondition& bc, const SolverParams& params);

    virtual ~Solver() = default;

    /**
     * @brief Execute the iterative solution procedure.
     */
    virtual void solve() = 0;

    /** @brief Number of iterations performed. */
    int    iterations() const { return iterations_; }

    /** @brief Final convergence error. */
    double error()      const { return error_; }

    /** @brief Wall-clock time in seconds spent in solve(). */
    double solveTime()  const { return solveTime_; }

    /**
     * @brief Factory method: create a solver by name.
     *
     * @param type    @c "jacobi" or @c "schwarz"
     * @param grid    Reference to the distributed grid.
     * @param bc      Boundary conditions.
     * @param params  Convergence parameters.
     * @return        Owning pointer to the concrete solver.
     * @throws std::invalid_argument for unknown type strings.
     */
    static std::unique_ptr<Solver> create(const std::string&      type,
                                          Grid&                   grid,
                                          const BoundaryCondition& bc,
                                          const SolverParams&     params);

protected:
    Grid&                   grid_;     // Distributed grid reference
    const BoundaryCondition& bc_;     // Boundary conditions
    SolverParams            params_;   // Tolerance / max-iter
    int                     iterations_ = 0;   // Iteration counter
    double                  error_      = 0.0; // Convergence error
    double                  solveTime_  = 0.0; // Wall-clock time [s]
};
