/**
 * @file Solver.cpp
 * @brief Base Solver constructor and factory method.
 */

#include "../include/Solver.hpp"
#include "../include/JacobiSolver.hpp"
#include "../include/SchwarzSolver.hpp"

#include <stdexcept>
#include <algorithm>


Solver::Solver(Grid& grid, const BoundaryCondition& bc, const SolverParams& params)
    : grid_(grid), bc_(bc), params_(params)
{}


std::unique_ptr<Solver> Solver::create(const std::string&       type,
                                        Grid&                    grid,
                                        const BoundaryCondition& bc,
                                        const SolverParams&      params)
{
    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);

    if (t == "jacobi")
        return std::make_unique<JacobiSolver>(grid, bc, params);

    if (t == "schwarz")
        return std::make_unique<SchwarzSolver>(grid, bc, params);

    throw std::invalid_argument(
        "Solver::create – unknown solver type \"" + type + "\". "
        "Valid options: jacobi, schwarz");
}
