/**
 * @file main.cpp
 * @brief Entry point for the parallel Laplace solver.
 *
 * Reads setup.json and BC.json from the current working directory,
 * constructs the MPI-distributed grid, runs the chosen solver, exports
 * the VTK solution, and logs convergence / timing / L2 error.
 *
 * Typical invocation (via run.sh):
 * @code
 *   mpirun -np 4 laplace-solver
 * @endcode
 */

#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>
#include <sys/stat.h>

#include "../include/Utils.hpp"
#include "../include/BoundaryCondition.hpp"
#include "../include/Grid.hpp"
#include "../include/Solver.hpp"
#include "../include/VTKExporter.hpp"
#include "../include/Logger.hpp"

// L2 error (distributed)

/**
 * @brief Compute the global L2 error between the numerical and exact solutions.
 *
 * @param grid      Distributed grid (contains the numerical solution in U_).
 * @param exactExpr Mathematical expression for the exact solution u(x,y).
 * @return          Global L2 norm: sqrt(h * sum_ij (u_h - u_exact)^2).
 */
static double computeL2Error(const Grid& grid, const std::string& exactExpr)
{
    const double h  = grid.h();
    const int    lr = grid.localRows();
    const int    n  = grid.n();

    double localErr2 = 0.0;

    for (int li = 1; li <= lr; ++li)
    {
        const int    gi = grid.globalRow(li);
        const double y  = grid.coordY(gi);

        for (int j = 0; j < n; ++j)
        {
            const double x    = grid.coordX(j);
            const double uNum = grid.U()(li, j);
            const double uEx  = Utils::evalExpression(exactExpr, x, y);
            const double diff = uNum - uEx;
            localErr2 += diff * diff;
        }
    }

    double globalErr2 = 0.0;
    MPI_Allreduce(&localErr2, &globalErr2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(h * globalErr2);
}

// main 

int main(int argc, char** argv)
{
    // MPI initialisation 
    MPI_Init(&argc, &argv);

    int rank, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    try
    {
        // Read setup.json 
        auto setup = Utils::readJson("setup.json");

        const int         n         = setup.at("n").get<int>();
        const std::string fExpr     = setup.at("f").get<std::string>();
        const std::string method    = setup.value("solver", "jacobi");
        const std::string exactExpr = setup.value("exact", "");
        const double      tol       = setup.value("tol",      1e-6);
        const int         maxIter   = setup.value("max_iter", 100000);
        const int         nThreads  = setup.value("omp_threads", 1);

        // Set OpenMP thread count.
        omp_set_num_threads(nThreads);

        // Read BC.json 
        BoundaryCondition bc;
        bc.loadFromFile("BC.json");

        // Ensure results directory exists 
        if (rank == 0)
            mkdir("results", 0755);
        MPI_Barrier(MPI_COMM_WORLD);

        // Logger
        Logger logger("results/run.log", rank);

        logger.logConfig(method, n, nProcs, nThreads, tol, maxIter);
        if (rank == 0)
            std::cout << bc.summary() << "\n";

        // Build grid and initialise 
        Grid grid(n, rank, nProcs);
        grid.initialize(bc, fExpr);

        logger.info("Grid initialised  |  rank " + std::to_string(rank) +
                    " owns rows [" + std::to_string(grid.rowStart()) +
                    ", " + std::to_string(grid.rowEnd()) + "]");

        MPI_Barrier(MPI_COMM_WORLD);

        // Solve 
        SolverParams params;
        params.tol     = tol;
        params.maxIter = maxIter;

        auto solver = Solver::create(method, grid, bc, params);

        logger.info("Starting " + method + " solver …");
        solver->solve();

        logger.logConvergence(solver->iterations(), solver->error());
        logger.logTiming("Solve time", solver->solveTime());

        // L2 error 
        if (!exactExpr.empty())
        {
            double L2 = computeL2Error(grid, exactExpr);
            logger.logL2Error(n, grid.h(), L2);
        }

        // VTK export
        VTKExporter exporter("results");
        exporter.exportSolution(grid, "solution");
        logger.info("Solution written to results/solution.vtk");

        if (!exactExpr.empty())
        {
            exporter.exportExact(grid, exactExpr, "exact");
            logger.info("Exact solution written to results/exact.vtk");
        }

        logger.separator();
        logger.info("Done.");
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
            std::cerr << "\n[ERROR] " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}