/**
 * @file Logger.hpp
 * @brief Structured logging to stdout and a .log file, rank-0 only.
 */

#pragma once

#include <fstream>
#include <string>

/**
 * @brief Thread-safe (rank-0-only) logger with file and stdout output.
 *
 * Usage:
 * @code
 * Logger log("results/run.log", rank);
 * log.info("Solver started");
 * log.logConvergence(1000, 1e-7);
 * @endcode
 */
class Logger
{
public:
    /**
     * @brief Construct a Logger.
     * @param filepath  Path to the output .log file.
     * @param rank      MPI rank of this process.
     *                  Only rank 0 produces any output.
     */
    Logger(const std::string& filepath, int rank);

    ~Logger();

    /**
     * @brief Log an informational message.
     * @param msg  Text to log.
     */
    void info(const std::string& msg) const;

    /**
     * @brief Log a section separator line.
     */
    void separator() const;

    /**
     * @brief Log solver configuration parameters.
     * @param method   Solver name (e.g. "jacobi").
     * @param n        Grid size.
     * @param nProcs   Number of MPI processes.
     * @param nThreads Number of OpenMP threads per process.
     * @param tol      Convergence tolerance.
     * @param maxIter  Maximum iterations.
     */
    void logConfig(const std::string& method,
                   int  n,
                   int  nProcs,
                   int  nThreads,
                   double tol,
                   int  maxIter) const;

    /**
     * @brief Log convergence statistics after solve().
     * @param iter   Number of iterations performed.
     * @param error  Final convergence error.
     */
    void logConvergence(int iter, double error) const;

    /**
     * @brief Log timing information.
     * @param label  Description of the timed operation.
     * @param sec    Elapsed time in seconds.
     */
    void logTiming(const std::string& label, double sec) const;

    /**
     * @brief Log the L2 error against the exact solution.
     * @param n    Grid size.
     * @param h    Grid spacing.
     * @param L2   Computed L2 error.
     */
    void logL2Error(int n, double h, double L2) const;

private:
    mutable std::ofstream file_; // Output log file
    int rank_; // MPI rank (only rank 0 writes)

    /**
     * @brief Internal write: outputs to both file and stdout.
     * @param line  Text line (no trailing newline needed).
     */
    void write(const std::string& line) const;

    /**
     * @brief Get current timestamp as a string, e.g. "[12:34:56]".
     */
    static std::string timestamp();
};
