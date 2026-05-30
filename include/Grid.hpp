/**
 * @file Grid.hpp
 * @brief Local grid owned by one MPI rank, with ghost-row communication.
 *
 * The global nxn grid is decomposed by rows among @c size MPI processes.
 * Each rank owns a contiguous block of rows [row_start, row_end] of the
 * global grid, plus one ghost row above and one below for neighbour data.
 *
 * Memory layout of U_ (local storage):
 * @verbatim
 *   row 0 -> ghost row from rank-1  (or unused if rank==0)
 *   row 1..local_rows -> actual owned rows
 *   row local_rows+1 -> ghost row from rank+1  (or unused if rank==size-1)
 * @endverbatim
 */

#pragma once

#include <Eigen/Dense>
#include <string>
#include "BoundaryCondition.hpp"

/**
 * @brief Row-major dense matrix type used throughout the solver.
 */
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Distributed grid for the Jacobi/Schwarz solver.
 *
 * Owns the local solution matrix U_, the update U_new_, and the
 * local right-hand-side matrix F_.
 */
class Grid
{
public:
    /**
     * @brief Construct a distributed grid.
     *
     * @param n     Total number of grid points per side (including boundary).
     * @param rank  MPI rank of this process.
     * @param size  Total number of MPI processes.
     */
    Grid(int n, int rank, int size);

    /**
     * @brief Initialise U_ to zero, apply BCs to global boundary rows/cols,
     *        and evaluate the forcing term f on all local interior nodes.
     *
     * @param bc     Boundary conditions descriptor.
     * @param fExpr  Mathematical expression for f(x,y) as a string.
     */
    void initialize(const BoundaryCondition& bc, const std::string& fExpr);

    /**
     * @brief Apply boundary conditions to the local portion of U_.
     *
     * @param bc  Boundary conditions descriptor.
     */
    void applyBoundaryConditions(const BoundaryCondition& bc);

    /**
     * @brief Exchange ghost rows with neighbouring MPI ranks.
     */
    void exchangeGhostRows();

    /**
     * @brief Swap U_ and U_new_ (called after each Jacobi sweep).
     */
    void swapBuffers();


    /** @brief Global number of grid points per side. */
    int n() const { return n_; }

    /** @brief Grid spacing h = 1/(n-1). */
    double h() const { return h_; }

    /** @brief MPI rank of this process. */
    int rank() const { return rank_; }

    /** @brief Total number of MPI processes. */
    int size() const { return size_; }

    /** @brief First global row owned by this rank (0-based, includes boundary). */
    int rowStart() const { return rowStart_; }

    /** @brief Last global row owned by this rank (inclusive). */
    int rowEnd() const { return rowEnd_; }

    /** @brief Number of rows owned (excluding ghost rows). */
    int localRows() const { return localRows_; }

    /**
     * @brief Reference to the local solution matrix (with ghost rows).
     *
     * Dimensions: (localRows_+2) x n_.
     * Row 0 and row localRows_+1 are ghost rows.
     */
    RowMatrix& U() { return U_; }
    const RowMatrix& U() const { return U_; }

    /**
     * @brief Reference to the update buffer (same layout as U_).
     */
    RowMatrix& Unew() { return Unew_; }
    const RowMatrix& Unew() const { return Unew_; }

    /**
     * @brief Reference to the local forcing-term matrix.
     *
     * Dimensions: localRows_ x n_  (no ghost rows).
     */
    const RowMatrix& F() const { return F_; }

    /**
     * @brief Convert a local row index (1-based, excluding ghost rows) to a
     *        global row index.
     * @param localI  Local row index in [1, localRows_].
     */
    int globalRow(int localI) const { return rowStart_ + localI - 1; }

    /** @brief x-coordinate of column j (0-based global column). */
    double coordX(int j) const { return j * h_; }

    /** @brief y-coordinate of global row i (0-based). */
    double coordY(int globalI) const { return globalI * h_; }

    /** @brief True if this rank owns the first global row (y = 0). */
    bool isBottom() const { return rank_ == 0; }

    /** @brief True if this rank owns the last global row (y = 1). */
    bool isTop() const { return rank_ == size_ - 1; }

private:
    int n_;           // Global grid size
    double h_;        // Grid spacing
    int rank_;        // MPI rank
    int size_;        // MPI size
    int rowStart_;    // First global row (inclusive)
    int rowEnd_;      // Last global row (inclusive)
    int localRows_;   // Number of owned rows

    RowMatrix U_;    // Local solution + ghost rows  [(localRows+2) x n]
    RowMatrix Unew_; // Update buffer
    RowMatrix F_;    // Local forcing term           [localRows x n]

    /**
     * @brief Compute the balanced row decomposition.
     *
     * Interior rows are n-2; distributed as evenly as possible:
     *   - ranks 0 .. remainder-1 get (base+1) rows
     *   - ranks remainder .. size-1 get base rows
     */
    void computeDecomposition();
};