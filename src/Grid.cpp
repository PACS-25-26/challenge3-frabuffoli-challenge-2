/**
 * @file Grid.cpp
 * @brief Implementation of Grid: decomposition, initialisation, ghost-row exchange.
 */

#include "../include/Grid.hpp"
#include "../include/Utils.hpp"

#include <mpi.h>
#include <cmath>
#include <stdexcept>

Grid::Grid(int n, int rank, int size)
    : n_(n), h_(1.0 / (n - 1)), rank_(rank), size_(size)
{
    if (n < 3)
        throw std::invalid_argument("Grid: n must be at least 3");
    if (size > n - 2)
        throw std::invalid_argument("Grid: more MPI ranks than interior rows");

    computeDecomposition();

    // Allocate matrices:
    //   U_ and Unew_ have (localRows_+2) rows to accommodate ghost rows.
    //   F_ has only localRows_ rows (no ghost rows needed for the forcing term).
    U_ = Eigen::MatrixXd::Zero(localRows_ + 2, n_);
    Unew_ = Eigen::MatrixXd::Zero(localRows_ + 2, n_);
    F_ = Eigen::MatrixXd::Zero(localRows_, n_);
}


void Grid::computeDecomposition()
{
    // Total interior rows (excluding the two boundary rows y=0 and y=1).
    const int interiorRows = n_ - 2;

    const int base = interiorRows / size_;
    const int remainder = interiorRows % size_;

    // Ranks 0 .. remainder-1 own (base+1) rows; the rest own base rows.
    localRows_ = (rank_ < remainder) ? base + 1 : base;

    // Compute the global starting row index (0-based, including boundary rows).
    // Row 0 is the bottom boundary (y=0); interior rows start at global index 1.
    int offset = 0;
    for (int r = 0; r < rank_; ++r)
        offset += (r < remainder) ? base + 1 : base;

    rowStart_ = 1 + offset;          // First interior row owned by this rank
    rowEnd_  = rowStart_ + localRows_ - 1;
}


void Grid::initialize(const BoundaryCondition& bc, const std::string& fExpr)
{
    // Set U_ to zero everywhere (includes ghost rows).
    U_.setZero();
    Unew_.setZero();

    // Evaluate the forcing term on each locally owned interior node.
    for (int li = 1; li <= localRows_; ++li)         // local row (1-based, excluding ghost)
    {
        const int gi = globalRow(li);                 // global row index
        const double y = coordY(gi);
        for (int j = 0; j < n_; ++j)
        {
            const double x = coordX(j);
            F_(li - 1, j) = Utils::evalExpression(fExpr, x, y);
        }
    }

    // Apply boundary conditions.
    applyBoundaryConditions(bc);
}


void Grid::applyBoundaryConditions(const BoundaryCondition& bc)
{
    const double h = h_;
    for (int li = 1; li <= localRows_; ++li)
    {
        const int  gi = globalRow(li);
        const double y  = coordY(gi);

        // Left boundary
        switch (bc.getType(Side::Left))
        {
            case BCType::Dirichlet:
                U_(li, 0) = bc.evaluate(Side::Left, y);
                break;
            case BCType::Neumann:
            {
                // Forward difference: (U(li,1) - U(li,0)) / h = g  ->  U(li,0) = U(li,1) - h*g
                double g = bc.evaluate(Side::Left, y);
                U_(li, 0) = U_(li, 1) - h * g;
                break;
            }
            case BCType::Robin:
            {
                // (U(li,1) - U(li,0)) / h + alpha * U(li,0) = g
                // U(li,0) = (U(li,1)/h - g) / (1/h - alpha)
                double g     = bc.evaluate(Side::Left, y);
                double alpha = bc.getAlpha(Side::Left);
                U_(li, 0) = (U_(li, 1) / h - g) / (1.0 / h - alpha);
                break;
            }
        }

        // Right boundary
        const int jR = n_ - 1;
        switch (bc.getType(Side::Right))
        {
            case BCType::Dirichlet:
                U_(li, jR) = bc.evaluate(Side::Right, y);
                break;
            case BCType::Neumann:
            {
                double g = bc.evaluate(Side::Right, y);
                U_(li, jR) = U_(li, jR - 1) + h * g;
                break;
            }
            case BCType::Robin:
            {
                double g     = bc.evaluate(Side::Right, y);
                double alpha = bc.getAlpha(Side::Right);
                U_(li, jR) = (U_(li, jR - 1) / h + g) / (1.0 / h + alpha);
                break;
            }
        }
    }

    // Bottom (y=0, global row 0)
    if (isBottom())
    {
        // For Dirichlet, row index 0 in U_ corresponds to global row 0 only
        // if this rank owns it as a ghost row (which it does: ghost row 0
        // represents global row rowStart_-1 = 0 for rank 0).
        // We store the BC directly in the ghost-row slot U_(0, j).
        for (int j = 0; j < n_; ++j)
        {
            const double x = coordX(j);
            switch (bc.getType(Side::Bottom))
            {
                case BCType::Dirichlet:
                    U_(0, j) = bc.evaluate(Side::Bottom, x);
                    break;
                case BCType::Neumann:
                {
                    double g = bc.evaluate(Side::Bottom, x);
                    U_(0, j) = U_(1, j) - h * g;
                    break;
                }
                case BCType::Robin:
                {
                    double g     = bc.evaluate(Side::Bottom, x);
                    double alpha = bc.getAlpha(Side::Bottom);
                    U_(0, j) = (U_(1, j) / h - g) / (1.0 / h - alpha);
                    break;
                }
            }
        }
    }

    // Top (y=1, global row n-1)
    if (isTop())
    {
        // Ghost row for the top boundary sits at index localRows_+1 in U_.
        for (int j = 0; j < n_; ++j)
        {
            const double x = coordX(j);
            switch (bc.getType(Side::Top))
            {
                case BCType::Dirichlet:
                    U_(localRows_ + 1, j) = bc.evaluate(Side::Top, x);
                    break;
                case BCType::Neumann:
                {
                    double g = bc.evaluate(Side::Top, x);
                    U_(localRows_ + 1, j) = U_(localRows_, j) + h * g;
                    break;
                }
                case BCType::Robin:
                {
                    double g     = bc.evaluate(Side::Top, x);
                    double alpha = bc.getAlpha(Side::Top);
                    U_(localRows_ + 1, j) = (U_(localRows_, j) / h + g) / (1.0 / h + alpha);
                    break;
                }
            }
        }
    }

    // Copy BCs into Unew_ as well, so they are preserved after swapBuffers().
    Unew_ = U_;
}


void Grid::exchangeGhostRows()
{
    // Row 1 (first owned) is sent to rank-1 as its bottom ghost.
    // Row localRows_ (last owned) is sent to rank+1 as its top ghost.
    // We receive:
    //   - Row 0 (bottom ghost) from rank-1   -> rank-1's last owned row
    //   - Row localRows_+1 (top ghost) from rank+1 -> rank+1's first owned row

    MPI_Status status;

    const int tag_down = 0; // send downward (to rank-1)
    const int tag_up   = 1; // send upward   (to rank+1)

    // Send first owned row down; receive top ghost from rank+1
    {
        double* sendBuf = U_.row(1).data();
        double* recvBuf = U_.row(localRows_ + 1).data();

        int dest   = (rank_ < size_ - 1) ? rank_ + 1 : MPI_PROC_NULL;
        int source = (rank_ < size_ - 1) ? rank_ + 1 : MPI_PROC_NULL;

        MPI_Sendrecv(
            sendBuf, n_, MPI_DOUBLE, dest,   tag_down,
            recvBuf, n_, MPI_DOUBLE, source, tag_down,
            MPI_COMM_WORLD, &status);
    }

    // Send last owned row up; receive bottom ghost from rank-1
    {
        double* sendBuf = U_.row(localRows_).data();
        double* recvBuf = U_.row(0).data();

        int dest   = (rank_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
        int source = (rank_ > 0) ? rank_ - 1 : MPI_PROC_NULL;

        MPI_Sendrecv(
            sendBuf, n_, MPI_DOUBLE, dest,   tag_up,
            recvBuf, n_, MPI_DOUBLE, source, tag_up,
            MPI_COMM_WORLD, &status);
    }
}


void Grid::swapBuffers()
{
    U_.swap(Unew_);
}
