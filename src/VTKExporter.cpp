/**
 * @file VTKExporter.cpp
 * @brief Implementation of VTK legacy structured-points exporter.
 */

#include "../include/VTKExporter.hpp"
#include "../include/Utils.hpp"

#include <mpi.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>   
#include <sys/types.h>
#include <iomanip>


VTKExporter::VTKExporter(const std::string& outputDir)
    : outputDir_(outputDir)
{
    // Only rank 0 creates the output directory.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        // Create directory (ignore error if it already exists).
        mkdir(outputDir_.c_str(), 0755);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


void VTKExporter::exportSolution(const Grid& grid, const std::string& filename) const
{
    const int n    = grid.n();
    const double h = grid.h();
    const int rank = grid.rank();
    const int size = grid.size();
    const int lr   = grid.localRows();

    std::vector<double> sendBuf(lr * n);
    for (int li = 1; li <= lr; ++li)
        for (int j = 0; j < n; ++j)
            sendBuf[(li - 1) * n + j] = grid.U()(li, j);

    std::vector<int> rowCounts(size, 0);
    MPI_Gather(&lr, 1, MPI_INT, rowCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(size, 0);
    std::vector<int> recvCounts(size, 0);
    if (rank == 0)
    {
        for (int r = 0; r < size; ++r)
            recvCounts[r] = rowCounts[r] * n;
        for (int r = 1; r < size; ++r)
            displs[r] = displs[r - 1] + recvCounts[r - 1];
    }

    const int totalInteriorRows = n - 2;
    std::vector<double> recvBuf;
    if (rank == 0)
        recvBuf.resize(totalInteriorRows * n, 0.0);

    MPI_Gatherv(sendBuf.data(),  lr * n,       MPI_DOUBLE,
                recvBuf.data(),  recvCounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Reconstruct the full nxn matrix.
        // Row 0 (y=0) and row n-1 (y=1) are boundary rows (all zeros for
        // homogeneous Dirichlet; otherwise copy from BCs).
        RowMatrix Ufull = RowMatrix::Zero(n, n);

        // Interior rows: global rows 1 .. n-2
        for (int r = 0; r < size; ++r)
        {
            // Global row offset for rank r
            int globalStart = 0;
            for (int rr = 0; rr < r; ++rr)
                globalStart += rowCounts[rr];

            for (int li = 0; li < rowCounts[r]; ++li)
            {
                const int gi = 1 + globalStart + li; // global row (1-based interior)
                for (int j = 0; j < n; ++j)
                {
                    Ufull(gi, j) = recvBuf[displs[r] / n * n + li * n + j];
                }
            }
        }

        // Rebuild boundary rows from BCs (zero for homogeneous Dirichlet).
        // If BCs are non-homogeneous, this will need to be passed in.
        // For now we leave row 0 and row n-1 as zero.

        const std::string filepath = outputDir_ + "/" + filename + ".vtk";
        writeVTK(Ufull, n, h, "u", filepath);
    }
}


void VTKExporter::exportExact(const Grid&        grid,
                               const std::string& exactExpr,
                               const std::string& filename) const
{
    const int rank = grid.rank();
    if (rank != 0) return; // Only rank 0 writes

    const int    n = grid.n();
    const double h = grid.h();

    RowMatrix Uexact(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Uexact(i, j) = Utils::evalExpression(exactExpr, j * h, i * h);

    const std::string filepath = outputDir_ + "/" + filename + ".vtk";
    writeVTK(Uexact, n, h, "u_exact", filepath);
}


void VTKExporter::writeVTK(const RowMatrix& data,
                             int                    n,
                             double                 h,
                             const std::string&     fieldName,
                             const std::string&     filepath) const
{
    std::ofstream f(filepath);
    if (!f.is_open())
        throw std::runtime_error("VTKExporter: cannot open file " + filepath);

    // header 
    f << "# vtk DataFile Version 3.0\n";
    f << "Laplace solution\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << n << " " << n << " 1\n";
    f << "ORIGIN 0.0 0.0 0.0\n";
    f << "SPACING " << h << " " << h << " 1.0\n";
    f << "POINT_DATA " << n * n << "\n";
    f << "SCALARS " << fieldName << " double 1\n";
    f << "LOOKUP_TABLE default\n";

    // Write in VTK order: x varies fastest, then y.
    // VTK STRUCTURED_POINTS: point ordering is (x0,y0), (x1,y0), …, (xN-1,yN-1).
    // Our matrix: data(i,j) = U at global row i, column j.
    // Row i corresponds to y = i*h; column j to x = j*h.
    // VTK point index k = i*n + j  (row i from bottom).
    f << std::scientific << std::setprecision(12);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            f << data(i, j) << "\n";
}
