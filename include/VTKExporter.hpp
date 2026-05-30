/**
 * @file VTKExporter.hpp
 * @brief Export the global solution to VTK format (structured points).
 *
 * Only rank 0 performs the actual write; all ranks participate in the
 * MPI_Gather call.
 */

#pragma once

#include <string>
#include "Grid.hpp"

/**
 * @brief Exporter class for VTK structured-points files.
 *
 * Typical usage:
 * @code
 * VTKExporter exp("results");
 * exp.exportSolution(grid, "solution");
 * @endcode
 */
class VTKExporter
{
public:
    /**
     * @brief Construct a VTKExporter.
     * @param outputDir  Directory where VTK files will be written.
     *                   The directory is created if it does not exist (rank 0 only).
     */
    explicit VTKExporter(const std::string& outputDir);

    /**
     * @brief Gather the distributed solution and write a VTK file.
     *
     * Each rank sends its owned rows (rows 1..localRows of U_, excluding
     * ghost rows) to rank 0 via MPI_Gather.  Rank 0 then reconstructs
     * the full nxn matrix and calls writeVTK().
     *
     * @param grid      The local distributed grid.
     * @param filename  Base filename (without extension), e.g. @c "solution".
     *                  The file is written as @c outputDir/filename.vtk.
     */
    void exportSolution(const Grid& grid, const std::string& filename) const;

    /**
     * @brief Also export the exact solution for comparison.
     *
     * Evaluates @p exactExpr at every grid point and writes the result.
     *
     * @param grid       Local distributed grid (used for dimensions/spacing).
     * @param exactExpr  Mathematical expression for the exact solution u(x,y).
     * @param filename   Base filename, e.g. @c "exact".
     */
    void exportExact(const Grid& grid,
                     const std::string& exactExpr,
                     const std::string& filename) const;

private:
    std::string outputDir_; // Output directory

    /**
     * @brief Write a full nxn matrix to a VTK structured-points file.
     *
     * Called only on rank 0.
     *
     * @param data      Row-major nxn matrix (row 0 = y=0 boundary).
     * @param n         Grid size.
     * @param h         Grid spacing.
     * @param fieldName Name of the scalar field in the VTK file.
     * @param filepath  Full output file path.
     */
    void writeVTK(const RowMatrix& data,
                  int                    n,
                  double                 h,
                  const std::string&     fieldName,
                  const std::string&     filepath) const;
};
