/**
 * @file BoundaryCondition.hpp
 * @brief Boundary condition descriptor for the Laplace solver.
 *
 * Supports Dirichlet, Neumann and Robin conditions on each of the four
 * sides of the unit-square domain, loaded from a JSON file.
 *
 * BC.json expected format:
 * @code{.json}
 * {
 *   "top"    : { "type": "dirichlet", "expr": "0.0" },
 *   "bottom" : { "type": "dirichlet", "expr": "0.0" },
 *   "left"   : { "type": "dirichlet", "expr": "0.0" },
 *   "right"  : { "type": "dirichlet", "expr": "0.0" }
 * }
 * @endcode
 * For Robin conditions an additional @c alpha key must be provided.
 */

#pragma once

#include <string>
#include <array>
#include <stdexcept>
#include "../external/json.hpp"

/**
 * @brief Enumeration of supported boundary condition types.
 */
enum class BCType
{
    Dirichlet, // u = g  on the boundary
    Neumann,   // du/dn = h  on the boundary
    Robin      // du/dn + alpha·u = h  on the boundary
};

/**
 * @brief Enumeration identifying the four sides of the domain.
 */
enum class Side { Top, Bottom, Left, Right };

/**
 * @brief Struct containing one side's boundary data.
 */
struct SideBC
{
    BCType      type  = BCType::Dirichlet; // Type of condition
    std::string expr  = "0.0";             // Value/flux expression (function of x or y)
    double      alpha = 0.0;              // Robin coefficient alpha (ignored for other types)
};

/**
 * @brief Boundary conditions class.
 */
class BoundaryCondition
{
public:

    /**
     * @brief Default constructor - sets all sides to homogeneous Dirichlet.
     */
    BoundaryCondition() = default;

    /**
     * @brief Load boundary conditions from a JSON file.
     * @param path  Path to BC.json.
     * @throws std::runtime_error on IO or parsing errors.
     */
    void loadFromFile(const std::string& path);

    /**
     * @brief Evaluate the boundary value/flux at a point on a given side.
     *
     * For horizontal sides (Top/Bottom) the free coordinate is @p coord = [0,1].
     * For vertical sides (Left/Right) the free coordinate is @p coord = [0,1].
     *
     * @param side   Which side of the domain.
     * @param coord  The free coordinate along that side.
     * @return       Evaluated scalar value.
     */
    double evaluate(Side side, double coord) const;

    /**
     * @brief Return the BCType for a given side.
     * @param side  The side of interest.
     */
    BCType getType(Side side) const;

    /**
     * @brief Return the Robin coefficient alpha for a given side.
     *        Returns 0 if the side is not Robin.
     * @param side  The side of interest.
     */
    double getAlpha(Side side) const;

    /**
     * @brief Human-readable description of all BCs (for logging).
     */
    std::string summary() const;

private:
    SideBC top_;    // Top boundary (y = 1)
    SideBC bottom_; // Bottom boundary (y = 0)
    SideBC left_;   // Left boundary (x = 0)
    SideBC right_;  // Right boundary (x = 1)

    /**
     * @brief Parse a single side entry from a JSON object.
     * @param j     JSON object for that side.
     * @param side  Destination SideBC struct.
     */
    static void parseSide(const nlohmann::json& j, SideBC& side);

    /**
     * @brief Return a const reference to the SideBC for a given Side enum.
     */
    const SideBC& getSideBC(Side side) const;
};
