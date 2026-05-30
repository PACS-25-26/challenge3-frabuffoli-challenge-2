/**
 * @file Utils.hpp
 * @brief Utility functions for expression evaluation and JSON parsing.
 *
 * Expression evaluation is delegated to the muParser library.
 */

#pragma once

#include <string>
#include <stdexcept>

#include "../external/json.hpp"

namespace Utils
{

/**
 * @brief Evaluate a mathematical expression string at point (x, y).
 *
 * @param expr  Mathematical expression as a string.
 *              Example: @c "8*pi^2*sin(2*pi*x)*sin(2*pi*y)"
 * @param x     x-coordinate of the evaluation point.
 * @param y     y-coordinate of the evaluation point.
 * @return      Scalar value of the expression at (x, y).
 *
 * @throws std::runtime_error if muParser fails to parse or evaluate @p expr.
 */
double evalExpression(const std::string& expr, double x, double y);

/**
 * @brief Read and parse a JSON file from disk.
 *
 * @param path  Absolute or relative path to the JSON file.
 * @return      Parsed @c nlohmann::json object.
 *
 * @throws std::runtime_error if the file cannot be opened or if the JSON
 *         is malformed.
 */
nlohmann::json readJson(const std::string& path);


} // namespace Utils
