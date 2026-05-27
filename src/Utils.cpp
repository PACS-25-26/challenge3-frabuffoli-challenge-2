/**
 * @file Utils.cpp
 * @brief Implementation of utility functions using muParser and nlohmann/json.
 */

#include "Utils.hpp"

#include <muParser.h>

#include <fstream>
#include <cmath>
#include <stdexcept>

namespace Utils
{

double evalExpression(const std::string& expr, double x, double y)
{
    double xv = x;
    double yv = y;

    try
    {
        mu::Parser p;
        p.DefineVar("x", &xv);
        p.DefineVar("y", &yv);
        p.DefineConst("pi", M_PI);
        p.DefineConst("e",  M_E);
        p.SetExpr(expr);
        return p.Eval();
    }
    catch (const mu::Parser::exception_type& ex)
    {
        throw std::runtime_error(
            std::string("Utils::evalExpression – muParser error while parsing \"") +
            expr + "\": " + ex.GetMsg());
    }
}


nlohmann::json readJson(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Utils::readJson – cannot open file: " + path);

    nlohmann::json j;
    try
    {
        f >> j;
    }
    catch (const nlohmann::json::parse_error& ex)
    {
        throw std::runtime_error(
            std::string("Utils::readJson – JSON parse error in '") +
            path + "': " + ex.what());
    }
    return j;
}

} // namespace Utils
