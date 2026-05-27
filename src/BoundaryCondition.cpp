/**
 * @file BoundaryCondition.cpp
 * @brief Implementation of BoundaryCondition: JSON loading and evaluation.
 */

#include "BoundaryCondition.hpp"
#include "Utils.hpp"

#include <sstream>
#include <stdexcept>


static BCType bcTypeFromString(const std::string& s)
{
    if (s == "dirichlet") return BCType::Dirichlet;
    if (s == "neumann")   return BCType::Neumann;
    if (s == "robin")     return BCType::Robin;
    throw std::invalid_argument("BoundaryCondition: unknown BC type " + s );
}

static std::string bcTypeToString(BCType t)
{
    switch (t)
    {
        case BCType::Dirichlet: return "Dirichlet";
        case BCType::Neumann:   return "Neumann";
        case BCType::Robin:     return "Robin";
    }
    return "Unknown";
}


void BoundaryCondition::parseSide(const nlohmann::json& j, SideBC& side)
{
    side.type  = bcTypeFromString(j.at("type").get<std::string>());
    side.expr  = j.at("expr").get<std::string>();
    side.alpha = j.value("alpha", 0.0); // optional, default 0
}


void BoundaryCondition::loadFromFile(const std::string& path)
{
    auto j = Utils::readJson(path);

    parseSide(j.at("top"),    top_);
    parseSide(j.at("bottom"), bottom_);
    parseSide(j.at("left"),   left_);
    parseSide(j.at("right"),  right_);
}


const SideBC& BoundaryCondition::getSideBC(Side side) const
{
    switch (side)
    {
        case Side::Top:    return top_;
        case Side::Bottom: return bottom_;
        case Side::Left:   return left_;
        case Side::Right:  return right_;
    }
    throw std::logic_error("BoundaryCondition: invalid Side enum value");
}


double BoundaryCondition::evaluate(Side side, double coord) const
{
    const SideBC& s = getSideBC(side);
    return Utils::evalExpression(s.expr, coord, coord);
}


BCType BoundaryCondition::getType(Side side) const
{
    return getSideBC(side).type;
}


double BoundaryCondition::getAlpha(Side side) const
{
    return getSideBC(side).alpha;
}


std::string BoundaryCondition::summary() const
{
    auto describe = [](const char* name, const SideBC& s) -> std::string {
        std::ostringstream oss;
        oss << "  " << name << ": " << bcTypeToString(s.type)
            << "  expr=\"" << s.expr << "\"";
        if (s.type == BCType::Robin)
            oss << "  alpha=" << s.alpha;
        return oss.str();
    };

    return std::string("Boundary Conditions:\n") +
           describe("Top   ", top_)    + "\n" +
           describe("Bottom", bottom_) + "\n" +
           describe("Left  ", left_)   + "\n" +
           describe("Right ", right_);
}
