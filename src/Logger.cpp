/**
 * @file Logger.cpp
 * @brief Structured logger implementation.
 */

#include "../include/Logger.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>
#include <stdexcept>


Logger::Logger(const std::string& filepath, int rank)
    : rank_(rank)
{
    if (rank_ == 0)
    {
        file_.open(filepath, std::ios::out | std::ios::trunc);
        if (!file_.is_open())
            throw std::runtime_error("Logger: cannot open log file: " + filepath);
    }
}

Logger::~Logger()
{
    if (file_.is_open())
        file_.close();
}


void Logger::write(const std::string& line) const
{
    if (rank_ != 0) return;

    const std::string out = timestamp() + "  " + line;
    std::cout << out << "\n";
    file_    << out << "\n";
    file_.flush();
}


std::string Logger::timestamp()
{
    using namespace std::chrono;
    auto now   = system_clock::now();
    auto time  = system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    std::ostringstream oss;
    oss << "[" << std::setfill('0')
        << std::setw(2) << tm.tm_hour << ":"
        << std::setw(2) << tm.tm_min  << ":"
        << std::setw(2) << tm.tm_sec  << "]";
    return oss.str();
}


void Logger::info(const std::string& msg) const
{
    write(msg);
}

void Logger::separator() const
{
    write(std::string(60, '-'));
}

void Logger::logConfig(const std::string& method,
                        int    n,
                        int    nProcs,
                        int    nThreads,
                        double tol,
                        int    maxIter) const
{
    separator();
    write("Laplace Solver – Configuration");
    separator();
    write("  Solver method   : " + method);
    write("  Grid size       : " + std::to_string(n) + " x " + std::to_string(n));
    write("  Grid spacing h  : " + std::to_string(1.0 / (n - 1)));
    write("  MPI processes   : " + std::to_string(nProcs));
    write("  OpenMP threads  : " + std::to_string(nThreads));
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2) << tol;
    write("  Tolerance       : " + oss.str());
    write("  Max iterations  : " + std::to_string(maxIter));
    separator();
}

void Logger::logConvergence(int iter, double error) const
{
    std::ostringstream oss;
    oss << "Convergence after " << iter << " iteration(s)"
        << "  |  final error = " << std::scientific << std::setprecision(4) << error;
    write(oss.str());
}

void Logger::logTiming(const std::string& label, double sec) const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << label << " : " << sec << " s";
    write(oss.str());
}

void Logger::logL2Error(int n, double h, double L2) const
{
    std::ostringstream oss;
    oss << "L2 error vs exact solution"
        << "  |  n=" << n
        << "  h=" << std::fixed << std::setprecision(5) << h
        << "  L2=" << std::scientific << std::setprecision(6) << L2;
    write(oss.str());
}
