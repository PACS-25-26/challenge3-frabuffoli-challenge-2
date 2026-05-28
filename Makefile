# =============================================================================
# Makefile — Parallel Laplace Solver (MPI + OpenMP)
#
# SELF-CONTAINED BUILD: all third-party dependencies are bundled in external/
#   - nlohmann/json : header-only      (external/json.hpp)
#   - Eigen 3.4     : header-only      (external/eigen3/)
#   - muParser 2.3  : compiled from source (external/muparser/)
# The ONLY external requirement is an MPI compiler (mpicxx) with OpenMP.
#
# On a cluster (e.g. CINECA) you do NOT need to install anything or request
# permissions — just load an MPI module and build in your home directory:
#
#     module load openmpi        # or intelmpi, etc. (provides mpicxx)
#     make
#
# Targets:
#   make            build ./laplace-solver
#   make clean      remove build/ and the binary
#   make distclean  also remove results/
# =============================================================================

# ── Compiler ──────────────────────────────────────────────────────────────────
CXX      ?= mpicxx
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -fopenmp

# ── Bundled dependencies (no system libraries needed) ─────────────────────────
EXTERNAL   := external
EIGEN_INC  := $(EXTERNAL)/eigen3
JSON_INC   := $(EXTERNAL)
MUPARSER   := $(EXTERNAL)/muparser
MUPARSER_INC := $(MUPARSER)/include
MUPARSER_SRC := $(MUPARSER)/src

INCLUDES := -Iinclude -I$(JSON_INC) -I$(EIGEN_INC) -I$(MUPARSER_INC)

# ── Project sources ───────────────────────────────────────────────────────────
SRC_DIR := src
OBJ_DIR := build
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

# ── muParser sources (compiled from source, bundled) ──────────────────────────
# Exclude the Windows DLL wrapper and the bundled test driver.
MU_SOURCES := $(filter-out $(MUPARSER_SRC)/muParserDLL.cpp $(MUPARSER_SRC)/muParserTest.cpp, \
                           $(wildcard $(MUPARSER_SRC)/*.cpp))
MU_OBJECTS := $(patsubst $(MUPARSER_SRC)/%.cpp, $(OBJ_DIR)/muparser_%.o, $(MU_SOURCES))

TARGET := laplace-solver

# =============================================================================
.PHONY: all
all: $(TARGET)
	@echo ""
	@echo "  Build OK -> ./$(TARGET)   (fully self-contained, no system libs)"
	@echo ""

$(TARGET): $(OBJECTS) $(MU_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Project objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# muParser objects (compiled once, from bundled source).
# muParser headers are not -Wall clean on every compiler, so silence warnings.
$(OBJ_DIR)/muparser_%.o: $(MUPARSER_SRC)/%.cpp | $(OBJ_DIR)
	$(CXX) -std=c++17 -O3 -w -I$(MUPARSER_INC) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# =============================================================================
.PHONY: clean distclean info
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

distclean: clean
	rm -rf results/

info:
	@echo "CXX       = $(CXX)"
	@echo "CXXFLAGS  = $(CXXFLAGS)"
	@echo "INCLUDES  = $(INCLUDES)"
	@echo "SOURCES   = $(SOURCES)"
	@echo "MU_SOURCES= $(MU_SOURCES)"
