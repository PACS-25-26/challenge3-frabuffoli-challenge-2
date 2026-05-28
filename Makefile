# =============================================================================
# Makefile – Parallel Laplace Solver (MPI + OpenMP + Eigen)
# =============================================================================
#
# Usage:
#   make            – build the solver executable
#   make clean      – remove object files and executable
#   make distclean  – also remove the results directory
#   make docs       – generate Doxygen HTML documentation
#
# The Makefile auto-detects the Eigen include path via pkg-config.
# If Eigen is not installed system-wide, set EIGEN_INC manually:
#   make EIGEN_INC=/path/to/eigen
# =============================================================================

# ── Compiler and flags ────────────────────────────────────────────────────────
CXX      := mpicxx
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fopenmp

# ── Eigen include path ────────────────────────────────────────────────────────
# Try pkg-config first; fall back to common system paths.
EIGEN_INC ?= $(shell pkg-config --cflags eigen3 2>/dev/null | sed 's/-I//')
ifeq ($(EIGEN_INC),)
    # Fallback search
    EIGEN_CANDIDATES := /usr/include/eigen3 \
                        /usr/local/include/eigen3 \
                        $(HOME)/include/eigen3
    EIGEN_INC := $(firstword $(foreach d,$(EIGEN_CANDIDATES),$(wildcard $(d))))
endif

ifeq ($(EIGEN_INC),)
    $(error "Eigen3 not found. Set EIGEN_INC=/path/to/eigen or install via 'apt install libeigen3-dev'")
endif

# ── Include paths ─────────────────────────────────────────────────────────────
INCLUDES := -Iinclude -Iexternal -I$(EIGEN_INC)

# ── Source and object files ───────────────────────────────────────────────────
SRC_DIR := src
OBJ_DIR := build

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

# ── Target ────────────────────────────────────────────────────────────────────
TARGET := laplace-solver

# =============================================================================
# Default target
# =============================================================================
.PHONY: all
all: $(TARGET)
	@echo ""
	@echo "  Build successful → ./$(TARGET)"
	@echo "  Run with:  ./run.sh"
	@echo ""

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lmuparser

# ── Compile each source file ──────────────────────────────────────────────────
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# =============================================================================
# Clean targets
# =============================================================================
.PHONY: clean distclean
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

distclean: clean
	rm -rf results/

# =============================================================================
# Documentation (requires Doxygen)
# =============================================================================
.PHONY: docs
docs:
	doxygen Doxyfile
	@echo "Docs generated in docs/html/index.html"

# =============================================================================
# Info target
# =============================================================================
.PHONY: info
info:
	@echo "CXX      = $(CXX)"
	@echo "CXXFLAGS = $(CXXFLAGS)"
	@echo "EIGEN    = $(EIGEN_INC)"
	@echo "SOURCES  = $(SOURCES)"
