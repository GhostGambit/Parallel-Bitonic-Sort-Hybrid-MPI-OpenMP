CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra
INCLUDES = -Iinclude
HEADERS = include/common.hpp include/bitonic_kernels.hpp
BIN = bin
EXE := $(if $(filter Windows_NT,$(OS)),.exe,)

.PHONY: all clean seq_main hybrid_v1 hybrid_v2

all: seq_main hybrid_v1 hybrid_v2

seq_main: $(BIN)/seq_main$(EXE)

hybrid_v1: $(BIN)/hybrid_v1$(EXE)

hybrid_v2: $(BIN)/hybrid_v2$(EXE)

$(BIN)/%$(EXE): src/%.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
