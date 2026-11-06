# Thin passthrough to CMake, so `make` and `make test` keep working.
# See README.md for the CMake commands directly.
BUILD ?= build
CMAKE_ARGS ?=

.PHONY: all test gpu gpu-test clean

all:
	cmake -B $(BUILD) $(CMAKE_ARGS) && cmake --build $(BUILD) -j

test: all
	ctest --test-dir $(BUILD) --output-on-failure

gpu:
	cmake -B $(BUILD)-cuda -DVMC_CUDA=ON $(CMAKE_ARGS) && cmake --build $(BUILD)-cuda -j

gpu-test: gpu
	ctest --test-dir $(BUILD)-cuda --output-on-failure

clean:
	rm -rf $(BUILD) $(BUILD)-cuda
