#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Shared error reporting for the three independent GPU bridge exercises.
#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

inline void check_close(double actual, double expected, const char* label, int index) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-4 + 2e-4 * std::abs(expected)) {
        std::fprintf(stderr, "%s[%d]: got %.9g, expected %.9g\n", label, index, actual, expected);
        std::exit(EXIT_FAILURE);
    }
}
