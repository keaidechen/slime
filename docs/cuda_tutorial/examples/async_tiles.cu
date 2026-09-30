// Two shared-memory slots. Requires an sm_80+ compilation target and compatible GPU.
#include "lab_common.cuh"
#include <cuda_pipeline.h>
#include <vector>

constexpr int B = 128;
constexpr int TILES_PER_BLOCK = 4;

__global__ void async_tiles(const float* x, float* y, int n) {
    __shared__ float slots[2][B];
    int t = threadIdx.x;
    int first = blockIdx.x * TILES_PER_BLOCK * B;
    // Prologue: first tile enters slot 0. Invalid positions are explicitly zeroed.
    if (first + t < n) __pipeline_memcpy_async(&slots[0][t], &x[first + t], sizeof(float));
    else slots[0][t] = 0;
    __pipeline_commit();

    for (int tile = 0; tile < TILES_PER_BLOCK; ++tile) {
        int current = tile % 2;
        // Wait for this thread's committed copies, then publish the whole tile.
        __pipeline_wait_prior(0);
        __syncthreads();
        if (tile + 1 < TILES_PER_BLOCK) {
            int next = 1 - current;
            int index = first + (tile + 1) * B + t;
            if (index < n) __pipeline_memcpy_async(&slots[next][t], &x[index], sizeof(float));
            else slots[next][t] = 0;
            __pipeline_commit();
        }
        // Read a neighbor's value to make cross-thread visibility necessary.
        int index = first + tile * B + t;
        if (index < n) y[index] = slots[current][t] + slots[current][(t + 1) % B];
        __syncthreads();  // All consumers finish before this slot is reused.
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) {
        std::fprintf(stderr, "This async-copy lab requires compute capability >= 8.0.\n");
        return EXIT_FAILURE;
    }
    for (int n : {1, 127, 128, 129, 511, 512, 513, 1031}) {
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; ++i) x[i] = float(i % 29 - 14) * 0.25f;
        float *dx, *dy;
        size_t bytes = size_t(n) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&dx, bytes));
        CUDA_CHECK(cudaMalloc(&dy, bytes));
        CUDA_CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
        async_tiles<<<(n + B * TILES_PER_BLOCK - 1) / (B * TILES_PER_BLOCK), B>>>(dx, dy, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            int neighbor = (i / B) * B + (i % B + 1) % B;
            double expected = x[i] + (neighbor < n ? x[neighbor] : 0.0f);
            check_close(y[i], expected, "async tiles", i);
        }
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }
    std::puts("PASS async double buffer: tile boundaries and zero padding checked");
}
