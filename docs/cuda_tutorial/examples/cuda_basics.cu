// Original teaching examples: vector add, block reduction, tiled matrix multiply.
// Run from the repository root using the command in ../README.md.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(error));                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int BLOCK = 256;  // A power of two, required by block_sum below.
constexpr int TILE = 16;

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Launch with exactly BLOCK threads. Each block writes one independent partial.
__global__ void block_sum(const float* x, float* partial, int n) {
    __shared__ float values[BLOCK];
    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK + t;
    values[t] = i < n ? x[i] : 0.0f;
    __syncthreads();

    for (int step = BLOCK / 2; step > 0; step >>= 1) {
        if (t < step) values[t] += values[t + step];
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = values[0];
}

// Contiguous, row-major A[M,K], B[K,N], C[M,N]. Launch TILE x TILE threads.
__global__ void matmul_tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int base = 0; base < k; base += TILE) {
        as[ty][tx] = row < m && base + tx < k ? a[row * k + base + tx] : 0.0f;
        bs[ty][tx] = base + ty < k && col < n ? b[(base + ty) * n + col] : 0.0f;
        __syncthreads();  // Every tile element must be loaded before it is used.
        for (int j = 0; j < TILE; ++j) acc += as[ty][j] * bs[j][tx];
        __syncthreads();  // Every reader must finish before the next tile overwrites it.
    }
    if (row < m && col < n) c[row * n + col] = acc;
}

void check_close(double actual, double expected, const char* label, int index) {
    double tolerance = 1e-4 + 1e-4 * std::abs(expected);
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s[%d]: got %.9g, expected %.9g\n", label, index, actual, expected);
        std::exit(EXIT_FAILURE);
    }
}

void run_vector_add(int n, bool benchmark) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 97 - 48) * 0.125f;
        b[i] = (i % 31 - 15) * 0.25f;
    }
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    int blocks = (n + BLOCK - 1) / BLOCK;
    vector_add<<<blocks, BLOCK>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) check_close(c[i], a[i] + b[i], "vector_add", i);

    if (benchmark) {
        for (int i = 0; i < 10; ++i) {
            vector_add<<<blocks, BLOCK>>>(d_a, d_b, d_c, n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        constexpr int repeats = 100;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) {
            vector_add<<<blocks, BLOCK>>>(d_a, d_b, d_c, n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::printf("vector_add n=%d: %.6f ms/launch (device interval, excludes copies)\n",
                    n, elapsed_ms / repeats);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void run_sum(int n) {
    int blocks = (n + BLOCK - 1) / BLOCK;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    size_t partial_bytes = static_cast<size_t>(blocks) * sizeof(float);
    std::vector<float> x(n), partial(blocks);
    double expected = 0;
    for (int i = 0; i < n; ++i) {
        x[i] = (i % 13 - 6) * 0.125f;
        expected += x[i];
    }
    float *d_x, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, partial_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    block_sum<<<blocks, BLOCK>>>(d_x, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, partial_bytes, cudaMemcpyDeviceToHost));
    double actual = 0;
    for (int block = 0; block < blocks; ++block) {
        double expected_partial = 0;
        for (int t = 0; t < BLOCK && block * BLOCK + t < n; ++t) {
            expected_partial += x[block * BLOCK + t];
        }
        check_close(partial[block], expected_partial, "block_sum partial", block);
        actual += partial[block];
    }
    check_close(actual, expected, "block_sum total", 0);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_partial));
}

void run_matmul(int m, int n, int k) {
    std::vector<float> a(m * k), b(k * n), c(m * n);
    for (int i = 0; i < m * k; ++i) a[i] = (i % 11 - 5) * 0.125f;
    for (int i = 0; i < k * n; ++i) b[i] = (i % 7 - 3) * 0.25f;
    size_t a_bytes = a.size() * sizeof(float);
    size_t b_bytes = b.size() * sizeof(float);
    size_t c_bytes = c.size() * sizeof(float);
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));
    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    matmul_tiled<<<blocks, threads>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double expected = 0;
            for (int j = 0; j < k; ++j) expected += double(a[row * k + j]) * b[j * n + col];
            check_close(c[row * n + col], expected, "matmul", row * n + col);
        }
    }
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, matmul_tiled));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, matmul_tiled, TILE * TILE, 0));
    int warps_per_block = (TILE * TILE + prop.warpSize - 1) / prop.warpSize;
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    std::printf("Device: %s, SMs=%d, warp=%d\n", prop.name, prop.multiProcessorCount, prop.warpSize);
    std::printf("Per SM: max threads=%d, registers(32-bit)=%d, shared bytes=%zu\n",
                prop.maxThreadsPerMultiProcessor, prop.regsPerMultiprocessor,
                prop.sharedMemPerMultiprocessor);
    std::printf("matmul_tiled: threads/block=%d, registers/thread=%d, static shared bytes=%zu, local bytes/thread=%zu\n",
                TILE * TILE, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes);
    std::printf("Capacity estimate: blocks/SM=%d, resident warps/SM=%d, theoretical occupancy=%.1f%%\n",
                active_blocks, active_blocks * warps_per_block,
                100.0 * active_blocks * warps_per_block / max_warps_per_sm);
    std::puts("Capacity is not a measured schedule or throughput; small grids may leave SMs idle.");

    for (int n : {1, 31, 32, 255, 256, 257, 1003}) run_vector_add(n, false);
    std::puts("PASS vector_add: boundary sizes checked");
    for (int n : {1, 31, 32, 255, 256, 257, 1003}) run_sum(n);
    std::puts("PASS block_sum: per-block and total results checked");
    run_matmul(1, 1, 1);
    run_matmul(16, 16, 16);
    run_matmul(17, 19, 23);
    run_matmul(31, 7, 33);
    std::puts("PASS matmul_tiled: aligned and ragged matrices checked");
    run_vector_add((1 << 20) + 3, true);
    return EXIT_SUCCESS;
}
