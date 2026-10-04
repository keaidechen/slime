// One warp cooperatively computes one 16x16x16 Tensor Core operation.
#include "lab_common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>

__global__ void wmma_tile(const __half* a, const __half* b, float* c) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    // Pointers and leading dimensions are identical for every lane in this warp.
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::fprintf(stderr, "This WMMA exercise requires a compatible GPU with compute capability >= 7.0.\n");
        return EXIT_FAILURE;
    }
    std::vector<__half> a(256), b(256);
    std::vector<float> c(256);
    for (int i = 0; i < 256; ++i) {
        a[i] = __float2half(float(i % 7 - 3) * 0.25f);
        b[i] = __float2half(float(i % 5 - 2) * 0.5f);
    }
    __half *da, *db;
    float* dc;
    CUDA_CHECK(cudaMalloc(&da, 256 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&db, 256 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dc, 256 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), 256 * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), 256 * sizeof(__half), cudaMemcpyHostToDevice));
    wmma_tile<<<1, 32>>>(da, db, dc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < 16; ++r) {
        for (int col = 0; col < 16; ++col) {
            double reference = 0;
            for (int j = 0; j < 16; ++j) reference += double(__half2float(a[r * 16 + j])) * __half2float(b[j * 16 + col]);
            check_close(c[r * 16 + col], reference, "WMMA", r * 16 + col);
        }
    }
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    std::puts("PASS WMMA 16x16x16: FP16 inputs, FP32 accumulator, CPU reference");
}
