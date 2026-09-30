// Teaching kernels: finite FP32 inputs, contiguous row-major matrices, one head.
// limits[row] is the number of visible keys in a prefix: 0 <= limits[row] <= nk.
// No dropout/backward; online_attention requires 1 <= dv <= THREADS.
#include "lab_common.cuh"
#include <math_constants.h>
#include <algorithm>
#include <limits>
#include <random>
#include <vector>

constexpr int THREADS = 128;
enum class AttentionCase { Dense, Causal, Masked, Rescale };

// All 32 lanes must call this; the combined value is valid in lane 0 only.
template <bool IsMax>
__device__ float warp_reduce(float x) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, x, delta);
        x = IsMax ? fmaxf(x, other) : x + other;
    }
    return x;
}

// Full-warps-only blocks. All threads call, all receive the result.
// The final barrier protects scratch reuse by the next reduction.
template <int B, bool IsMax>
__device__ float block_reduce(float x, float* scratch) {
    static_assert(B >= 32 && B <= 1024 && B % 32 == 0);
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    x = warp_reduce<IsMax>(x);
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        float neutral = IsMax ? -CUDART_INF_F : 0.0f;
        float value = lane < B / 32 ? scratch[lane] : neutral;
        value = warp_reduce<IsMax>(value);
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    float result = scratch[0];
    __syncthreads();
    return result;
}

// B=32 teaches one warp per row; B=128 additionally teaches cross-warp reduction.
template <int B>
__global__ void row_softmax(const float* x, float* p, int cols) {
    __shared__ float scratch[B / 32];
    int row = blockIdx.x;
    int t = threadIdx.x;
    float local_max = -CUDART_INF_F;
    for (int j = t; j < cols; j += B) local_max = fmaxf(local_max, x[row * cols + j]);
    float maximum = block_reduce<B, true>(local_max, scratch);
    if (maximum == -CUDART_INF_F) {  // Uniform branch for an entirely masked row.
        for (int j = t; j < cols; j += B) p[row * cols + j] = 0.0f;
        return;
    }
    float local_sum = 0;
    for (int j = t; j < cols; j += B) local_sum += expf(x[row * cols + j] - maximum);
    float denominator = block_reduce<B, false>(local_sum, scratch);
    for (int j = t; j < cols; j += B) p[row * cols + j] = expf(x[row * cols + j] - maximum) / denominator;
}

__global__ void qk_scores(const float* q, const float* k, float* s, const int* limits,
                          int nq, int nk, int d) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nq * nk) return;
    int row = index / nk;
    int key = index % nk;
    float value = -CUDART_INF_F;
    if (key < limits[row]) {
        value = 0;
        for (int c = 0; c < d; ++c) value += q[row * d + c] * k[key * d + c];
        value /= sqrtf(float(d));
    }
    s[index] = value;
}

__global__ void pv_output(const float* p, const float* v, float* out, int nq, int nk, int dv) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nq * dv) return;
    int row = index / dv;
    int c = index % dv;
    float acc = 0;
    for (int j = 0; j < nk; ++j) acc += p[row * nk + j] * v[j * dv + c];
    out[index] = acc;
}

// One block owns one query; one lane/thread initially owns one key in a tile.
// Scalar FP32 math deliberately keeps MMA/layout details out of this first step.
__global__ void online_attention(const float* q, const float* k, const float* v, float* out,
                                 const int* limits, int nk, int d, int dv) {
    __shared__ float scratch[THREADS / 32];
    __shared__ float weights[THREADS];
    __shared__ float maximum, denominator, alpha;
    int row = blockIdx.x;
    int t = threadIdx.x;
    float numerator = 0;  // Thread t owns output channel t when t < dv.
    if (t == 0) {
        maximum = -CUDART_INF_F;
        denominator = 0;
    }
    __syncthreads();
    for (int base = 0; base < nk; base += THREADS) {
        int key = base + t;
        bool valid = key < nk && key < limits[row];
        float score = -CUDART_INF_F;
        if (valid) {
            score = 0;
            for (int c = 0; c < d; ++c) score += q[row * d + c] * k[key * d + c];
            score /= sqrtf(float(d));
        }
        float tile_max = block_reduce<THREADS, true>(score, scratch);
        if (tile_max == -CUDART_INF_F) continue;  // Uniform; no -inf - -inf.
        if (t == 0) {
            float new_max = fmaxf(maximum, tile_max);
            alpha = expf(maximum - new_max);
            maximum = new_max;
        }
        __syncthreads();
        float weight = valid ? expf(score - maximum) : 0.0f;
        weights[t] = weight;
        float tile_sum = block_reduce<THREADS, false>(weight, scratch);
        // block_reduce's barriers also ensure all weights are visible here.
        if (t == 0) denominator = alpha * denominator + tile_sum;
        if (t < dv) {
            numerator *= alpha;
            for (int j = 0; j < THREADS && base + j < nk; ++j) {
                numerator += weights[j] * v[(base + j) * dv + t];
            }
        }
        __syncthreads();  // Readers finish before weights/alpha can be overwritten.
    }
    if (t < dv) out[row * dv + t] = denominator > 0 ? numerator / denominator : 0.0f;
}

std::vector<double> softmax_reference(const std::vector<float>& x, int rows, int cols) {
    std::vector<double> p(x.size(), 0);
    for (int r = 0; r < rows; ++r) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < cols; ++c) maximum = std::max(maximum, double(x[r * cols + c]));
        if (!std::isfinite(maximum)) continue;
        double total = 0;
        for (int c = 0; c < cols; ++c) total += std::exp(double(x[r * cols + c]) - maximum);
        for (int c = 0; c < cols; ++c) p[r * cols + c] = std::exp(double(x[r * cols + c]) - maximum) / total;
    }
    return p;
}

void check_softmax(int cols) {
    constexpr int rows = 4;
    std::vector<float> x(rows * cols), p(x.size());
    for (int c = 0; c < cols; ++c) {
        x[c] = (c % 17 - 8) * 0.25f;
        x[cols + c] = x[c] + 1000.0f;
        x[2 * cols + c] = -CUDART_INF_F;
        x[3 * cols + c] = c < cols / 2 + 1 ? -float(c) : -CUDART_INF_F;
    }
    auto expected = softmax_reference(x, rows, cols);
    size_t bytes = x.size() * sizeof(float);
    float *dx, *dp;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dp, bytes));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
    for (int b : {32, THREADS}) {
        if (b == 32) row_softmax<32><<<rows, 32>>>(dx, dp, cols);
        else row_softmax<THREADS><<<rows, THREADS>>>(dx, dp, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(p.data(), dp, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < rows * cols; ++i) check_close(p[i], expected[i], "softmax", i);
        for (int r = 0; r < rows; ++r) {
            double sum = 0;
            for (int c = 0; c < cols; ++c) sum += p[r * cols + c];
            check_close(sum, r == 2 ? 0 : 1, "softmax row sum", r);
        }
    }
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dp));
}

void check_attention(int nq, int nk, int d, int dv, AttentionCase scenario) {
    if (dv < 1 || dv > THREADS || (scenario == AttentionCase::Causal && nq != nk)) std::exit(EXIT_FAILURE);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1, 1);
    std::vector<float> q(nq * d), k(nk * d), v(nk * dv), materialized(nq * dv), online(nq * dv);
    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng);
    for (auto& x : v) x = dist(rng);
    if (scenario == AttentionCase::Rescale) {
        // The last, one-key tile must rescale the previous 128-key state.
        if (nq != 1 || nk != 129 || d != 1 || dv != 1) std::exit(EXIT_FAILURE);
        q[0] = 1.0f;
        std::fill(k.begin(), k.end(), 0.0f);
        std::fill(v.begin(), v.end(), 1.0f);
        k[128] = std::log(128.0f);
        v[128] = 3.0f;  // Analytic output: (128*1 + 128*3) / (128+128) = 2.
    }
    std::vector<int> limits(nq, nk);
    for (int r = 0; r < nq; ++r) {
        if (scenario == AttentionCase::Causal) limits[r] = r + 1;  // Equal-length causal only.
        if (scenario == AttentionCase::Masked) limits[r] = r == 0 ? 0 : nk - r % 3;
    }
    // Independent FP64 CPU reference computes each full score row before normalization.
    std::vector<double> reference(nq * dv, 0);
    for (int r = 0; r < nq; ++r) {
        if (limits[r] == 0) continue;
        std::vector<double> scores(limits[r]);
        for (int j = 0; j < limits[r]; ++j) {
            double s = 0;
            for (int c = 0; c < d; ++c) s += double(q[r * d + c]) * k[j * d + c];
            scores[j] = s / std::sqrt(double(d));
        }
        double maximum = *std::max_element(scores.begin(), scores.end());
        double total = 0;
        for (auto& s : scores) { s = std::exp(s - maximum); total += s; }
        for (int c = 0; c < dv; ++c) {
            for (int j = 0; j < limits[r]; ++j) reference[r * dv + c] += scores[j] / total * v[j * dv + c];
        }
    }
    float *dq, *dk, *dv_gpu, *ds, *dp, *do_materialized, *do_online;
    int* dl;
    size_t qbytes = q.size() * sizeof(float), kbytes = k.size() * sizeof(float);
    size_t vbytes = v.size() * sizeof(float), obytes = materialized.size() * sizeof(float);
    size_t sbytes = size_t(nq) * nk * sizeof(float), lbytes = limits.size() * sizeof(int);
    CUDA_CHECK(cudaMalloc(&dq, qbytes));
    CUDA_CHECK(cudaMalloc(&dk, kbytes));
    CUDA_CHECK(cudaMalloc(&dv_gpu, vbytes));
    CUDA_CHECK(cudaMalloc(&ds, sbytes));
    CUDA_CHECK(cudaMalloc(&dp, sbytes));
    CUDA_CHECK(cudaMalloc(&do_materialized, obytes));
    CUDA_CHECK(cudaMalloc(&do_online, obytes));
    CUDA_CHECK(cudaMalloc(&dl, lbytes));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), qbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), kbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv_gpu, v.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dl, limits.data(), lbytes, cudaMemcpyHostToDevice));
    qk_scores<<<(nq * nk + THREADS - 1) / THREADS, THREADS>>>(dq, dk, ds, dl, nq, nk, d);
    CUDA_CHECK(cudaGetLastError());
    row_softmax<THREADS><<<nq, THREADS>>>(ds, dp, nk);
    CUDA_CHECK(cudaGetLastError());
    pv_output<<<(nq * dv + THREADS - 1) / THREADS, THREADS>>>(dp, dv_gpu, do_materialized, nq, nk, dv);
    CUDA_CHECK(cudaGetLastError());
    online_attention<<<nq, THREADS>>>(dq, dk, dv_gpu, do_online, dl, nk, d, dv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(materialized.data(), do_materialized, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(online.data(), do_online, obytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < nq * dv; ++i) {
        check_close(materialized[i], reference[i], "materialized attention", i);
        check_close(online[i], reference[i], "online attention", i);
    }
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dv_gpu));
    CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(dp)); CUDA_CHECK(cudaFree(do_materialized));
    CUDA_CHECK(cudaFree(do_online)); CUDA_CHECK(cudaFree(dl));
}

int main() {
    for (int n : {1, 7, 31, 32, 33, 127, 128, 129, 513}) check_softmax(n);
    std::puts("PASS warp/block softmax: ragged rows, large shift, partial/fully masked rows");
    check_attention(1, 1, 1, 1, AttentionCase::Dense);
    check_attention(3, 7, 5, 3, AttentionCase::Dense);
    check_attention(5, 129, 7, 17, AttentionCase::Dense);
    check_attention(4, 257, 8, 128, AttentionCase::Masked);
    check_attention(33, 33, 16, 9, AttentionCase::Causal);
    check_attention(1, 129, 1, 1, AttentionCase::Rescale);
    std::puts("PASS materialized and online attention against FP64 CPU reference");
    return EXIT_SUCCESS;
}
