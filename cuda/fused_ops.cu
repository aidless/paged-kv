// 融合算子：softmax 与 silu-and-mul
//
// 这两个是推理引擎里最热的 elementwise 算子：
//   - attention 每个 head 做完 QK^T 就要 softmax（decode 阶段每 step 都跑）
//   - MLP 的 SwiGLU 门控是 silu(gate) * up（每层每个 token 都跑）
//
// 它们的共同特征是**带宽受限**（bandwidth-bound），不是算力受限：
// 每读 4 字节只做几次浮点运算，所以性能上界由显存带宽决定，不由 FLOPS 决定。
// 因此这里的设计目标不是"少做运算"，而是"少读写显存"：
//   - softmax 融合成一个 kernel（不产生中间张量，省掉中间结果的两轮全量往返）
//   - silu_and_mul 用 float4 向量化，让每条指令搬 16 字节
//
// 正确性由 CPU 参考实现对拍（见 bench.cu），性能由有效带宽衡量（见 bench.cu）。

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cstdlib>

#include "fused_ops.h"

// 与 bench.cu 一致的错误宏（launch 里也做错误检查，不静默吞掉）。
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t err_ = (expr);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                         cudaGetErrorString(err_), __FILE__, __LINE__);           \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)
#endif

namespace kv {

// ---------------------------------------------------------------------------
// 块内归约（softmax 要用两次：一次求最大值、一次求和）
//
// 结构：warp 内用 shuffle 归约 -> 每个 warp 的 lane 0 写 shared ->
// 第一个 warp 再归约这些部分和 -> 结果留在 shared[0]。
// 这是 CUDA 上最省同步次数的写法：只需两次 __syncthreads()，
// 而不是每轮都用 shared 做一次全块归约。
// ---------------------------------------------------------------------------

__device__ __forceinline__ float block_reduce_max(float v, float* shared) {
    // warp 内归约
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    const int nwarps = (blockDim.x + 31) >> 5;
    // 第一个 warp 负责收尾；其余 warp 读到的是无意义值但不会用
    v = (threadIdx.x < nwarps) ? shared[threadIdx.x] : -CUDART_INF_F;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        }
        if (threadIdx.x == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    const int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (threadIdx.x == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

// 基线（真正不融合的路径）：三次独立 kernel，中间结果落显存。
//   1) 求每行 max        -> row_max[row]
//   2) 读 row + row_max 求 sum -> row_sum[row]
//   3) 读 row + row_max + row_sum 归一化 -> out
// 这是"框架里 softmax 如果拆成三次算子调用"实际会发生的显存流量：
// 每行输入被读 3 次，外加 row_max/row_sum 的写读。对比 fused 的单 kernel
// （输入基本留在 L2，DRAM 流量约 1 读 + 1 写），能直接看出融合省了多少显存。
__global__ void softmax_max_kernel(const float* __restrict__ in,
                                   float* __restrict__ row_max,
                                   int rows, int cols) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const float* row_in = in + static_cast<long long>(row) * cols;

    float m = -CUDART_INF_F;
    for (int c = tid; c < cols; c += bdim) m = fmaxf(m, row_in[c]);
    m = block_reduce_max(m, shared);
    if (tid == 0) row_max[row] = m;
}

__global__ void softmax_sum_kernel(const float* __restrict__ in,
                                   const float* __restrict__ row_max,
                                   float* __restrict__ row_sum,
                                   int rows, int cols) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const float* row_in = in + static_cast<long long>(row) * cols;

    float s = 0.0f;
    for (int c = tid; c < cols; c += bdim) s += __expf(row_in[c] - row_max[row]);
    s = block_reduce_sum(s, shared);
    if (tid == 0) row_sum[row] = s;
}

__global__ void softmax_norm_kernel(const float* __restrict__ in,
                                    const float* __restrict__ row_max,
                                    const float* __restrict__ row_sum,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const float* row_in = in + static_cast<long long>(row) * cols;
    float* row_out = out + static_cast<long long>(row) * cols;

    const float inv = 1.0f / row_sum[row];
    for (int c = tid; c < cols; c += bdim) {
        row_out[c] = __expf(row_in[c] - row_max[row]) * inv;
    }
}

// 融合版：单 kernel，无中间张量。
//
// 为什么不把整行塞进寄存器：cols 运行期才知道，寄存器数量编译期固定。
// 所以走"每线程按 stride 负责若干列 + 两次块内归约"的结构——省掉的不是块内
// 重复计算（那本来就是并行的），而是中间结果的全量显存往返，这才是带宽瓶颈。
//
// 数值稳定性：先减行最大值再取 exp，与 PyTorch softmax 口径一致，
// 避免 expf 在输入较大时溢出成 inf。
__global__ void softmax_fused_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int rows, int cols) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const float* row_in = in + static_cast<long long>(row) * cols;
    float* row_out = out + static_cast<long long>(row) * cols;

    float m = -CUDART_INF_F;
    for (int c = tid; c < cols; c += bdim) m = fmaxf(m, row_in[c]);
    m = block_reduce_max(m, shared);

    float s = 0.0f;
    for (int c = tid; c < cols; c += bdim) s += __expf(row_in[c] - m);
    s = block_reduce_sum(s, shared);

    const float inv = 1.0f / s;
    for (int c = tid; c < cols; c += bdim) row_out[c] = __expf(row_in[c] - m) * inv;
}

void launch_softmax_fused(const float* in, float* out, int rows, int cols,
                          cudaStream_t stream) {
    const int threads = softmax_block_size(cols);
    softmax_fused_kernel<<<rows, threads, threads * sizeof(float), stream>>>(in, out, rows, cols);
}

void launch_softmax_three_pass(const float* in, float* out, int rows, int cols,
                               cudaStream_t stream) {
    const int threads = softmax_block_size(cols);
    float *row_max = nullptr, *row_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&row_max, static_cast<size_t>(rows) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&row_sum, static_cast<size_t>(rows) * sizeof(float)));
    softmax_max_kernel<<<rows, threads, threads * sizeof(float), stream>>>(in, row_max, rows,
                                                                           cols);
    softmax_sum_kernel<<<rows, threads, threads * sizeof(float), stream>>>(in, row_max, row_sum,
                                                                           rows, cols);
    softmax_norm_kernel<<<rows, threads, threads * sizeof(float), stream>>>(in, row_max, row_sum,
                                                                            out, rows, cols);
    CUDA_CHECK(cudaFree(row_max));
    CUDA_CHECK(cudaFree(row_sum));
}

// ---------------------------------------------------------------------------
// silu_and_mul（SwiGLU 门控）
// ---------------------------------------------------------------------------

// out[i] = silu(gate[i]) * up[i]，silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// 用 float4 向量化：一条 load/store 指令搬 16 字节、四个分量一起算。
// 对带宽受限算子，这个改动的收益比调 block 大小更直接——
// 标量版每次只搬 4 字节，访存事务利用率低。
__global__ void silu_and_mul_vec4_kernel(const float4* __restrict__ gate,
                                         const float4* __restrict__ up,
                                         float4* __restrict__ out,
                                         long long n4) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (; i < n4; i += stride) {
        const float4 g = gate[i];
        const float4 u = up[i];
        float4 r;
        r.x = __fdividef(g.x, 1.0f + __expf(-g.x)) * u.x;
        r.y = __fdividef(g.y, 1.0f + __expf(-g.y)) * u.y;
        r.z = __fdividef(g.z, 1.0f + __expf(-g.z)) * u.z;
        r.w = __fdividef(g.w, 1.0f + __expf(-g.w)) * u.w;
        out[i] = r;
    }
}

// 标量对照版：给 bench 一个"不向量化会差多少"的读数。
__global__ void silu_and_mul_scalar_kernel(const float* __restrict__ gate,
                                           const float* __restrict__ up,
                                           float* __restrict__ out,
                                           long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        const float g = gate[i];
        out[i] = __fdividef(g, 1.0f + __expf(-g)) * up[i];
    }
}

void launch_silu_and_mul_vec4(const float* gate, const float* up, float* out,
                              long long n, cudaStream_t stream) {
    // 调用方保证 n 能被 4 整除（bench 里显式校验）；否则应改走标量路径。
    const long long n4 = n / 4;
    const int threads = 256;
    int blocks = static_cast<int>((n4 + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    silu_and_mul_vec4_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float4*>(gate), reinterpret_cast<const float4*>(up),
        reinterpret_cast<float4*>(out), n4);
}

void launch_silu_and_mul_scalar(const float* gate, const float* up, float* out,
                                long long n, cudaStream_t stream) {
    const int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    silu_and_mul_scalar_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

}  // namespace kv
