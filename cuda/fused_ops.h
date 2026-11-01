// 融合算子的对外接口（CUDA 侧）
//
// 只暴露 launcher，kernel 本身留在 .cu 里——调用方不需要知道 block/grid 怎么配。
// 每个算子都提供"融合版"和"不融合的对照版"两套 launcher：
// 对照版存在的唯一目的是让 bench 能证明收益来自融合，而不是来自别处。

#ifndef KV_FUSED_OPS_H
#define KV_FUSED_OPS_H

#include <cuda_runtime.h>

namespace kv {

// softmax 的 block 大小：列数少时没必要开满 256 线程，避免一半线程空转。
inline int softmax_block_size(int cols) {
    if (cols <= 64) return 64;
    if (cols <= 128) return 128;
    return 256;
}

// out[r, c] = exp(in[r, c] - max_c) / sum_c(exp(in[r, c] - max_c))
// 融合版：单个 kernel、无中间张量。
void launch_softmax_fused(const float* in, float* out, int rows, int cols,
                          cudaStream_t stream);

// 三次独立扫行的对照版（等价于不融合时的行为）。
void launch_softmax_three_pass(const float* in, float* out, int rows, int cols,
                               cudaStream_t stream);

// out[i] = silu(gate[i]) * up[i]，n 为元素个数（vec4 版要求 n % 4 == 0）。
void launch_silu_and_mul_vec4(const float* gate, const float* up, float* out,
                              long long n, cudaStream_t stream);
void launch_silu_and_mul_scalar(const float* gate, const float* up, float* out,
                                long long n, cudaStream_t stream);

}  // namespace kv

#endif  // KV_FUSED_OPS_H
