// 融合算子的对外接口（HIP/ROCm 侧）
//
// 这是 CUDA 版 fused_ops.h 的 HIP 移植。接口完全一致（名字、参数、语义），
// 让 bench 只需要换掉编译工具链（hipcc 代替 nvcc）就能在同一套代码上跑——
// 这正是"异构算子可移植"的最小证明：同一份算法描述，CUDA 与 HIP 各编译一份。

#ifndef KV_FUSED_OPS_H
#define KV_FUSED_OPS_H

#include <hip/hip_runtime.h>

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
                          hipStream_t stream);

// 三次独立扫行的对照版（等价于不融合时的行为）。
void launch_softmax_three_pass(const float* in, float* out, int rows, int cols,
                               hipStream_t stream);

// out[i] = silu(gate[i]) * up[i]，n 为元素个数（vec4 版要求 n % 4 == 0）。
void launch_silu_and_mul_vec4(const float* gate, const float* up, float* out,
                              long long n, hipStream_t stream);
void launch_silu_and_mul_scalar(const float* gate, const float* up, float* out,
                                long long n, hipStream_t stream);

}  // namespace kv

#endif  // KV_FUSED_OPS_H
