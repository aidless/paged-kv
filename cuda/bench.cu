// 融合算子的基准驱动：正确性对拍 + 有效带宽测量
//
// 为什么不用 ncu：Kaggle 的 GPU 容器不给 perf counter 权限，ncu 起不来。
// 所以这里换成 roofline 口径——把"读了多少字节、写了多少字节"算清楚，
// 除以实测耗时得到有效带宽，再和设备的理论峰值带宽比。
// 这个口径能回答性能问题里最关键的那句：**这个 kernel 离硬件上限还有多远**，
// 而不是只报一个孤立的毫秒数。
//
// 输出格式：既有给人看的分节表格，也有给脚本解析的 `key=value` 行（末尾）。
// 所有数字都来自本次运行，不存在预先写死的期望值。

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fused_ops.h"

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t err_ = (expr);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                         __FILE__, __LINE__);                                     \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace {

// 确定性伪随机：同一 seed 在任何机器上生成同一组数据，
// 这样"对拍通过"这件事才是可复现的，而不是碰巧这次数据温和。
struct Lcg {
    unsigned long long s;
    explicit Lcg(unsigned long long seed) : s(seed) {}
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        // 取高位映射到 [-4, 4)，覆盖 softmax 里 exp 的典型动态范围
        const unsigned int hi = static_cast<unsigned int>(s >> 33);
        return (static_cast<float>(hi) / static_cast<float>(0x7FFFFFFF)) * 8.0f - 4.0f;
    }
};

// ---- CPU 参考实现（按 PyTorch 口径写，作为对拍的唯一真相源）----

void cpu_softmax(const std::vector<float>& in, std::vector<float>& out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = in.data() + static_cast<long long>(r) * cols;
        float m = row[0];
        for (int c = 1; c < cols; ++c) m = std::fmax(m, row[c]);
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += std::exp(static_cast<double>(row[c] - m));
        for (int c = 0; c < cols; ++c) {
            out[static_cast<long long>(r) * cols + c] =
                static_cast<float>(std::exp(static_cast<double>(row[c] - m)) / s);
        }
    }
}

void cpu_silu_and_mul(const std::vector<float>& gate, const std::vector<float>& up,
                      std::vector<float>& out, long long n) {
    for (long long i = 0; i < n; ++i) {
        const double g = gate[i];
        const double sig = 1.0 / (1.0 + std::exp(-g));
        out[i] = static_cast<float>(g * sig * up[i]);
    }
}

// ---- 计时 ----

// 跑 warmup 次预热（让 GPU 时钟爬升到稳态），再跑 iters 次取平均。
// 不做 warmup 会把频率爬升阶段算进去，读数偏低。
template <typename Fn>
float time_kernel_ms(Fn&& fn, int warmup, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float e = std::fabs(a[i] - b[i]);
        if (e > d) d = e;
    }
    return d;
}

// 理论峰值带宽（GB/s）：DDR 双沿，故乘 2。
// 注意：HIP 下 memoryClockRate 的语义与 CUDA 不一致（gfx1100 报 1124000 kHz，
// 但真实 GDDR6 有效数据率是时钟的 4 倍），这个公式会**低估**真实带宽约 8 倍，
// 所以它不能当"离硬件上限多远"的分母。分母改用下面的实测上限。
// 在 CUDA 上该公式对多数消费/数据中心卡也是近似，实测上限更稳。
double naive_theoretical_gbs(const cudaDeviceProp& p) {
    return 2.0 * static_cast<double>(p.memoryClockRate) * 1e3 *
           (static_cast<double>(p.memoryBusWidth) / 8.0) / 1e9;
}

// 实测可达上限：1 GiB device-to-device 拷贝，读+写字节数 / 耗时。
// 这是"这台上限到底在哪"的最直接证据——不依赖任何规格常数，
// 不依赖对 memoryClockRate 的语义假设（那条路在 HIP 栈上错了 8 倍）。
double measure_peak_bw_gbs() {
    const size_t nbytes = 1ULL << 30;  // 1 GiB
    float *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, nbytes));
    CUDA_CHECK(cudaMalloc(&b, nbytes));
    CUDA_CHECK(cudaMemset(a, 0x3f, nbytes));

    const int warmup = 10, iters = 50;
    for (int i = 0; i < warmup; ++i)
        CUDA_CHECK(cudaMemcpy(b, a, nbytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i)
        CUDA_CHECK(cudaMemcpy(b, a, nbytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    ms /= static_cast<float>(iters);

    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));

    // 拷贝要读一份写一份，所以流量是 2 * nbytes
    return (2.0 * static_cast<double>(nbytes)) / (ms * 1e6);
}

}  // namespace

int main() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const double naive_peak = naive_theoretical_gbs(prop);
    const double peak_bw = measure_peak_bw_gbs();
    std::printf("=== 设备 ===\n");
    std::printf("name                     : %s\n", prop.name);
    std::printf("compute capability       : %d.%d\n", prop.major, prop.minor);
    std::printf("SMs                      : %d\n", prop.multiProcessorCount);
    std::printf("global mem               : %.1f GB\n",
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    std::printf("memory clock / bus width  : %d MHz / %d bit\n", prop.memoryClockRate / 1000,
                prop.memoryBusWidth);
    std::printf("naive spec bandwidth     : %.1f GB/s (公式低估，仅供参考)\n", naive_peak);
    std::printf("measured D2D peak        : %.1f GB/s (1 GiB 拷贝，分母)\n\n", peak_bw);

    const int rows = 4096;   // 典型 batch*seq 规模的一行一个 softmax
    const int cols = 1024;   // 典型 head_dim 或 vocab 分块
    const long long n_softmax = static_cast<long long>(rows) * cols;
    const size_t bytes_softmax = static_cast<size_t>(n_softmax) * sizeof(float);

    std::vector<float> h_in(static_cast<size_t>(n_softmax));
    Lcg rng(12345);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = rng.next();

    std::vector<float> h_ref(static_cast<size_t>(n_softmax));
    cpu_softmax(h_in, h_ref, rows, cols);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_softmax));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_softmax));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes_softmax, cudaMemcpyHostToDevice));

    std::vector<float> h_out(static_cast<size_t>(n_softmax));

    // ---------------- softmax ----------------
    std::printf("=== softmax (%d x %d) ===\n", rows, cols);

    cudaStream_t stream = nullptr;
    const int warmup = 10;
    const int iters = 100;

    CUDA_CHECK(cudaMemset(d_out, 0, bytes_softmax));
    const float ms_fused = time_kernel_ms(
        [&] { kv::launch_softmax_fused(d_in, d_out, rows, cols, stream); }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes_softmax, cudaMemcpyDeviceToHost));
    const float diff_fused = max_abs_diff(h_out, h_ref);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes_softmax));
    const float ms_three = time_kernel_ms(
        [&] { kv::launch_softmax_three_pass(d_in, d_out, rows, cols, stream); }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes_softmax, cudaMemcpyDeviceToHost));
    const float diff_three = max_abs_diff(h_out, h_ref);

    // 带宽口径：两个实现都必须"读一遍输入、写一遍输出"，
    // 各自的差异在于中间结果有没有多跑显存——这体现在耗时上，不体现在字节数上。
    const double bw_fused = (2.0 * bytes_softmax) / (ms_fused * 1e6);
    const double bw_three = (2.0 * bytes_softmax) / (ms_three * 1e6);

    std::printf("fused      : %.3f ms   %.1f GB/s   max|diff|=%.3e   (%.1f%% of peak)\n",
                ms_fused, bw_fused, diff_fused, 100.0 * bw_fused / peak_bw);
    std::printf("three-pass : %.3f ms   %.1f GB/s   max|diff|=%.3e   (%.1f%% of peak)\n",
                ms_three, bw_three, diff_three, 100.0 * bw_three / peak_bw);
    std::printf("speedup    : %.2fx\n\n", ms_three / ms_fused);

    // ---------------- silu_and_mul ----------------
    // 取 SwiGLU 的典型规模：一个 token 的中间维度 11008（Llama-2 7B 的 ff_dim）
    const long long n = static_cast<long long>(rows) * 11008LL;
    const size_t bytes_gate = static_cast<size_t>(n) * sizeof(float);

    std::printf("=== silu_and_mul (n = %lld = %d rows x 11008) ===\n", n, rows);

    std::vector<float> h_gate(static_cast<size_t>(n)), h_up(static_cast<size_t>(n));
    h_gate.resize(static_cast<size_t>(n));
    h_up.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        h_gate[i] = rng.next();
        h_up[i] = rng.next();
    }

    std::vector<float> h_gout(static_cast<size_t>(n));
    cpu_silu_and_mul(h_gate, h_up, h_gout, n);

    float *d_gate = nullptr, *d_up = nullptr, *d_gout = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gate, bytes_gate));
    CUDA_CHECK(cudaMalloc(&d_up, bytes_gate));
    CUDA_CHECK(cudaMalloc(&d_gout, bytes_gate));
    CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), bytes_gate, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), bytes_gate, cudaMemcpyHostToDevice));

    std::vector<float> h_gcheck(static_cast<size_t>(n));

    // 每个元素读 2 个 float、写 1 个 float
    const double bytes_silu = 3.0 * static_cast<double>(bytes_gate);

    CUDA_CHECK(cudaMemset(d_gout, 0, bytes_gate));
    const float ms_vec = time_kernel_ms(
        [&] { kv::launch_silu_and_mul_vec4(d_gate, d_up, d_gout, n, stream); }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(h_gcheck.data(), d_gout, bytes_gate, cudaMemcpyDeviceToHost));
    const float diff_vec = max_abs_diff(h_gcheck, h_gout);

    CUDA_CHECK(cudaMemset(d_gout, 0, bytes_gate));
    const float ms_scalar = time_kernel_ms(
        [&] { kv::launch_silu_and_mul_scalar(d_gate, d_up, d_gout, n, stream); }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(h_gcheck.data(), d_gout, bytes_gate, cudaMemcpyDeviceToHost));
    const float diff_scalar = max_abs_diff(h_gcheck, h_gout);

    const double bw_vec = bytes_silu / (ms_vec * 1e6);
    const double bw_scalar = bytes_silu / (ms_scalar * 1e6);

    std::printf("vec4       : %.3f ms   %.1f GB/s   max|diff|=%.3e   (%.1f%% of peak)\n", ms_vec,
                bw_vec, diff_vec, 100.0 * bw_vec / peak_bw);
    std::printf("scalar     : %.3f ms   %.1f GB/s   max|diff|=%.3e   (%.1f%% of peak)\n",
                ms_scalar, bw_scalar, diff_scalar, 100.0 * bw_scalar / peak_bw);
    std::printf("speedup    : %.2fx\n\n", ms_scalar / ms_vec);

    // ---------------- 机器可解析的汇总 ----------------
    // 这些行会被脚本读回去写进文档，所以键名要稳定、单位要写在名字里。
    std::printf("### SUMMARY ###\n");
    std::printf("gpu_name=%s\n", prop.name);
    std::printf("gpu_cc=%d.%d\n", prop.major, prop.minor);
    std::printf("gpu_sms=%d\n", prop.multiProcessorCount);
    std::printf("peak_bw_gbs=%.1f\n", peak_bw);
    std::printf("peak_bw_method=measured_d2d_1gib\n");
    std::printf("softmax_rows=%d\n", rows);
    std::printf("softmax_cols=%d\n", cols);
    std::printf("softmax_fused_ms=%.4f\n", ms_fused);
    std::printf("softmax_fused_gbs=%.1f\n", bw_fused);
    std::printf("softmax_fused_pct_peak=%.1f\n", 100.0 * bw_fused / peak_bw);
    std::printf("softmax_fused_maxdiff=%.3e\n", diff_fused);
    std::printf("softmax_threepass_ms=%.4f\n", ms_three);
    std::printf("softmax_threepass_gbs=%.1f\n", bw_three);
    std::printf("softmax_threepass_maxdiff=%.3e\n", diff_three);
    std::printf("softmax_speedup=%.3f\n", ms_three / ms_fused);
    std::printf("silu_n=%lld\n", n);
    std::printf("silu_vec4_ms=%.4f\n", ms_vec);
    std::printf("silu_vec4_gbs=%.1f\n", bw_vec);
    std::printf("silu_vec4_pct_peak=%.1f\n", 100.0 * bw_vec / peak_bw);
    std::printf("silu_vec4_maxdiff=%.3e\n", diff_vec);
    std::printf("silu_scalar_ms=%.4f\n", ms_scalar);
    std::printf("silu_scalar_gbs=%.1f\n", bw_scalar);
    std::printf("silu_scalar_maxdiff=%.3e\n", diff_scalar);
    std::printf("silu_speedup=%.3f\n", ms_scalar / ms_vec);

    // 对拍是门：对不上就是失败，不能用"看起来差不多"放行
    const float tol = 1e-4f;  // __expf 的精度上限约 2 ulp，1e-4 足够宽又足够严
    const bool ok = diff_fused < tol && diff_three < tol && diff_vec < tol &&
                    diff_scalar < tol;
    std::printf("correctness=%s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_gout));
    return ok ? 0 : 1;
}
