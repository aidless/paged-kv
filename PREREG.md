# PREREG — paged-kv 预注册规格

> 本文件是 paged-kv（C++ PagedAttention 风格 KV cache 块管理器）的**预注册规格**：
> 把研究假设、评测方案、成功标准与明确边界写在分析之前。所有数字可从仓库内
> 确定性代码/实测结果复算。

## 1. 研究问题

大模型推理时，vLLM 的核心贡献是把 KV cache 像操作系统虚拟内存一样**分页管理**：
物理块按需分配、多序列共享、写时复制。面试推理引擎岗时，「PagedAttention 是什么」
人人能答；「COW 触发时新旧块的引用计数怎么变、内存不足时失败必须无副作用」只有
写过才答得出。

**核心假设（H1）**：PagedAttention 的**块管理层**（不含 KV 数据本身，专注生命周期
与引用计数的簿记）可从零实现并**逐边界验证**——分配/回收/COW/引用计数/前缀缓存，
对应 vLLM `vllm/v1/core/block_pool.py` + `block_manager` 那一层。

**辅助假设（H2，性能）**：同一份算子算法两份编译（CUDA 版 + HIP 同构移植）是
「异构算子可移植」的最小证明；CUDA 侧未在 NVIDIA 硬件验证（本机无 nvcc）。

**辅助假设（H3，方法）**：实测 D2D 带宽随拷贝尺寸变化（L2 内 6MB 会虚高到
800–1600 GB/s），分母必须用 **1 GiB 稳态值**（`peak_bw_method=measured_d2d_1gib`），
否则 naive 公式低估 ~8 倍（gfx1100 报 1124 MHz 但有效数据率是 4 倍）。

## 2. 评测方案（可复算）

### 2.1 单元测试（三套件）

```bash
make test    # block_manager / kv_cache / prefix_cache 三套件
```

实测：block_manager ALL PASS；kv_cache ALL PASS (74 checks)；
prefix_cache ALL PASS (133 checks)。

### 2.2 变异验证（断言不是装饰）

测试本身做过变异验证（7+ 个变异，全部变红）：

| 变异 | 结果 |
|---|---|
| 槽位计算去掉 `+ offset` | 8 checks 红 |
| 批量写去掉越界预校验 | `double free or corruption (out)` 崩溃 |
| 去掉哈希碰撞的内容复验 | 3 checks 红（靠注入必然碰撞的哈希才测得到） |
| `dec_ref` 不保留缓存块 | 6 checks 红 |
| 命中不回填引用计数 | 4 checks 红 |
| `cache_block` 不校验满块 | 5 checks 红 |
| 碰撞时覆盖旧映射 | 2 checks 红 |
| 驱逐不检查引用中 | 3 checks 红 |
| `cache_block` 不校验块号上界 | 段错误（越界写穿块状态数组） |

### 2.3 算子融合基准（HIP / gfx1100 实测）

对拍（CPU 参考）全部 PASS，`max|diff|`：softmax 1.86e-09、silu 1.91e-06（容差 1e-4）。

| 算子 | 变体 | 耗时 | 有效带宽 | 峰值占比 |
|------|------|------|----------|----------|
| softmax (4096×1024) | fused 单 kernel | 0.0415 ms | 817 GB/s | 114% |
| | three-pass（中间量落显存） | 0.0869 ms | 386 GB/s | 54% |
| silu-and-mul (n≈45.1M) | float4 向量化 | 0.755 ms | 716 GB/s | 100% |
| | scalar | 0.780 ms | 690 GB/s | 96% |

**成功标准**：
- `make test` 三套件全绿（ALL PASS，含 74 + 133 checks）；
- softmax 融合 **2.09×**、silu 向量化 **1.03×**（相对各自对照变体）；
- 基准分母用 1 GiB D2D 实测上限，不用规格常数。

## 3. 明确边界（不证明什么）

1. **不证明 CUDA 侧已在 NVIDIA 硬件验证**——本机无 nvcc，CUDA 版同构移植但只在
   HIP/gfx1100 上实测；这是「异构算子可移植」的最小证明，非双平台性能对照。
2. **不证明是 vLLM 的完整替代**——本仓专注块管理层簿记，不含 KV 数据本身、
   不含 continuous batching 调度器（Roadmap 未完成项）。
3. **softmax >100% 峰值占比**：输入基本留在 L2（>100% 正常），不是测量错误，
   但**不能**外推为任意规模都超峰值。
4. **silu 向量化 1.03× 是小提升**——标量版也已在带宽上限附近，这是**带宽受限
   算子**的特征，不是向量化无效。
5. **prefix cache 只缓存满块**：`cache_block` 拒绝不足 `block_size` 的 token；
   未满块的共享走 `fork_sequence` + COW，两者互补不重叠。

## 4. 成功标准（判据）

本预注册声明**成功**当且仅当：
1. `make test` 三套件全绿（block_manager / kv_cache 74 checks / prefix_cache 133）；
2. 变异注入全部变红（断言非装饰）；
3. 基准数字可在 README「算子融合基准」节对上（HIP/gfx1100 实测）；
4. 每一条"不证明"边界都在 README 或本文件有对应记录（不静默）。
