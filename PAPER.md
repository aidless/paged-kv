# paged-kv：PagedAttention 块管理层的边界验证与异构算子移植

> 本文档是 paged-kv 的**论文摘要草稿**，内容全部来自仓库真实实现与实测结果，
> 无虚构数字。

## 摘要（Abstract）

PagedAttention 是 vLLM 的核心贡献：把 KV cache 像操作系统虚拟内存一样**分页管理**。
本文从零实现其中的**块管理层**——不含 KV 数据本身，专注生命周期与引用计数的簿记，
对应 vLLM `vllm/v1/core/block_pool.py` + `block_manager` 那一层。

**验证（可复算）**：三个测试套件全绿——block_manager / kv_cache（74 checks）/
prefix_cache（133 checks）。测试做过变异验证（7+ 个变异全部变红），确认断言不是
装饰：去掉槽位 `+ offset` → 8 checks 红；去掉越界预校验 → `double free or
corruption` 崩溃；去掉哈希碰撞内容复验 → 3 checks 红（靠注入必然碰撞的哈希才
测得到）；`cache_block` 不校验块号上界 → 段错误（越界写穿块状态数组）。

**关键语义边界**：
- **失败必须无副作用**：`append_token` 内存不足返回 `false`，序列长度、block table、
  空闲链全部保持原状（alloc-then-commit 经典模式）。
- **双释放安全**：`free_sequence` 对不存在的序列是 no-op，不会把块放进空闲链两次。
- **COW 触发条件**：写入「未满且被共享（`ref_count > 1`）」的块时，新块接管 table
  位置、旧块引用 -1。
- **prefix cache 与 COW 不打架**：只缓存满块（内容由 token 序列唯一确定，之后
  无人再写，共享块天然只读）；未满块共享走 `fork_sequence` + COW，两者互补。

**性能（HIP / gfx1100 实测）**：softmax（4096×1024）融合单 kernel 0.0415 ms /
817 GB/s（峰值 114%），相对 three-pass 0.0869 ms / 386 GB/s 快 **2.09×**；silu-and-mul
（n≈45.1M）float4 向量化 0.755 ms / 716 GB/s，相对 scalar 0.780 ms 快 **1.03×**。
对拍（CPU 参考）全部 PASS：softmax `max|diff|` 1.86e-09、silu 1.91e-06（容差 1e-4）。

**方法教训（写进 notes/）**：两个变体可能是同一个 kernel——必须先 diff 验证变体
真的不同再跑对照；实测 D2D 带宽随拷贝尺寸变化（L2 内 6MB 虚高到 800–1600 GB/s），
分母必须用 **1 GiB 稳态值**（`peak_bw_method=measured_d2d_1gib`），因为 HIP 的
`memoryClockRate` 语义与 CUDA 不一致（gfx1100 报 1124 MHz 但有效数据率是 4 倍，
naive 公式低估 ~8 倍）。

**明确边界**：本仓**不证明** CUDA 侧已在 NVIDIA 硬件验证（本机无 nvcc，CUDA 版
同构移植但只在 HIP/gfx1100 实测）；**不证明**是 vLLM 的完整替代（不含 KV 数据、
不含 continuous batching 调度器）；softmax >100% 峰值占比是 L2 内输入所致，**不能**
外推为任意规模都超峰值。

## 关键词

PagedAttention；KV cache；块管理器；写时复制；prefix caching；vLLM；
异构算子移植；HIP；CUDA；推理引擎

## 与真实代码的对应

| 摘要声明 | 对应真实实现 |
|---|---|
| 块管理层 | `include/kv/block_manager.h` + `src/block_manager.cpp` |
| 槽位映射 + KV 存储 | `include/kv/kv_cache.h` + `src/kv_cache.cpp` |
| 链式哈希 + FIFO 驱逐 | `src/block_manager.cpp`（FNV-1a + evict_one_cached） |
| 74 / 133 checks | `tests/test_kv_cache.cpp` / `tests/test_prefix_cache.cpp` |
| CUDA 融合算子 | `cuda/`（fused softmax / silu-and-mul） |
| HIP 同构移植 | `hip/`（hipcc 编译，gfx1100 实测） |
| 源码对照笔记 | `notes/`（micro-vLLM + 实测方法论） |
