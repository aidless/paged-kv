# 源码对照笔记

本目录存放精读参考项目时的笔记，每篇对照本仓库实现，标注 `文件:行号`。

## 精读顺序

1. ✅ **mukhal/micro-vllm** —— 已精读，见 [micro-vllm.md](micro-vllm.md)
2. **psmarter/mini-infer** —— paged KV cache + continuous batching + TP，主线教材
3. **HJCheng0602/nanoPD** —— 从零 PD 分离引擎，直接对应推理岗 JD
4. **vLLM v1** —— `vllm/v1/core/block_pool.py`，工业标准实现，做对照
5. **shawnshekari/strixite** —— C++/HIP，AMD GPU 路线（持有 AMD 云额度，可实跑复现）

## 笔记格式

每篇一个文件，开头三行回答：

- 它解决了什么问题？
- 它的数据结构选型（和本仓库/vLLM 差在哪）？
- 我能带走的一个技巧是什么？

## 参考项目索引（2026-10-31 检索）

| 项目 | 价值 |
|---|---|
| psmarter/mini-infer (237★) | 从零：paged KV cache、continuous batching、prefix caching、TP |
| HJCheng0602/nanoPD (163★) | 从零 PD 分离推理引擎 |
| naklecha/simple-llm (492★) | ~1300 行极简引擎 |
| shawnshekari/strixite (91★) | C++/HIP，AMD Strix Halo 手写引擎 |
| mukhal/micro-vllm | 单文件 paged attention |

## 实现期语义笔记（2026-10-31，调试实测）

- **COW 触发边界**：COW 只在写入「未满且被共享（ref>1）」的块时触发。块已满时 append 走正常扩块路径，与共享无关。
- **COW 次数上限 = 共享者数 − 1**：三方共享一个块，前两个写入者各自 COW 搬走后，最后一个持有者写入时 ref 已降为 1，直接原地写，不再 COW。推论：块 0 的 ref 归零回收只发生在「所有共享者都搬走或释放」时，最后一个持有者天然免费接管。
- **调试陷阱**：`printf("a=%d b=%d", f(), g())` 的实参求值顺序不确定（GCC 从右到左），带副作用的函数（如 append_token）与读取函数（cow_count）混在同一 printf 里，打印的是调用前状态。探针代码必须一行一个副作用。

## 前缀缓存实现笔记（2026-11-01）

- **只缓存满块是硬约束，不是优化**：满块内容由 token 序列唯一确定、之后无人再写，共享块天然只读，所以 prefix cache 与 COW 互不干扰。若允许缓存未满块，一次 COW 就让该块前缀失效。
- **链式哈希必须挂父哈希**：`hash_block(parent_hash, block_tokens)`。只哈希本块内容的话，"同内容不同前缀"会被错误命中——两条完全不同的 prompt 若某块 token 恰好相同，后续块就会接错。
- **哈希碰撞防御不可省**：命中后必须复验 `blocks_[b].tokens == 期望内容`。少了这一步，碰撞会静默返回别人的 KV——比崩溃更糟，因为它产出的是"看起来正常的错误答案"。
- **碰撞防御怎么测**：真 FNV-1a 64 位碰撞要试约 2^32 组 token，测试里等于没测。解法是把哈希做成可替换策略（`set_hasher`），测试注入 `return 0x42` 的哈希，毫秒内走到防御分支。变异验证时这条起初**没红**，正是缺这个注入点。
- **变异不红 = 测试缺口，不是实现正确**：本轮 7 个变异，最初 2 个不红（内容复验、满块校验），补测试后才全红。
- **测试用料的坑**：测"块号上界校验"时必须用**没登记过的 token 内容**。第一次我用已登记内容，碰撞防御先返回 0，越界分支根本没走到，变异照样全绿。

## 算子基准实现笔记（2026-11-01，HIP/gfx1100 实测）

- **HIP `memoryClockRate` 语义 ≠ CUDA**：gfx1100 报 1124 MHz，但真实 GDDR6 有效数据率是时钟的 4 倍。用 `2*clock*bus/8` 的 naive 公式算理论带宽会低估 ~8 倍（107.9 vs 实测 716.7 GB/s），所有 kernel 的 % 峰值会虚高到 600%+。**凡用 HIP device prop 算带宽，必须先用 1 GiB D2D 拷贝实测出分母。**
- **实测带宽随拷贝尺寸变化**：L2 内（6 MB 缓存）会虚高到 800–1600 GB/s；1 GiB 稳态 ≈ 716–726 GB/s，才配当分母。
- **两个变体可能实现上是同一个 kernel**：本轮「fused vs three-pass」先 diff 才发现两个 kernel 体实际相同（只差一个局部指针）。**跑对照前必须先 diff 验证变体真的不同**，否则对比轴是假的——与「变异不红=测试缺口」同类静默缺陷。
- **融合 vs 三趟的真实对照**：融合单 kernel（无中间张量，输入基本留在 L2）817 GB/s = 114% of peak；真三趟（max/sum/norm 三个 kernel，row_max/row_sum 落显存）386 GB/s = 54%，**融合省 2.09×**。这就是「算子融合省显存往返」的直接证据。
- **silu float4 收益只有 1.03×**：两个变体都已在带宽上限附近（100% vs 96% of peak），指令数下降的收益被带宽瓶颈吸收——**带宽受限算子不要指望向量化带来大提升**，这是该算子的固有特征，不是实现差。
- **CUDA/HIP 同构移植**：同一份算法描述两份编译（`cuda/` 与 `hip/`），映射规则：`cuda_runtime.h`→`hip/hip_runtime.h`、`__shfl_down_sync(mask,...)`→`__shfl_down(...)`（HIP 无 mask 参数）、`CUDART_INF_F`→`-INFINITY`（用 `math.h`）、`<<<>>>` 语法 HIP 原生支持。这是「异构算子可移植」的最小证明。
