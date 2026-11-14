# paged-kv

[![CI](https://github.com/aidless/paged-kv/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/paged-kv/actions/workflows/ci.yml)

C++ 实现的 PagedAttention 风格 KV cache 块管理器。

大模型推理时，vLLM 的核心贡献是把 KV cache 像操作系统虚拟内存一样**分页管理**：
物理块按需分配、多序列共享、写时复制。这个仓库从零实现其中的**块管理层**——
不含 KV 数据本身，专注生命周期与引用计数的簿记，也就是 vLLM
`vllm/v1/core/block_pool.py` + `block_manager` 那一层做的事。

**验证过什么**：`make test` 三个 C++ 套件全绿（74 + 133 checks，覆盖 COW 引用计数、跨序列共享、prefix 命中、OOM 时失败无副作用）。

**怎么跑**：`make test`，只需 g++ 与 make，无第三方依赖；不需要 GPU。

## 为什么值得手写

面试推理引擎岗时，「PagedAttention 是什么」人人能答；
「COW 触发时新旧块的引用计数怎么变、内存不足时失败必须无副作用」只有写过才答得出。
这个项目就是这条边界的产物。

## 核心 API

```cpp
kv::BlockManager m(/*num_blocks=*/16, /*block_size=*/8);
kv::KVCache cache(/*num_blocks=*/16, /*block_size=*/8, /*head_dim=*/64);

m.alloc_block(1);              // 为序列 1 分配一个逻辑块
m.append_token(1);             // 追加 token；溢出自动扩块
m.fork_sequence(1, 2);         // 序列 2 共享序列 1 的全部块（beam search 语义）
m.append_token(1);             // 写共享块 -> COW：新块接管，旧块引用 -1
m.free_sequence(1);            // 释放；引用归零的块回收到空闲链

// 块管理器如何真正服务一次 decode step：
std::vector<int64_t> slots;
m.compute_slot_mapping(2, /*start_pos=*/0, /*num_new_tokens=*/1, &slots);
cache.write_tokens(slots, kv_data, /*num_floats=*/64);   // 写进物理槽位
cache.gather(m.block_table(2), m.seq_len(2), out);       // 拼回整段历史

// prefix caching：相同前缀不重复 prefill、不重复占显存
uint64_t h = m.cache_block(block_id, block_tokens, /*parent_hash=*/0);  // 登记满块
int hit = m.acquire_prefix(new_seq, prompt_tokens);   // 按整块步进命中，返回块数
```

语义约定（详见头文件注释）：

- **失败必须无副作用**：`append_token` 在内存不足时返回 `false`，
  序列长度、block table、空闲链全部保持原状——这是 alloc-then-commit 的经典模式。
- **双释放安全**：`free_sequence` 对不存在的序列是 no-op，不会把块放进空闲链两次。
- **COW 触发条件**：写入「未满且被共享（`ref_count > 1`）」的块时，新块接管 table 位置、
  旧块引用 -1。块已满时 append 走正常扩块路径，与共享无关。
- **槽位编号**：`slot = block_id * block_size + offset`，一次查表定位任意 token 的 KV，
  对应真实 kernel 里 `slot_mapping` 数组的取值语义。
- **只缓存满块**：`cache_block` 拒绝不足 `block_size` 的 token。满块内容由 token 序列唯一确定，
  之后谁都不会再写它，共享块天然只读——这样 prefix cache 与 COW 不打架。
  未满块的共享走 `fork_sequence` + COW，两者互补不重叠。
- **缓存块引用归零不回空闲链**：它留在缓存里等后续命中，只在池子真正用完时被
  `evict_one_cached` 驱逐（且只驱逐 `ref_count == 0` 的）。

## 测试

覆盖最容易写错的边界：

| 边界 | 场景 |
|---|---|
| 块耗尽 | 分配失败后状态零污染，释放后可复用 |
| COW 竞态 | fork 后多序列交替写入，各自块表正确分叉 |
| COW 上限 | 三方共享时只触发 2 次 COW（最后持有者原地写） |
| 双释放 | 重复 free 不破坏空闲链；重复 fork 被拒绝 |
| 槽位越界 | 批量写入混入非法槽位时一个字节都不写 |
| decode 闭环 | 追加 -> 算槽位 -> 写 KV -> 拼回，跨块边界正确 |
| 前缀链不匹配 | 同内容不同父哈希不算命中（链式哈希绑定整条前缀） |
| 缓存块存活 | 序列释放后缓存块仍在缓存、仍可被第三个请求命中 |
| 驱逐 | 池子用完时 FIFO 驱逐缓存块；引用中的缓存块不可驱逐 |
| 哈希碰撞 | 注入必然碰撞的哈希后，内容复验必须拦住假命中 |

```bash
make test    # 三个测试套件全绿：block_manager / kv_cache / prefix_cache
```

测试本身做过变异验证（9 个变异，全部变红），确认断言不是装饰：

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

## 仓库结构

```
include/kv/block_manager.h   BlockManager: 分配/回收/COW/引用计数/前缀缓存接口
include/kv/kv_cache.h        KVCache: 槽位映射 + 物理 KV 存储（decode 闭环）
src/block_manager.cpp        实现（含 FNV-1a 链式哈希 + FIFO 驱逐）
src/kv_cache.cpp             实现
tests/test_block_manager.cpp 块管理边界（耗尽/COW/双释放/驱逐）
tests/test_kv_cache.cpp      槽位映射 + decode 路径（74 checks）
tests/test_prefix_cache.cpp  链式哈希命中 + 驱逐（133 checks）
cuda/                        CUDA 融合算子（fused softmax / silu-and-mul）+ benchmark
hip/                         HIP 同构移植（hipcc 编译，gfx1100 实测）
notes/                       micro-vLLM 源码对照笔记 + 实测方法论
Makefile                     make test（三套件） / make clean
```

## 对照 vLLM 源码

| 本仓库 | vLLM 对应物 |
|---|---|
| `BlockManager` | `vllm/v1/core/block_pool.py`（BlockPool） |
| `free_list_` | free block queue |
| `fork_sequence` | `fork()`（prefix caching / beam search 复用） |
| COW 路径 | `copy_on_write` + `cow_tracker` |
| `compute_slot_mapping` | `slot_mapping`（每 step 由 block table 现算） |
| `cache_block` / `acquire_prefix` | prefix caching：`cache_full_blocks` + `get_computed_blocks` |
| `hash_block`（FNV-1a 链式） | `hash_request_tokens`（用 xxhash，同样链式挂父哈希） |
| `evict_one_cached` | `evict()`（LRU/FIFO 释放已无引用的缓存块） |

源码对照笔记见 [notes/](notes/)。

## Roadmap

- [x] 块管理器：分配/回收/COW/引用计数
- [x] 槽位映射 + KV 存储层（decode 闭环可测）
- [x] CUDA kernels：fused softmax、silu-and-mul，benchmark（HIP 实测，见下方）
- [x] prefix caching：链式哈希命中 + FIFO 驱逐（只缓存满块，见 notes/micro-vllm.md）
- [ ] continuous batching 调度器

## 算子融合基准（HIP / gfx1100）

> **证据边界**：下面的数字来自作者机器上一次 HIP 运行，**原始输出没有随仓发布**。
> 仓内可复现的是 `make test` 那三个 C++ 套件（74 + 133 checks），以及 `hip/`、`cuda/`
> 两份源码本身；带 `hipcc` / `nvcc` 的机器可以重跑 `hip/Makefile` 自行取数。
> 引用这组带宽数时请带上这个前提。

对拍（CPU 参考）全部 PASS，`max|diff|`：softmax 1.86e-09、silu 1.91e-06（容差 1e-4）。
分母不用规格常数，用 **1 GiB D2D 拷贝实测上限**（`peak_bw_method=measured_d2d_1gib`），
因为 HIP 的 `memoryClockRate` 语义与 CUDA 不一致（gfx1100 报 1124 MHz 但有效数据率是 4 倍，
naive 公式低估 ~8 倍）。

| 算子 | 变体 | 耗时 | 有效带宽 | 峰值占比 |
|------|------|------|----------|----------|
| softmax (4096×1024) | fused 单 kernel | 0.0415 ms | 817 GB/s | 114% |
| | three-pass（中间量落显存） | 0.0869 ms | 386 GB/s | 54% |
| silu-and-mul (n≈45.1M) | float4 向量化 | 0.755 ms | 716 GB/s | 100% |
| | scalar | 0.780 ms | 690 GB/s | 96% |

- **softmax 融合 2.09×**：单 kernel 无中间张量，输入基本留在 L2（>100% 正常）；
  three-pass 每行读 3 次 + row_max/row_sum 写读，DRAM 流量约为融合的 2 倍。
- **silu 向量化 1.03×**：float4 每条指令搬 16B，已近峰值带宽，剩余收益来自指令数下降
  （标量版也已在带宽上限附近，故提升小——这正是带宽受限算子的特征）。
- 代码：`cuda/`（CUDA 版）与 `hip/`（HIP 同构移植，`hipcc` 编译），同一份算法两份编译，
  是「异构算子可移植」的最小证明。CUDA 侧未在 NVIDIA 硬件验证（本机无 nvcc）。

实现期关键教训（写进 [notes/](notes/)）：
- 两个变体可能是同一个 kernel——必须先 diff 验证变体真的不同，再跑对照；
- 实测 D2D 带宽随拷贝尺寸变化（L2 内 6MB 会虚高到 800–1600 GB/s），分母必须用 1 GiB 稳态值；
- printf 多参数求值顺序在 GCC 下从右到左，带副作用的调用与读状态混在同一 printf 会读错。

## 参考

学习时精读过这些项目，设计对照来自它们：

- [vLLM](https://github.com/vllm-project/vllm) —— 块管理的工业标准实现
- [mukhal/micro-vllm](https://github.com/mukhal/micro-vllm) —— 单文件 paged attention
- [psmarter/mini-infer](https://github.com/psmarter/mini-infer) —— 从零推理引擎

## 验收题库与复算

`eval/cases.jsonl` 是 20 条可执行验收题，覆盖块管理器五个面的关键口径
（分配/回收、按需扩块、COW 引用计数、槽位映射失败原子性、KV 读写与 gather），
以及前缀缓存的四类边界（链式哈希、整块命中、尾部不匹配、FIFO 驱逐、哈希碰撞复验），
加上构造函数入参校验与 C++ 测试套件的实际 check 数。**17 条是 `expr`/`raises`，
期望值不是抄文档，而是真编译 `src/` 并调用 `BlockManager` / `KVCache` 现算出来的**
（探针源码内联在 `verify` 里，用 `g++ -std=c++17 -Iinclude` 编到临时目录后运行；
对象文件与探针产物缓存在 `$TMPDIR/paged-kv-eval-cache`，第二次起不再重编）：

```bash
python3 eval/run_cases.py            # 20/20 cases passed（需要 g++）
python3 eval/run_cases.py --save     # 顺带把结果写进 eval/run-report.json
python3 eval/run_cases.py --selftest # 篡改一条期望值，必须变红（19/20，证明这套题会失败）
```

**数字如何复算**：`eval/run-report.json` 里每条 case 都带 `id` / `axis` / `detail`，
把 `detail` 与 `cases.jsonl` 的 `verify` 代码一起看，就能从源码复现该数字；
`--save` 重跑一次即可重建整份报告，不需要手工抄数。题面用到的常量都能在仓内对上：
`hash_block` 的 FNV-1a 64 值可在任何语言里独立重算（先混父哈希 8 字节、再逐 token 混
4 字节小端），三个 C++ 套件的 check 数以 `make test` 的打印为准（74 / 133）。

**失败归因**：`run_cases.py` 不只报通过率，失败时按 `axis` 汇总（例如
`block/cow: 1 条 —— PV-05`），直接指出是哪个模块的哪个口径坏了。
按轴落点：`block/alloc`、`block/append`、`block/cow`、`block/free` 归 `src/block_manager.cpp`
的分配/扩块/引用计数路径；`kv/slot`、`kv/decode` 归 `src/kv_cache.cpp` 的槽位映射与 gather；
`prefix/hash`、`prefix/hit`、`prefix/evict` 归 FNV-1a 链式哈希、满块命中与 FIFO 驱逐；
`api/ctor`、`doc/prereg` 归构造入参校验与文档口径。

## 验收判据与统计口径

**验收判据（什么算过）**：`python3 eval/run_cases.py` 必须 **20/20** 全通过（通过率
**≥ 100%**，任一条红即判定不达标）；`make test` 三个 C++ 套件必须全绿，其中
`test_kv_cache` 74 checks、`test_prefix_cache` 133 checks，合计 **≥ 207 checks** 零失败。
两个条件同时成立才算本仓当前状态达标，缺一不可。

**统计口径**：验收样本量 **n = 20**（`eval/cases.jsonl` 的 20 条题；逐条结果在
`eval/run-report.json` 的 `results` 数组里，`total` 字段即 n）。按 Wilson 区间算
20/20 的 **95% CI = [0.839, 1.000]**（通过率的区间估计；复算命令如下，纯标准库）：

```bash
python3 - <<'EOF'
import math
n, k, z = 20, 20, 1.959963984540054      # n = cases.jsonl 条数；k = 通过条数
p = k / n
den = 1 + z*z/n
c = (p + z*z/(2*n)) / den
h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
print(round(c-h, 4), round(c+h, 4))       # 0.8389 1.0
EOF
```

**成本/延迟**：整套验收题在本机的墙钟是**首次 10.4 s、命中编译缓存后 0.17 s**
（`rm -rf ${TMPDIR:-/tmp}/paged-kv-eval-cache && /usr/bin/time -f %e python3 eval/run_cases.py`，
再跑一次取第二次；差别全在 `g++ -O0` 编译 `src/` 与 17 个内联探针上）。
算子侧的延迟数见下方「算子融合基准」表的 0.0415 / 0.0869 / 0.755 / 0.780 ms，
来自作者机器一次 HIP 运行；`make eval` 一条命令即可同时重跑题库与三套件。

**数字边界（哪些自造、哪些引自何处）**：README 里所有 C++ 行为数字（块号、引用计数、
seq_len、check 数 74/133、20/20）都是**本仓自己的**代码现跑出来的，不引自 vLLM 或任何
论文；算子带宽/延迟是**作者在 gfx1100 上自测**，非引自厂商规格表，也非原图转载，原始
输出未随仓发布；「参考」一节列出的三个项目只用于设计对照，**没有复制其源码**；
`notes/micro-vllm.md` 里标注为「作者整理」的段落是阅读后的归纳，不是原文摘录。

## License

MIT
