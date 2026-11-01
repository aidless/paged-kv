# micro-vllm 精读：paged KV cache 的最小完整实现

- 仓库：`mukhal/micro-vllm`，默认分支 **master**（不是 main，抓源码时踩过）
- 锁定 commit：`d4c74378b194a67606b362ac9899a471e48d022d`（2026-03-18）
  —— 本笔记所有 `文件:行号` 都相对这个 commit；上游再有提交，行号会漂移
- 取源码的方式：`git clone` 被代理 TLS 切断，改用 GitHub contents API
  （`/repos/.../contents/<path>?ref=master`，base64 解码）取回
- 体量：`engine.py` 547 行、`model.py` 218 行、`attention.py` 105 行
- 它是什么：一个 kilo-scale 的推理引擎，骨架与 vLLM v1 同构——block manager + paged KV cache + 调度 + 采样

## 三问

### 1. 它解决了什么问题？

把"KV cache 按块管理"这条主线打通，且**止步于恰到好处的位置**：所有张量操作真实发生，但没有手写 CUDA kernel。它的选择是把 paged 语义（block table / slot 定位）完整实现，而把 attention 丢给 `torch.nn.functional.scaled_dot_product_attention`。

代价写在 `engine.py:203-218` 的 `gather_kv_for_sequence`：每步 decode 都要把整段历史 K/V 用 `torch.cat` 拼成连续张量。这恰是真实 paged attention kernel 存在的原因——kernel 按 block table 原地索引，不产生这份拼接。

### 2. 数据结构选型，和本仓库 / vLLM 差在哪？

| 维度 | micro-vllm | 本仓库 `aidless/paged-kv` | 差异的含义 |
|---|---|---|---|
| 空闲块来源 | `deque.popleft()`（FIFO，`engine.py:135`） | `vector.pop_back()`（LIFO 栈式） | FIFO 让老块占用更久、降低"正在跑的序列被迫换块"概率；LIFO 复用刚释放的块，缓存局部性好。两者都不是错的，是策略取舍 |
| 共享机制 | **hash 链前缀缓存**（`engine.py:124-130`） | **fork + 引用计数 + COW** | 见下方核心洞察 |
| 共享块是否可变 | **不可变**（只共享满块） | 可变（COW 保护未满块） | 本仓库能覆盖 beam search / 推测解码的未满块共享，micro-vllm 覆盖不了 |
| KV 布局 | 5 维张量 `(blocks, layers, block_size, heads, head_dim)`，block-major | 扁平 1D `float` 数组 + `slot = block*size + offset` | 前者 layer 也进同一张量；后者把 layer 留给调用方按 stride 切 |
| 拼接历史 | `torch.cat` 每层每步（`engine.py:216`） | `gather()` 同样做 O(seq_len) 拷贝 | 两边都有这个缺口，都该由 kernel 消除 |
| 写 KV | Python 逐 token 循环（`engine.py:256-262`） | `memcpy` 批量按槽位写 | 前者在 prefill 长 prompt 时开销明显（每 token 一次 Python 迭代） |
| 失败原子性 | 无（`_allocate_block` 直接 `raise`） | **显式约定**：失败不产生任何副作用 | 本仓库的核心工程约束，见下 |

### 3. 我能带走的一个技巧是什么？

**"只缓存满块"是让共享块免于 COW 的关键设计。**

`engine.py:151-168` 的 `allocate` 直接复用哈希命中的块：命中时只做 `ref_count += 1` 或把它从 free list 摘出，**不做任何数据拷贝**。为什么这样可以？因为可被复用的块必然是"已被写满、内容确定"的块——它的内容由 token 序列唯一决定，之后谁都不会再往里写。

`engine.py:157` 有一道容易忽略的防御：命中哈希后还要比对 `blocks[block_id].token_ids == token_ids`。哈希碰撞是可能的（xxhash64 虽低但非零），碰撞时若不复验就返回别人的 KV，是**静默的错误答案**——比崩溃可怕得多。

一个跨语言的坑，值得记进任何推理实现笔记：**`advance` 只在最后一层调一次**（`model.py:78-79`）。位置计数是序列级属性，若每层都推进，第二层开始 K/V 就写到错位的槽上，且不会报错。

## 与 vLLM v1 的对应关系

| micro-vllm | vLLM v1 |
|---|---|
| `BlockManager` (`engine.py:115`) | `vllm/v1/core/block_pool.py` 的 `BlockPool` + `kv_cache_manager.py` |
| `compute_hash` (`engine.py:124`) | `block_hash.py`（同样用父哈希 + 本块 token 累积） |
| `hash_to_block_id` (`engine.py:120`) | `cached_blocks` 字典 |
| `seq.block_table` | `KVCacheBlocks` |
| —（无 slot mapping） | `slot_mapping` tensor，每 step 由 block table 现算 |
| `PagedKVCache` (`engine.py:185`) | `vllm/v1/worker/gpu_model_runner.py` 的张量分配 |

**注意 micro-vllm 没有 slot_mapping 这个显式产物**：它把位置换算写在 `write_kv` 的循环里（`engine.py:258-260`，`block_idx = pos // block_size`）。本仓库把这一步提取成 `BlockManager::compute_slot_mapping`，是为了让它可单测——这也正是我在这次实现里补上的功能。

## 对本仓库的结论（下一步方向）

1. **prefix caching 是明确缺口** —— **已于 2026-11-01 落地**（`cache_block` / `acquire_prefix` / `evict_one_cached`）。引入 `hash -> block_id` 映射 + 块内容指纹，并采纳 micro-vllm 的约束：**只缓存满块**，否则共享块会和 COW 逻辑打架（COW 每触发一次，前缀缓存就失效一次）。
   实现期与本项目的三处差异：
   - 哈希用 **FNV-1a 64** 而非 xxhash（零外部依赖），并做成**可替换策略**（`set_hasher`）——测试注入必然碰撞的哈希才能验证碰撞防御，真哈希撞一次要 2^32 次尝试。
   - 驱逐是 **FIFO**（`cached_order_` 队首），micro-vllm 的 free list 也是 `popleft()` FIFO；但本仓库只驱逐 `ref_count == 0` 的块，引用中的块挪队尾（vLLM 用 LRU + `evict()`）。
   - 缓存块**引用归零不回空闲链**，留在缓存里等后续命中——这是与 micro-vllm 一致的行为（`engine.py:170-177` 只在 ref 归零时把块**还给 free list**，但保留 hash 映射的做法各家不同；本仓库选择保留映射直到真正驱逐）。
2. **COW 与 prefix cache 是两个互补而非重叠的机制**：
   - prefix cache 解决"两个新请求有相同前缀 → 不重复 prefill"
   - COW 解决"已共享的未满块要写入 → 不污染对方"
   - 前者要求共享块只读，后者专为共享块可变而存在。同一个引擎里两者共存时，满块走 prefix cache（只读共享）、未满块走 fork+COW（写时复制），边界清晰。
3. **`gather` 只该当参考实现**：真实路径必须是 kernel 按 slot_mapping 原地索引。本仓库的 `KVCache::gather` 已在注释里标明这一点，避免后来者把它当性能路径。
4. **失败原子性是对 micro-vllm 的刻意改进**：它用 `raise RuntimeError("Out of KV cache blocks")`（`engine.py:134`）中断，调用方在 `_run_decode` 里 try/except 后把序列标记 `FINISHED`（`engine.py:494-496`）。这是"分配失败就丢弃请求"的策略；本仓库选择"失败无副作用 + 返回 false"，把决策权留给调度器。两种都成立，但后者可测得多。
