# paged-kv

C++ 实现的 PagedAttention 风格 KV cache 块管理器。

大模型推理时，vLLM 的核心贡献是把 KV cache 像操作系统虚拟内存一样**分页管理**：
物理块按需分配、多序列共享、写时复制。这个仓库从零实现其中的**块管理层**——
不含 KV 数据本身，专注生命周期与引用计数的簿记，也就是 vLLM
`vllm/v1/core/block_pool.py` + `block_manager` 那一层做的事。

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
```

语义约定（详见头文件注释）：

- **失败必须无副作用**：`append_token` 在内存不足时返回 `false`，
  序列长度、block table、空闲链全部保持原状——这是 alloc-then-commit 的经典模式。
- **双释放安全**：`free_sequence` 对不存在的序列是 no-op，不会把块放进空闲链两次。
- **COW 触发条件**：写入「未满且被共享（`ref_count > 1`）」的块时，新块接管 table 位置、
  旧块引用 -1。块已满时 append 走正常扩块路径，与共享无关。
- **槽位编号**：`slot = block_id * block_size + offset`，一次查表定位任意 token 的 KV，
  对应真实 kernel 里 `slot_mapping` 数组的取值语义。

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

```bash
make test    # 两个测试套件全绿：ALL PASS
```

测试本身做过变异验证：去掉槽位计算的 `+ offset`、去掉批量写的越界预校验，
测试都会变红（后者直接 `double free or corruption` 崩溃），确认断言不是装饰。

## 对照 vLLM 源码

| 本仓库 | vLLM 对应物 |
|---|---|
| `BlockManager` | `vllm/v1/core/block_pool.py`（BlockPool） |
| `free_list_` | free block queue |
| `fork_sequence` | `fork()`（prefix caching / beam search 复用） |
| COW 路径 | `copy_on_write` + `cow_tracker` |
| `compute_slot_mapping` | `slot_mapping`（每 step 由 block table 现算） |

源码对照笔记见 [notes/](notes/)。

## Roadmap

- [x] 块管理器：分配/回收/COW/引用计数
- [x] 槽位映射 + KV 存储层（decode 闭环可测）
- [ ] CUDA kernels：fused softmax、silu-and-mul，ncu profile
- [ ] prefix caching：`hash -> block_id` 映射（只缓存满块，见 notes）
- [ ] continuous batching 调度器

## 参考

学习时精读过这些项目，设计对照来自它们：

- [vLLM](https://github.com/vllm-project/vllm) —— 块管理的工业标准实现
- [mukhal/micro-vllm](https://github.com/mukhal/micro-vllm) —— 单文件 paged attention
- [psmarter/mini-infer](https://github.com/psmarter/mini-infer) —— 从零推理引擎

## License

MIT
