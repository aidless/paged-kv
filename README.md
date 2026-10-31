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

m.alloc_block(1);              // 为序列 1 分配一个逻辑块
m.append_token(1);             // 追加 token；溢出自动扩块
m.fork_sequence(1, 2);         // 序列 2 共享序列 1 的全部块（beam search 语义）
m.append_token(1);             // 写共享块 -> COW：新块接管，旧块引用 -1
m.free_sequence(1);            // 释放；引用归零的块回收到空闲链
```

语义约定（详见头文件注释）：

- **失败必须无副作用**：`append_token` 在内存不足时返回 `false`，
  序列长度、block table、空闲链全部保持原状——这是 alloc-then-commit 的经典模式。
- **双释放安全**：`free_sequence` 对不存在的序列是 no-op，不会把块放进空闲链两次。
- **COW 触发条件**：写入 `ref_count > 1` 的块时，新块接管 table 位置、
  旧块引用 -1。数据拷贝由上层引擎执行，这里只做簿记。

## 测试

覆盖三条最容易写错的边界：

| 边界 | 场景 |
|---|---|
| 块耗尽 | 分配失败后状态零污染，释放后可复用 |
| COW 竞态 | fork 后父子序列交替写入，各自块表正确分叉 |
| 双释放 | 重复 free 不破坏空闲链；重复 fork 被拒绝 |

```bash
make test    # 全绿输出 ALL PASS
```

## 对照 vLLM 源码

| 本仓库 | vLLM 对应物 |
|---|---|
| `BlockManager` | `vllm/v1/core/block_pool.py`（BlockPool） |
| `free_list_` | free block queue |
| `fork_sequence` | `fork()`（prefix caching / beam search 复用） |
| COW 路径 | `copy_on_write` + `cow_tracker` |

源码对照笔记见 [notes/](notes/)。

## Roadmap

- [x] 块管理器：分配/回收/COW/引用计数（本仓库）
- [ ] CUDA kernels：fused softmax、silu-and-mul，ncu profile
- [ ] 单文件 paged attention 前向（对标 micro-vllm）
- [ ] continuous batching 调度器

## 参考

学习时精读过这些项目，设计对照来自它们：

- [vLLM](https://github.com/vllm-project/vllm) —— 块管理的工业标准实现
- [mukhal/micro-vllm](https://github.com/mukhal/micro-vllm) —— 单文件 paged attention
- [psmarter/mini-infer](https://github.com/psmarter/mini-infer) —— 从零推理引擎

## License

MIT
