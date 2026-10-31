# 源码对照笔记

本目录存放精读参考项目时的笔记，每篇对照本仓库实现，标注 `文件:行号`。

## 精读顺序

1. **mukhal/micro-vllm** —— 单文件 paged attention，第一个通读
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
