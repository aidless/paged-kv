#pragma once
// PagedAttention 风格的 KV cache 块管理器
//
// 参考 vLLM BlockManager 的核心思想：
//   - KV cache 按"块"管理：每个物理块容纳 block_size 个 token 的 KV
//   - 每个序列持有"逻辑块 -> 物理块"的映射表（block table）
//   - 多个序列可共享同一物理块（prefix caching / beam search fork）
//   - 写入共享块时触发写时复制（Copy-on-Write, COW）
//
// 本实现只管理块的生命周期与引用计数，不管理 KV 数据本身——
// 数据的拷贝/复用由上层引擎按这里给出的簿记结果执行。

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace kv {

using BlockId = int32_t;  // 物理块编号，kInvalidBlock 表示无效
constexpr BlockId kInvalidBlock = -1;

struct Block {
    int ref_count = 0;  // 当前共享该物理块的序列数
};

class BlockManager {
public:
    BlockManager(int num_blocks, int block_size);

    // ---- 序列生命周期 ----

    // 为 seq 新分配一个逻辑块，追加到它的 block table 尾部。
    // 序列不存在时会被隐式创建。
    // 无空闲物理块时返回 kInvalidBlock，且不产生任何副作用。
    BlockId alloc_block(int seq);

    // fork：子序列共享父序列的全部物理块（每个块引用计数 +1），
    // 语义等价于 beam search 展开或 prefix cache 命中后开新序列。
    // 父序列不存在、或子序列已存在时返回 false。
    bool fork_sequence(int parent, int child);

    // 释放序列占用的全部物理块。序列不存在则为 no-op（防双释放）。
    void free_sequence(int seq);

    // ---- token 追加 ----

    // 向序列追加一个 token：
    //   - 需要新块时分配；内存不足返回 false，且不改变任何状态
    //   - 目标块被共享（ref_count > 1）时触发 COW：
    //     新块接管 table 中的位置，旧块引用计数 -1，cow_count() +1
    //     （数据拷贝由上层引擎完成，这里只做簿记）
    // 序列不存在时返回 false。
    bool append_token(int seq);

    // ---- 查询 ----

    int num_free_blocks() const;
    int num_blocks() const { return num_blocks_; }
    int block_size() const { return block_size_; }
    int seq_len(int seq) const;      // 已追加的 token 数，序列不存在返回 0
    int cow_count() const { return cow_count_; }  // 累计 COW 次数（观察用）
    int ref_count(BlockId b) const;  // 无效/越界块返回 0

    // 序列的物理块表；序列不存在返回空表
    const std::vector<BlockId>& block_table(int seq) const;

private:
    // 取一个空闲物理块；失败返回 kInvalidBlock
    BlockId take_free_block();

    // 引用计数 -1；归零则回收到空闲链
    void dec_ref(BlockId b);

    int num_blocks_;
    int block_size_;
    std::vector<Block> blocks_;
    std::vector<BlockId> free_list_;  // 空闲物理块（栈式复用）

    std::unordered_map<int, std::vector<BlockId>> tables_;  // seq -> 物理块表
    std::unordered_map<int, int> lens_;                     // seq -> token 数
    int cow_count_ = 0;
};

}  // namespace kv
