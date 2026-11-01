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
#include <deque>
#include <unordered_map>
#include <vector>

namespace kv {

using BlockId = int32_t;  // 物理块编号，kInvalidBlock 表示无效
constexpr BlockId kInvalidBlock = -1;

struct Block {
    int ref_count = 0;  // 当前共享该物理块的序列数
    // 前缀缓存状态：登记过的块内容确定、可跨请求复用。
    // 这类块引用归零后不立刻回空闲链——留给后续请求按前缀命中。
    bool cached = false;
    std::vector<int32_t> tokens;  // 内容指纹，用于哈希碰撞复验
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

    // ---- 槽位映射（块管理器的下游消费者）----

    // 计算 seq 从 start_pos 开始追加 num_new_tokens 个 token 时，
    // 每个 token 应写入的物理槽位编号：slot = block_id * block_size + offset。
    // 块表必须已覆盖到 start_pos + num_new_tokens，否则返回 false 且 *out 不变。
    // 对应 vLLM 的 slot_mapping：prefill 一次算整段 prompt，decode 每步只算 1 个。
    bool compute_slot_mapping(int seq, int64_t start_pos, int64_t num_new_tokens,
                              std::vector<int64_t>* out) const;

    // ---- 前缀缓存（跨请求复用已写满的块）----

    // 把一个**已写满**的块登记为可复用的前缀：块内容由 token_ids 唯一确定。
    // parent_hash 是该块之前所有块的链式哈希（首块传 0），
    // 这样只有整条前缀链都相同才会命中——同内容不同前缀不算命中。
    // 返回该块的累积哈希，供登记下一块时作为 parent_hash。
    // 块号非法返回 0。
    //
    // 只缓存满块是刻意约束：满块之后谁都不会再写它，共享块天然只读，
    // 从而与前缀缓存之外的 COW 逻辑互不干扰（详见 notes/micro-vllm.md）。
    uint64_t cache_block(BlockId b, const std::vector<int32_t>& token_ids,
                         uint64_t parent_hash);

    // 按 tokens 的前缀逐满块查找可复用块，命中者追加到 seq 的 block table
    // （每个命中块引用计数 +1）。末尾不足一块的 token 不参与匹配。
    // 返回命中的满块数；seq 已有内容或块表非空时返回 -1（只对空序列生效）。
    // 命中块保持登记状态，所以第三个请求同样能命中。
    int acquire_prefix(int seq, const std::vector<int32_t>& tokens);

    // 链式哈希：把 parent_hash 与单个块的 token 内容混成新指纹。
    static uint64_t hash_block(uint64_t parent_hash, const std::vector<int32_t>& token_ids);

    // 可替换的哈希策略（默认 FNV-1a 64）。真实引擎里换 xxhash 能再快一截；
    // 测试里换成必然碰撞的实现，才能验证碰撞防御真的生效——
    // 靠真实哈希撞一次需要约 2^32 次尝试，那样这条防御等于没测。
    using Hasher = uint64_t (*)(uint64_t parent_hash, const std::vector<int32_t>& token_ids);
    void set_hasher(Hasher fn);  // 传 nullptr 恢复默认 FNV-1a

    int num_cached_blocks() const { return static_cast<int>(cached_order_.size()); }

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

    // 引用计数 -1；归零则回收到空闲链（前缀缓存块除外：留在缓存里等命中）
    void dec_ref(BlockId b);

    // 池里没有空闲块时，驱逐一个引用归零的缓存块回空闲链。
    // 引用中的缓存块不可驱逐（正在被别的序列读）。成功返回 true。
    bool evict_one_cached();

    int num_blocks_;
    int block_size_;
    std::vector<Block> blocks_;
    std::vector<BlockId> free_list_;  // 空闲物理块（栈式复用）

    std::unordered_map<int, std::vector<BlockId>> tables_;  // seq -> 物理块表
    std::unordered_map<int, int> lens_;                     // seq -> token 数
    int cow_count_ = 0;

    std::unordered_map<uint64_t, BlockId> cached_;  // 累积哈希 -> 物理块
    std::deque<BlockId> cached_order_;              // 登记顺序（FIFO 驱逐）
    Hasher hasher_;                                  // 当前哈希策略
};

}  // namespace kv
