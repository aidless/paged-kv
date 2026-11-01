#pragma once
// KV 数据的扁平存储层
//
// 块管理器（BlockManager）只做簿记，不碰数据；这个类才是真正存 K/V 的地方。
// 两者通过"物理槽位编号"解耦：
//
//     slot = block_id * block_size + offset_in_block
//
// 这样一个 block table 查表就能定位任意 token 的 KV，不需要二维下标，
// 也正是真实 paged attention kernel 里 slot_mapping 数组的取值语义。
//
// 典型调用链（一次 decode step）：
//   1. BlockManager::append_token(seq)              追加 1 个 token 的逻辑簿记
//   2. BlockManager::compute_slot_mapping(...)       算出新 token 落到哪个物理槽位
//   3. KVCache::write_tokens(slots, k_or_v)          把这一步的 KV 真正写进去
//   4. KVCache::gather(block_table, seq_len, dst)    拼出整段历史（参考实现，供 attention 用）

#include <cstdint>
#include <vector>

#include "kv/block_manager.h"

namespace kv {

class KVCache {
public:
    // num_blocks / block_size / head_dim 都必须为正数，否则抛 std::invalid_argument。
    // head_dim 是单个槽位的元素数（多头的扁平化由调用方按 stride 约定处理）。
    KVCache(int num_blocks, int block_size, int head_dim);

    int num_slots() const { return num_slots_; }
    int head_dim() const { return head_dim_; }
    int block_size() const { return block_size_; }

    // 写一个槽位。槽位越界返回 false——坏写会静默踩坏别的序列，必须拦住。
    bool write(int64_t slot, const float* src);

    // 读一个槽位的起始指针；槽位越界返回 nullptr。
    const float* read(int64_t slot) const;

    // 按 slots 逐个写入，src 连续排列（slots.size() * head_dim 个 float）。
    // 遵循本项目的失败原子性约定：任一槽位越界、或 num_floats 与
    // slots.size() * head_dim 不符，则整体失败返回 -1，且一个字节都不写。
    // 成功返回写入的槽位数。
    int write_tokens(const std::vector<int64_t>& slots, const float* src, int num_floats);

    // 按 block_table 拼出前 seq_len 个 token 的 KV，连续写入 dst。
    // 块表不足以覆盖 seq_len 时返回 -1，且不改动 dst。
    // 这是给没有 paged kernel 的参考路径用的（每步 O(seq_len) 拷贝）；
    // 真实 kernel 应当原地按 slot_mapping 索引，不产生这份拼接。
    int gather(const std::vector<BlockId>& block_table, int seq_len, float* dst) const;

private:
    int block_size_;
    int head_dim_;
    int num_slots_;
    std::vector<float> data_;
};

}  // namespace kv
