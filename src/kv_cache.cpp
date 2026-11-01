#include "kv/kv_cache.h"

#include <cstring>
#include <stdexcept>

namespace kv {

KVCache::KVCache(int num_blocks, int block_size, int head_dim)
    : block_size_(block_size), head_dim_(head_dim) {
    if (num_blocks <= 0 || block_size <= 0 || head_dim <= 0) {
        throw std::invalid_argument("num_blocks / block_size / head_dim 必须为正数");
    }
    num_slots_ = num_blocks * block_size;
    data_.assign(static_cast<size_t>(num_slots_) * head_dim_, 0.0f);
}

bool KVCache::write(int64_t slot, const float* src) {
    if (slot < 0 || slot >= num_slots_ || src == nullptr) return false;
    std::memcpy(data_.data() + slot * head_dim_, src, sizeof(float) * head_dim_);
    return true;
}

const float* KVCache::read(int64_t slot) const {
    if (slot < 0 || slot >= num_slots_) return nullptr;
    return data_.data() + slot * head_dim_;
}

int KVCache::write_tokens(const std::vector<int64_t>& slots, const float* src, int num_floats) {
    if (src == nullptr) return -1;
    // 期望的浮点数个数必须严格匹配，防止按错的维度半径写坏相邻槽位
    if (num_floats != static_cast<int>(slots.size()) * head_dim_) return -1;

    // 先整体校验，再动手写：任一槽位非法就不写一个字节
    for (int64_t slot : slots) {
        if (slot < 0 || slot >= num_slots_) return -1;
    }

    for (size_t i = 0; i < slots.size(); ++i) {
        std::memcpy(data_.data() + slots[i] * head_dim_,
                    src + static_cast<size_t>(i) * head_dim_,
                    sizeof(float) * head_dim_);
    }
    return static_cast<int>(slots.size());
}

int KVCache::gather(const std::vector<BlockId>& block_table, int seq_len, float* dst) const {
    if (dst == nullptr || seq_len < 0) return -1;
    // 块表必须覆盖 seq_len 个 token
    if (static_cast<size_t>(seq_len) > block_table.size() * static_cast<size_t>(block_size_)) {
        return -1;
    }

    for (int pos = 0; pos < seq_len; ++pos) {
        int block_idx = pos / block_size_;
        int offset = pos % block_size_;
        BlockId b = block_table[static_cast<size_t>(block_idx)];
        int64_t slot = static_cast<int64_t>(b) * block_size_ + offset;
        if (slot < 0 || slot >= num_slots_) return -1;  // 块表里混进了非法块号
        std::memcpy(dst + static_cast<size_t>(pos) * head_dim_,
                    data_.data() + slot * head_dim_,
                    sizeof(float) * head_dim_);
    }
    return seq_len;
}

}  // namespace kv
