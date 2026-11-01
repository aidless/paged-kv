#include "kv/block_manager.h"

#include <stdexcept>

namespace kv {

BlockManager::BlockManager(int num_blocks, int block_size)
    : num_blocks_(num_blocks), block_size_(block_size) {
    if (num_blocks <= 0 || block_size <= 0) {
        throw std::invalid_argument("num_blocks 和 block_size 必须为正数");
    }
    hasher_ = &BlockManager::hash_block;  // 默认 FNV-1a，可被 set_hasher 替换
    blocks_.resize(num_blocks);
    free_list_.reserve(num_blocks);
    for (int i = num_blocks - 1; i >= 0; --i) {
        free_list_.push_back(i);  // 压栈后 pop 出来的是最小编号
    }
}

BlockId BlockManager::take_free_block() {
    // 池里没空闲块时，唯一还能腾出来的来源是引用归零的前缀缓存块
    if (free_list_.empty() && !evict_one_cached()) return kInvalidBlock;
    if (free_list_.empty()) return kInvalidBlock;
    BlockId b = free_list_.back();
    free_list_.pop_back();
    return b;
}

void BlockManager::dec_ref(BlockId b) {
    blocks_[b].ref_count -= 1;
    if (blocks_[b].ref_count <= 0) {
        blocks_[b].ref_count = 0;
        // 前缀缓存块即使没人引用也留在缓存里——它的价值是能被后续请求命中，
        // 立刻收回空闲链等于把刚攒下来的 prefill 成果扔掉。
        if (blocks_[b].cached) return;
        free_list_.push_back(b);
    }
}

bool BlockManager::evict_one_cached() {
    // 只驱逐引用归零的缓存块；正在被序列读的块不能动，否则对方会读到别人的 KV
    size_t attempts = cached_order_.size();
    while (attempts-- > 0 && !cached_order_.empty()) {
        BlockId b = cached_order_.front();
        cached_order_.pop_front();
        if (!blocks_[b].cached) continue;  // 已被处置过
        if (blocks_[b].ref_count > 0) {
            cached_order_.push_back(b);  // 在用：挪到队尾，这轮先看别人
            continue;
        }
        for (auto it = cached_.begin(); it != cached_.end();) {
            if (it->second == b) {
                it = cached_.erase(it);
            } else {
                ++it;
            }
        }
        blocks_[b].cached = false;
        blocks_[b].tokens.clear();
        free_list_.push_back(b);
        return true;
    }
    return false;
}

uint64_t BlockManager::hash_block(uint64_t parent_hash, const std::vector<int32_t>& token_ids) {
    // FNV-1a 64 位：先把父哈希混进指纹（绑定前缀链），再混入本块 token。
    // 不用 xxhash 是为了零外部依赖——块哈希不需要密码学强度，安全靠下面的内容复验。
    constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
    constexpr uint64_t kFnvPrime = 1099511628211ULL;

    uint64_t h = kFnvOffset;
    for (int i = 0; i < 8; ++i) {
        h ^= (parent_hash >> (i * 8)) & 0xFFULL;
        h *= kFnvPrime;
    }
    for (int32_t t : token_ids) {
        const uint32_t v = static_cast<uint32_t>(t);
        for (int i = 0; i < 4; ++i) {
            h ^= (v >> (i * 8)) & 0xFFU;
            h *= kFnvPrime;
        }
    }
    return h;
}

void BlockManager::set_hasher(Hasher fn) {
    hasher_ = (fn == nullptr) ? &BlockManager::hash_block : fn;
}

uint64_t BlockManager::cache_block(BlockId b, const std::vector<int32_t>& token_ids,
                                   uint64_t parent_hash) {
    if (b < 0 || b >= num_blocks_) return 0;
    if (static_cast<int>(token_ids.size()) != block_size_) return 0;  // 只缓存满块

    const uint64_t h = hasher_(parent_hash, token_ids);

    // 哈希碰撞防御：同一指纹已属于别的块时，保留先登记的那个，不覆盖。
    // 覆盖会让后来者拿到内容不符的 KV —— 静默的错误答案，比崩溃更糟。
    auto it = cached_.find(h);
    if (it != cached_.end() && it->second != b) return 0;

    if (!blocks_[b].cached) {
        blocks_[b].cached = true;
        cached_order_.push_back(b);
    }
    blocks_[b].tokens = token_ids;
    cached_[h] = b;
    return h;
}

int BlockManager::acquire_prefix(int seq, const std::vector<int32_t>& tokens) {
    auto tit = tables_.find(seq);
    if (tit != tables_.end() && !tit->second.empty()) return -1;  // 只对空序列生效
    if (lens_.count(seq) && lens_[seq] != 0) return -1;

    uint64_t h = 0;
    int matched = 0;
    std::vector<BlockId> acquired;

    for (size_t off = 0; off + static_cast<size_t>(block_size_) <= tokens.size();
         off += static_cast<size_t>(block_size_)) {
        std::vector<int32_t> bt(tokens.begin() + static_cast<long>(off),
                                tokens.begin() + static_cast<long>(off) + block_size_);
        h = hasher_(h, bt);
        auto it = cached_.find(h);
        if (it == cached_.end()) break;
        const BlockId b = it->second;
        // 内容复验：哈希相同不代表内容相同。少了这一步，碰撞会返回别人的 KV。
        if (blocks_[b].tokens != bt) break;

        blocks_[b].ref_count += 1;
        acquired.push_back(b);
        ++matched;
    }

    if (matched > 0) {
        auto& table = tables_[seq];
        for (BlockId b : acquired) table.push_back(b);
        lens_[seq] = matched * block_size_;
    }
    return matched;
}

BlockId BlockManager::alloc_block(int seq) {
    BlockId b = take_free_block();
    if (b == kInvalidBlock) return kInvalidBlock;
    blocks_[b].ref_count = 1;
    tables_[seq].push_back(b);
    return b;
}

bool BlockManager::fork_sequence(int parent, int child) {
    auto it = tables_.find(parent);
    if (it == tables_.end()) return false;      // 父序列不存在
    if (tables_.count(child) || lens_.count(child)) return false;  // 子序列已存在

    // 子序列共享父序列全部物理块，引用计数逐块 +1
    tables_[child] = it->second;
    for (BlockId b : tables_[child]) {
        blocks_[b].ref_count += 1;
    }
    auto lit = lens_.find(parent);
    lens_[child] = (lit == lens_.end()) ? 0 : lit->second;
    return true;
}

void BlockManager::free_sequence(int seq) {
    auto it = tables_.find(seq);
    if (it == tables_.end()) return;  // 双释放 / 不存在：no-op

    for (BlockId b : it->second) {
        dec_ref(b);
    }
    tables_.erase(it);
    lens_.erase(seq);
}

bool BlockManager::append_token(int seq) {
    auto it = tables_.find(seq);
    if (it == tables_.end()) return false;  // 未初始化的序列不能直接追加

    int tokens = lens_[seq] + 1;
    int need = (tokens + block_size_ - 1) / block_size_;  // ceil 除法

    if (need > static_cast<int>(it->second.size())) {
        // 溢出到新块；分配失败时不改任何状态
        BlockId b = take_free_block();
        if (b == kInvalidBlock) return false;
        blocks_[b].ref_count = 1;
        it->second.push_back(b);
    } else {
        // 写入最后一个块；若被共享则触发 COW
        BlockId last = it->second.back();
        if (blocks_[last].ref_count > 1) {
            BlockId nb = take_free_block();
            if (nb == kInvalidBlock) return false;  // COW 无处落新块
            blocks_[nb].ref_count = 1;              // 新块只归本序列
            dec_ref(last);                          // 旧块不再归本序列
            it->second.back() = nb;
            ++cow_count_;
            // 数据拷贝由上层引擎完成，这里只做簿记
        }
    }

    lens_[seq] = tokens;
    return true;
}

bool BlockManager::compute_slot_mapping(int seq, int64_t start_pos, int64_t num_new_tokens,
                                        std::vector<int64_t>* out) const {
    if (out == nullptr || start_pos < 0 || num_new_tokens < 0) return false;
    auto it = tables_.find(seq);
    if (it == tables_.end()) return false;  // 未初始化的序列没有块表

    const std::vector<BlockId>& table = it->second;
    const int64_t end_pos = start_pos + num_new_tokens;
    const int64_t needed_blocks = (end_pos + block_size_ - 1) / block_size_;
    if (needed_blocks > static_cast<int64_t>(table.size())) return false;

    std::vector<int64_t> slots;
    slots.reserve(static_cast<size_t>(num_new_tokens));
    for (int64_t i = 0; i < num_new_tokens; ++i) {
        const int64_t pos = start_pos + i;
        const int64_t block_idx = pos / block_size_;
        const int64_t offset = pos % block_size_;
        slots.push_back(static_cast<int64_t>(table[static_cast<size_t>(block_idx)]) * block_size_ +
                        offset);
    }
    *out = std::move(slots);  // 全部算成功后才改动调用方数据
    return true;
}

int BlockManager::num_free_blocks() const {
    return static_cast<int>(free_list_.size());
}

int BlockManager::seq_len(int seq) const {
    auto it = lens_.find(seq);
    return (it == lens_.end()) ? 0 : it->second;
}

int BlockManager::ref_count(BlockId b) const {
    if (b < 0 || b >= num_blocks_) return 0;
    return blocks_[b].ref_count;
}

const std::vector<BlockId>& BlockManager::block_table(int seq) const {
    static const std::vector<BlockId> kEmpty;
    auto it = tables_.find(seq);
    return (it == tables_.end()) ? kEmpty : it->second;
}

}  // namespace kv
