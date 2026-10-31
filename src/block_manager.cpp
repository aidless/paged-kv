#include "kv/block_manager.h"

#include <stdexcept>

namespace kv {

BlockManager::BlockManager(int num_blocks, int block_size)
    : num_blocks_(num_blocks), block_size_(block_size) {
    if (num_blocks <= 0 || block_size <= 0) {
        throw std::invalid_argument("num_blocks 和 block_size 必须为正数");
    }
    blocks_.resize(num_blocks);
    free_list_.reserve(num_blocks);
    for (int i = num_blocks - 1; i >= 0; --i) {
        free_list_.push_back(i);  // 压栈后 pop 出来的是最小编号
    }
}

BlockId BlockManager::take_free_block() {
    if (free_list_.empty()) return kInvalidBlock;
    BlockId b = free_list_.back();
    free_list_.pop_back();
    return b;
}

void BlockManager::dec_ref(BlockId b) {
    blocks_[b].ref_count -= 1;
    if (blocks_[b].ref_count == 0) {
        free_list_.push_back(b);
    }
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
