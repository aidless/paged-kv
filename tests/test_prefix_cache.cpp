// 前缀缓存（prefix caching）测试
//
// 解决的问题：两个新请求如果有相同的 prompt 前缀，不该重复 prefill、也不该
// 重复占 KV 显存。vLLM v1 的做法是给每个"已写满"的块算一个链式哈希
// （父哈希 + 本块 token 内容），用哈希命中复用物理块。
//
// 关键设计约束（来自 micro-vllm 精读，见 notes/micro-vllm.md）：
//   只缓存 **满块**。满块的内容由 token 序列唯一确定，之后谁都不会再写它，
//   所以共享块天然只读、不需要 COW。这条约束让 prefix cache 与 COW 不打架：
//   命中前缀后 seq_len 必然是 block_size 的整数倍，下一个 append 直接走扩块路径。
//
// 与 COW 的分工：
//   prefix cache：未满块之外，跨请求只读共享（省 prefill、省显存）
//   COW        ：已经共享的未满块要写入时，拷走不污染对方

#include "kv/block_manager.h"

#include <cstdio>
#include <vector>

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        ++g_checks;                                                       \
        if (!(cond)) {                                                    \
            ++g_failed;                                                   \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
        }                                                                 \
    } while (0)

// ---- 1. 命中整条前缀：两个请求共享同一批物理块 ----
static void test_full_prefix_hit() {
    kv::BlockManager m(8, 4);

    // 请求 A：16 个 token = 4 个满块，全部登记为可复用前缀
    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 16; ++i) CHECK(m.append_token(1));
    CHECK(m.block_table(1).size() == 4);

    std::vector<int32_t> tokens;
    for (int i = 0; i < 16; ++i) tokens.push_back(100 + i);

    // 逐块登记（父哈希串成链）
    uint64_t h = 0;
    for (int b = 0; b < 4; ++b) {
        std::vector<int32_t> block_tokens(tokens.begin() + b * 4, tokens.begin() + b * 4 + 4);
        h = m.cache_block(m.block_table(1)[static_cast<size_t>(b)], block_tokens, h);
        CHECK(h != 0);
    }
    CHECK(m.num_cached_blocks() == 4);

    // 请求 B 有完全相同的 16 token 前缀 -> 零新块，全部命中
    CHECK(m.acquire_prefix(2, tokens) == 4);
    CHECK(m.block_table(2).size() == 4);
    CHECK(m.block_table(2) == m.block_table(1));
    CHECK(m.num_free_blocks() == 4);            // 一个块都没新占
    CHECK(m.ref_count(m.block_table(2)[0]) == 2);  // 共享，引用 +1

    // 命中的块仍在缓存表里：第三个请求还能继续命中
    CHECK(m.acquire_prefix(3, tokens) == 4);
    CHECK(m.ref_count(m.block_table(2)[0]) == 3);
}

// ---- 2. 只命中共同前缀，分叉点之后不命中 ----
static void test_partial_prefix_hit() {
    kv::BlockManager m(8, 4);

    std::vector<int32_t> tokens_a = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 8; ++i) CHECK(m.append_token(1));

    uint64_t h = 0;
    for (int b = 0; b < 2; ++b) {
        std::vector<int32_t> bt(tokens_a.begin() + b * 4, tokens_a.begin() + b * 4 + 4);
        h = m.cache_block(m.block_table(1)[static_cast<size_t>(b)], bt, h);
    }

    // 请求 B 第 1 块相同、第 2 块不同
    std::vector<int32_t> tokens_b = {1, 2, 3, 4, 9, 9, 9, 9};
    CHECK(m.acquire_prefix(2, tokens_b) == 1);   // 只命中首块
    CHECK(m.block_table(2).size() == 1);
    CHECK(m.seq_len(2) == 4);
    CHECK(m.num_free_blocks() == 6);
}

// ---- 3. 只缓存满块：不满的前缀不参与匹配 ----
// acquire_prefix 只按整块步进，末尾不足一块的部分永远不会命中
static void test_tail_not_cached() {
    kv::BlockManager m(8, 4);

    std::vector<int32_t> full = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 8; ++i) CHECK(m.append_token(1));

    uint64_t h = 0;
    for (int b = 0; b < 2; ++b) {
        std::vector<int32_t> bt(full.begin() + b * 4, full.begin() + b * 4 + 4);
        h = m.cache_block(m.block_table(1)[static_cast<size_t>(b)], bt, h);
    }

    // 请求 B 有 10 个 token：前 8 个能命中 2 块，尾部 2 个不参与
    std::vector<int32_t> longer = full;
    longer.push_back(99);
    longer.push_back(100);
    CHECK(m.acquire_prefix(2, longer) == 2);
    CHECK(m.seq_len(2) == 8);  // 只认整块

    // 命中满块后继续 append：必然需要新块，不能触发 COW
    int cow_before = m.cow_count();
    CHECK(m.append_token(2));
    CHECK(m.cow_count() == cow_before);   // 共享块是满块 -> 不需要 COW
    CHECK(m.block_table(2).size() == 3);
}

// ---- 4. 父哈希不同则链断：同内容不同前缀链不命中 ----
static void test_chain_must_match() {
    kv::BlockManager m(8, 4);

    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 4; ++i) CHECK(m.append_token(1));
    // 以一个非零父哈希登记块 0（模拟"这不是序列的第一个块"）
    std::vector<int32_t> bt = {1, 2, 3, 4};
    CHECK(m.cache_block(m.block_table(1)[0], bt, 0xABCDEFULL) != 0);

    // 新请求从 parent=0 开始匹配，链条对不上 -> 不命中
    CHECK(m.acquire_prefix(2, bt) == 0);
    CHECK(m.block_table(2).empty());
}

// ---- 5. 释放后前缀仍可复用（prefix cache 的核心价值）----
static void test_cached_survives_free() {
    kv::BlockManager m(8, 4);

    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 8; ++i) CHECK(m.append_token(1));

    std::vector<int32_t> tokens;
    for (int i = 0; i < 8; ++i) tokens.push_back(10 + i);

    uint64_t h = 0;
    for (int b = 0; b < 2; ++b) {
        std::vector<int32_t> bt(tokens.begin() + b * 4, tokens.begin() + b * 4 + 4);
        h = m.cache_block(m.block_table(1)[static_cast<size_t>(b)], bt, h);
    }

    // 请求 1 结束：块引用归零，但因为是有价值的缓存块，不能立刻退回空闲链
    m.free_sequence(1);
    CHECK(m.num_free_blocks() == 6);          // 只回收了 2 个未登记块外的？见下
    CHECK(m.num_cached_blocks() == 2);        // 两个缓存块仍在表里

    // 新请求仍能命中
    CHECK(m.acquire_prefix(2, tokens) == 2);
    CHECK(m.seq_len(2) == 8);
}

// ---- 6. 池耗尽时驱逐引用归零的缓存块 ----
static void test_evict_when_exhausted() {
    kv::BlockManager m(4, 4);  // 只有 4 块

    // 占用 2 块并登记为缓存，然后释放（引用归零但仍在缓存）
    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 8; ++i) CHECK(m.append_token(1));
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t h = 0;
    for (int b = 0; b < 2; ++b) {
        std::vector<int32_t> bt(tokens.begin() + b * 4, tokens.begin() + b * 4 + 4);
        h = m.cache_block(m.block_table(1)[static_cast<size_t>(b)], bt, h);
    }
    m.free_sequence(1);
    CHECK(m.num_cached_blocks() == 2);
    CHECK(m.num_free_blocks() == 2);   // 4 - 2 个缓存块

    // 用光剩余 2 块
    CHECK(m.alloc_block(2) != kv::kInvalidBlock);
    CHECK(m.alloc_block(3) != kv::kInvalidBlock);
    CHECK(m.num_free_blocks() == 0);

    // 再要一块：必须驱逐缓存块才拿得到
    CHECK(m.num_cached_blocks() == 2);
    kv::BlockId b = m.alloc_block(4);
    CHECK(b != kv::kInvalidBlock);
    CHECK(m.num_cached_blocks() == 1);   // 驱逐了一个缓存块

    // 被驱逐的块不复用旧内容：再要一块能拿到，缓存清零
    CHECK(m.alloc_block(5) != kv::kInvalidBlock);
    CHECK(m.num_cached_blocks() == 0);

    // 缓存全被驱逐后，原前缀不再命中
    CHECK(m.acquire_prefix(6, tokens) == 0);
}

// ---- 7. 引用中的缓存块不可被驱逐 ----
static void test_no_evict_while_shared() {
    kv::BlockManager m(2, 4);

    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 4; ++i) CHECK(m.append_token(1));
    std::vector<int32_t> tokens = {7, 8, 9, 10};
    CHECK(m.cache_block(m.block_table(1)[0], tokens, 0) != 0);
    CHECK(m.num_cached_blocks() == 1);

    // 另一序列命中它 -> ref=2，仍在缓存表
    CHECK(m.acquire_prefix(2, tokens) == 1);
    CHECK(m.ref_count(m.block_table(2)[0]) == 2);

    // 池里还有 1 块空闲，仍可分配
    CHECK(m.alloc_block(3) != kv::kInvalidBlock);
    CHECK(m.num_free_blocks() == 0);

    // 再要块时：唯一的缓存块 ref=2 不可驱逐 -> 分配失败且无副作用
    CHECK(m.alloc_block(4) == kv::kInvalidBlock);
    CHECK(m.num_cached_blocks() == 1);
    CHECK(m.ref_count(m.block_table(2)[0]) == 2);
}


// 哈希碰撞防御（变异测试逼出来的用例）
//
// 为什么必须注入哈希：真 FNV-1a 64 位撞一次要试约 2^32 组 token，测试里等于没测。
// 用必然碰撞的哈希才能在毫秒内走到那条防御分支——否则代码删掉也不会红。
static uint64_t collide_hasher(uint64_t, const std::vector<int32_t>&) { return 0x42; }

void test_hash_collision_defense() {
    kv::BlockManager m(8, 4);
    m.set_hasher(&collide_hasher);

    const std::vector<int32_t> tokens_a = {1, 2, 3, 4};
    const std::vector<int32_t> tokens_b = {5, 6, 7, 8};

    // 块 0 登记 tokens_a，累积哈希恒为 0x42
    CHECK(m.cache_block(0, tokens_a, 0) == 0x42);
    CHECK(m.num_cached_blocks() == 1);

    // 块 1 内容不同但哈希相同：不能覆盖块 0 的映射
    CHECK(m.cache_block(1, tokens_b, 0) == 0);
    CHECK(m.num_cached_blocks() == 1);

    // 用 tokens_b 去命中：哈希能对上，但内容复验必须拦住 -> 一块都不命中
    kv::BlockManager m2(8, 4);
    m2.set_hasher(&collide_hasher);
    CHECK(m2.cache_block(0, tokens_a, 0) == 0x42);
    CHECK(m2.acquire_prefix(1, tokens_b) == 0);
    CHECK(m2.seq_len(1) == 0);
    CHECK(m2.ref_count(0) == 0);

    // 同内容仍能正常命中（证明拦住的是"假命中"，不是把所有命中都拦了）
    CHECK(m2.acquire_prefix(2, tokens_a) == 1);
    CHECK(m2.seq_len(2) == 4);

    // 恢复默认哈希后不再碰撞
    m2.set_hasher(nullptr);
    CHECK(kv::BlockManager::hash_block(0, tokens_a) !=
          kv::BlockManager::hash_block(0, tokens_b));
}

// 只缓存满块：不足一块的 token 不允许登记，否则 tail 块内容不确定却被当成可复用
void test_cache_block_rejects_partial_block() {
    kv::BlockManager m(8, 4);
    const std::vector<int32_t> partial = {1, 2, 3};
    const std::vector<int32_t> full = {1, 2, 3, 4};

    CHECK(m.cache_block(0, partial, 0) == 0);
    CHECK(m.num_cached_blocks() == 0);

    CHECK(m.cache_block(0, full, 0) != 0);
    CHECK(m.num_cached_blocks() == 1);

    // 空 token 同样不是满块
    CHECK(m.cache_block(1, std::vector<int32_t>{}, 0) == 0);
    CHECK(m.num_cached_blocks() == 1);

    // 非法块号：必须用**没登记过的内容**测，否则碰撞防御会先返回 0，
    // 越界校验根本没被走到（这正是变异 7 第一次没红的原因）
    const std::vector<int32_t> uniq = {9, 9, 9, 9};
    kv::BlockManager m2(8, 4);
    CHECK(m2.cache_block(-1, uniq, 0) == 0);
    CHECK(m2.cache_block(8, uniq, 0) == 0);
    CHECK(m2.cache_block(99, uniq, 0) == 0);
    CHECK(m2.num_cached_blocks() == 0);
}

int main() {
    test_full_prefix_hit();
    test_partial_prefix_hit();
    test_tail_not_cached();
    test_chain_must_match();
    test_cached_survives_free();
    test_evict_when_exhausted();
    test_no_evict_while_shared();
    test_hash_collision_defense();
    test_cache_block_rejects_partial_block();

    if (g_failed == 0) {
        std::printf("ALL PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("%d CHECK(s) FAILED\n", g_failed);
    return 1;
}
