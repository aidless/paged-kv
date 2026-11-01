// KV 存储层 + slot mapping 测试
//
// 这两个东西合起来才能让块管理器"真的服务一个 decode step"：
//   1. compute_slot_mapping：把逻辑 token 位置翻译成物理槽位编号
//      （对应 vLLM 的 slot_mapping，decode 时每步只有 1 个新 token，
//       但 prefill 一次要写整段 prompt，所以要支持任意 num_tokens）
//   2. KVCache：真正存 KV 数据的扁平存储，按槽位编号寻址
//
// 槽位编号约定：slot = block_id * block_size + offset_in_block
// 这样 block table 的一次查表就够定位，不需要二维下标。

#include "kv/block_manager.h"
#include "kv/kv_cache.h"

#include <cmath>
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

// ---- 1. slot mapping 基本正确性 ----
static void test_slot_mapping_basic() {
    kv::BlockManager m(8, 4);
    CHECK(m.alloc_block(1) == 0);
    CHECK(m.alloc_block(1) == 1);

    std::vector<int64_t> slots;

    // 从 pos 0 写 4 个 token：全落在块 0（物理槽位 0..3）
    CHECK(m.compute_slot_mapping(1, 0, 4, &slots));
    CHECK(slots.size() == 4);
    CHECK(slots[0] == 0 && slots[1] == 1 && slots[2] == 2 && slots[3] == 3);

    // 从 pos 2 写 4 个 token：跨块 0 -> 块 1（槽位 2,3,4,5）
    CHECK(m.compute_slot_mapping(1, 2, 4, &slots));
    CHECK(slots.size() == 4);
    CHECK(slots[0] == 2 && slots[1] == 3 && slots[2] == 4 && slots[3] == 5);
}

// ---- 2. 块表不够覆盖时必须失败且不留副作用 ----
static void test_slot_mapping_out_of_range() {
    kv::BlockManager m(8, 4);
    CHECK(m.alloc_block(1) == 0);  // 只有 1 个块，覆盖 pos 0..3

    std::vector<int64_t> slots = {777};  // 哨兵：失败时不许被改动

    // 需要 pos 0..4，块表只到 3
    CHECK(!m.compute_slot_mapping(1, 0, 5, &slots));
    CHECK(slots.size() == 1 && slots[0] == 777);

    // 起始位置越界
    CHECK(!m.compute_slot_mapping(1, 4, 1, &slots));
    CHECK(slots.size() == 1 && slots[0] == 777);

    // 序列不存在
    CHECK(!m.compute_slot_mapping(99, 0, 1, &slots));
    CHECK(slots.size() == 1 && slots[0] == 777);

    // 零 token 是合法请求，结果应为空
    CHECK(m.compute_slot_mapping(1, 0, 0, &slots));
    CHECK(slots.empty());
}

// ---- 3. KVCache 写入 / 读回 ----
static void test_kv_cache_write_read() {
    kv::KVCache cache(8, 4, 3);  // 8 块 * 4 槽 * 3 维
    CHECK(cache.num_slots() == 32);
    CHECK(cache.head_dim() == 3);

    float a[3] = {1.0f, 2.0f, 3.0f};
    CHECK(cache.write(5, a));
    CHECK(std::fabs(cache.read(5)[0] - 1.0f) < 1e-6);
    CHECK(std::fabs(cache.read(5)[1] - 2.0f) < 1e-6);
    CHECK(std::fabs(cache.read(5)[2] - 3.0f) < 1e-6);

    // 越界槽位必须拒绝（坏写会静默踩坏别的序列）
    CHECK(!cache.write(32, a));
    CHECK(!cache.write(-1, a));
}

// ---- 4. 批量写入的失败原子性 ----
// 只要有任一槽位非法，就必须一个字节都不写——
// 否则批量写会把数据踩进相邻序列的 KV 里，且事后无法恢复。
static void test_write_tokens_atomic() {
    kv::KVCache cache(2, 4, 2);  // 8 个槽位

    // 合法槽位 0 写已知值，作为"未被踩坏"的基线
    float base[2] = {9.0f, 9.0f};
    CHECK(cache.write(0, base));

    // 批量里混入一个越界槽位 8（合法范围 0..7）
    std::vector<int64_t> bad = {1, 2, 8};
    std::vector<float> src(6, 5.0f);
    CHECK(cache.write_tokens(bad, src.data(), 6) == -1);

    // 槽位 1 必须没被写入（失败不留半成品）
    CHECK(std::fabs(cache.read(1)[0]) < 1e-6);

    // 负槽位同理
    std::vector<int64_t> neg = {-1};
    CHECK(cache.write_tokens(neg, src.data(), 2) == -1);

    // 维度不匹配也必须拒绝（防止按错半径踩坏相邻槽位）
    std::vector<int64_t> ok = {1};
    CHECK(cache.write_tokens(ok, src.data(), 3) == -1);
    CHECK(std::fabs(cache.read(1)[0]) < 1e-6);
}

// ---- 5. 批量写 + 按 block table 拼回连续序列 ----
static void test_kv_cache_gather() {
    kv::BlockManager m(8, 4);
    CHECK(m.alloc_block(1) == 0);
    CHECK(m.alloc_block(1) == 1);

    kv::KVCache cache(8, 4, 2);

    // 一次性写 5 个 token（跨块 0 -> 块 1）
    std::vector<int64_t> slots;
    CHECK(m.compute_slot_mapping(1, 0, 5, &slots));
    std::vector<float> src(5 * 2);
    for (int i = 0; i < 5; ++i) {
        src[i * 2 + 0] = static_cast<float>(i);
        src[i * 2 + 1] = static_cast<float>(i) * 10.0f;
    }
    CHECK(cache.write_tokens(slots, src.data(), static_cast<int>(src.size())) == 5);

    // 拼回前 5 个 token，必须逐字节等于写入顺序
    std::vector<float> dst(5 * 2, -1.0f);
    CHECK(cache.gather(m.block_table(1), 5, dst.data()) == 5);
    for (int i = 0; i < 5 * 2; ++i) CHECK(std::fabs(dst[i] - src[i]) < 1e-6);

    // 只取前 3 个（块 0 内），不碰块 1
    std::vector<float> head(3 * 2, -1.0f);
    CHECK(cache.gather(m.block_table(1), 3, head.data()) == 3);
    for (int i = 0; i < 3 * 2; ++i) CHECK(std::fabs(head[i] - src[i]) < 1e-6);

    // 块表不够覆盖 seq_len 时失败
    std::vector<kv::BlockId> short_table = {0};
    std::vector<float> dummy(5 * 2);
    CHECK(cache.gather(short_table, 5, dummy.data()) == -1);
}

// ---- 5. decode step：管理器簿记 -> 槽位 -> 写入 -> 读回 ----
// 这是"块管理器真的服务一次 decode"的最小闭环
static void test_decode_step_end_to_end() {
    kv::BlockManager m(4, 4);
    kv::KVCache cache(4, 4, 1);

    // prefill：prompt 长度 3
    CHECK(m.alloc_block(1) == 0);
    for (int i = 0; i < 3; ++i) CHECK(m.append_token(1));
    CHECK(m.seq_len(1) == 3);

    std::vector<int64_t> slots;
    CHECK(m.compute_slot_mapping(1, 0, 3, &slots));
    float prompt[3] = {10.0f, 11.0f, 12.0f};
    CHECK(cache.write_tokens(slots, prompt, 3) == 3);

    // decode 一步：追加第 4 个 token（正好填满块 0，不触发扩块）
    CHECK(m.append_token(1));
    CHECK(m.seq_len(1) == 4);
    CHECK(m.block_table(1).size() == 1);

    CHECK(m.compute_slot_mapping(1, 3, 1, &slots));
    CHECK(slots.size() == 1 && slots[0] == 3);
    float t3 = 13.0f;
    CHECK(cache.write_tokens(slots, &t3, 1) == 1);

    // 读出完整 4 个 token
    std::vector<float> all(4, -1.0f);
    CHECK(cache.gather(m.block_table(1), 4, all.data()) == 4);
    CHECK(std::fabs(all[0] - 10.0f) < 1e-6);
    CHECK(std::fabs(all[3] - 13.0f) < 1e-6);

    // 再 decode 一步 -> 触发扩第 2 个块
    CHECK(m.append_token(1));
    CHECK(m.block_table(1).size() == 2);
    CHECK(m.compute_slot_mapping(1, 4, 1, &slots));
    CHECK(slots.size() == 1 && slots[0] == 4);  // 块 1 的第 0 槽
}

int main() {
    test_slot_mapping_basic();
    test_slot_mapping_out_of_range();
    test_kv_cache_write_read();
    test_write_tokens_atomic();
    test_kv_cache_gather();
    test_decode_step_end_to_end();

    if (g_failed == 0) {
        std::printf("ALL PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("%d CHECK(s) FAILED\n", g_failed);
    return 1;
}
