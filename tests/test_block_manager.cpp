// BlockManager 单元测试
// 覆盖三条边界：块耗尽、双释放、COW 竞态
// 运行：make test（全绿输出 ALL PASS）

#include "kv/block_manager.h"

#include <cstdio>

static int g_failed = 0;

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                              \
        }                                                            \
    } while (0)

// ---- 1. 基本分配与回收 ----
static void test_basic_alloc_free() {
    kv::BlockManager m(/*num_blocks=*/4, /*block_size=*/4);

    // 序列不存在时 alloc 会隐式创建
    CHECK(m.alloc_block(1) == 0);
    CHECK(m.num_free_blocks() == 3);
    CHECK(m.block_table(1).size() == 1);

    // 释放后块全部回收
    m.free_sequence(1);
    CHECK(m.num_free_blocks() == 4);
    CHECK(m.block_table(1).empty());

    // 释放后可重新分配：栈式复用，先拿回刚释放的块 0
    CHECK(m.alloc_block(2) == 0);
    CHECK(m.num_free_blocks() == 3);
}

// ---- 2. token 追加与按需扩块 ----
static void test_append_tokens() {
    kv::BlockManager m(4, 2);  // 每块 2 个 token

    CHECK(m.alloc_block(1) == 0);
    CHECK(m.append_token(1));  // 第 1 个 token，块 0 还有空位
    CHECK(m.append_token(1));  // 第 2 个 token，块 0 满
    CHECK(m.seq_len(1) == 2);
    CHECK(m.block_table(1).size() == 1);  // 还没溢出

    CHECK(m.append_token(1));  // 第 3 个 token -> 需要新块
    CHECK(m.block_table(1).size() == 2);
    CHECK(m.seq_len(1) == 3);
    CHECK(m.num_free_blocks() == 2);
}

// ---- 3. 块耗尽：失败必须无副作用 ----
static void test_exhaustion_no_side_effect() {
    kv::BlockManager m(1, 4);  // 只有 1 块

    CHECK(m.alloc_block(1) == 0);
    CHECK(m.append_token(1));
    CHECK(m.append_token(1));
    CHECK(m.append_token(1));
    CHECK(m.append_token(1));  // 块满，seq_len=4
    CHECK(m.seq_len(1) == 4);

    // 第 5 个 token 需要新块 -> 失败，且状态完全不变
    CHECK(!m.append_token(1));
    CHECK(m.seq_len(1) == 4);
    CHECK(m.block_table(1).size() == 1);
    CHECK(m.num_free_blocks() == 0);

    // 释放 1 号序列后，2 号可以分配
    m.free_sequence(1);
    CHECK(m.alloc_block(2) != kv::kInvalidBlock);
}

// ---- 4. fork 共享 + COW ----
// 注意：COW 只在写入「未满的共享块」时触发——块已满时 append
// 走正常扩块路径，与共享无关。
static void test_fork_and_cow() {
    kv::BlockManager m(4, 4);  // 每块 4 个 token

    CHECK(m.alloc_block(1) == 0);
    CHECK(m.append_token(1));  // 块 0 只装了 1/4，还有空位

    // fork：子序列共享块 0
    CHECK(m.fork_sequence(1, 2));
    CHECK(m.ref_count(0) == 2);
    CHECK(m.block_table(2) == m.block_table(1));
    CHECK(m.seq_len(2) == 1);

    // 三方共享：再 fork 出子序列 3
    CHECK(m.fork_sequence(1, 3));
    CHECK(m.ref_count(0) == 3);

    // 三个序列先后写入未满的共享块 -> 逐个 COW 搬走
    CHECK(m.append_token(1));
    CHECK(m.cow_count() == 1);
    CHECK(m.ref_count(0) == 2);
    CHECK(m.block_table(1).back() != 0); // 父序列换到了新块

    CHECK(m.append_token(2));
    CHECK(m.cow_count() == 2);
    CHECK(m.ref_count(0) == 1);
    CHECK(m.block_table(2).back() != 0);

    // 序列 3 写入时块 0 已只剩它一个持有者 -> 普通写入，不再 COW
    // （COW 次数上限 = 共享者数 - 1）
    CHECK(m.append_token(3));
    CHECK(m.cow_count() == 2);
    CHECK(m.ref_count(0) == 1);
    CHECK(m.block_table(3).back() == 0); // 序列 3 直接持有块 0
    CHECK(m.num_free_blocks() == 1);     // 4 块 - 块0 - 2 个 COW 新块
}

// ---- 5. COW 时内存不足：失败且状态不变 ----
static void test_cow_exhaustion() {
    kv::BlockManager m(1, 4);  // 只有 1 块，COW 没地方放新块

    CHECK(m.alloc_block(1) == 0);
    CHECK(m.append_token(1));
    CHECK(m.fork_sequence(1, 2));
    CHECK(m.ref_count(0) == 2);

    // 写入共享块需要 COW -> 无空闲块 -> 失败
    CHECK(!m.append_token(1));
    CHECK(m.cow_count() == 0);
    CHECK(m.seq_len(1) == 1);
    CHECK(m.ref_count(0) == 2);            // 引用关系未破坏
    CHECK(m.block_table(1).back() == 0);   // table 未变
}

// ---- 6. 双释放与重复 fork 防御 ----
static void test_double_free_and_fork() {
    kv::BlockManager m(4, 2);

    CHECK(m.alloc_block(1) == 0);
    CHECK(m.alloc_block(1) == 1);
    m.free_sequence(1);
    CHECK(m.num_free_blocks() == 4);

    // 双释放：no-op，不产生重复空闲块
    m.free_sequence(1);
    CHECK(m.num_free_blocks() == 4);

    // 重复 fork 同一子序列：拒绝
    CHECK(m.fork_sequence(1, 2) == false);  // 父序列已被释放
    CHECK(m.alloc_block(1) == 1);           // 重新建 1 号
    CHECK(m.fork_sequence(1, 2));
    CHECK(m.fork_sequence(1, 2) == false);  // 子序列已存在
    CHECK(m.ref_count(1) == 2);
}

// ---- 7. 非法输入 ----
static void test_invalid_inputs() {
    kv::BlockManager m(2, 4);

    // 不存在的序列不能直接 append
    CHECK(!m.append_token(99));
    CHECK(m.seq_len(99) == 0);
    CHECK(m.block_table(99).empty());

    // fork 不存在的父序列
    CHECK(!m.fork_sequence(42, 43));

    // ref_count 越界安全
    CHECK(m.ref_count(kv::kInvalidBlock) == 0);
    CHECK(m.ref_count(999) == 0);
}

int main() {
    test_basic_alloc_free();
    test_append_tokens();
    test_exhaustion_no_side_effect();
    test_fork_and_cow();
    test_cow_exhaustion();
    test_double_free_and_fork();
    test_invalid_inputs();

    if (g_failed == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d CHECK(s) FAILED\n", g_failed);
    return 1;
}
