CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2 -pthread
INCLUDES := -Iinclude

SRC := src/block_manager.cpp src/kv_cache.cpp
HDR := include/kv/block_manager.h include/kv/kv_cache.h

# 默认：跑全部测试。任一测试失败即非零退出，不遮蔽前一个结果。
test: build/test_block_manager build/test_kv_cache build/test_prefix_cache
	@echo "=== block_manager ==="
	@./build/test_block_manager
	@echo "=== kv_cache (slot mapping + decode path) ==="
	@./build/test_kv_cache
	@echo "=== prefix_cache (chain hash + eviction) ==="
	@./build/test_prefix_cache

build/test_block_manager: tests/test_block_manager.cpp $(SRC) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_block_manager.cpp $(SRC) -o $@

build/test_kv_cache: tests/test_kv_cache.cpp $(SRC) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_kv_cache.cpp $(SRC) -o $@

build/test_prefix_cache: tests/test_prefix_cache.cpp $(SRC) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_prefix_cache.cpp $(SRC) -o $@

clean:
	rm -rf build

.PHONY: test clean
