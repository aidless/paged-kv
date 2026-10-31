CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2 -pthread
INCLUDES := -Iinclude

test: build/test_block_manager
	./build/test_block_manager

build/test_block_manager: tests/test_block_manager.cpp src/block_manager.cpp include/kv/block_manager.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_block_manager.cpp src/block_manager.cpp -o $@

clean:
	rm -rf build

.PHONY: test clean
