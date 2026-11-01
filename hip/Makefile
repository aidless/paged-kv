# HIP/ROCm 基准构建
#
# 在 AMD Radeon Cloud 中国站实例上直接用（hipcc 已在 /usr/bin/hipcc）。
# gfx1100 = RDNA3。换 GPU 用 ARCH=gfx942（MI300X）/ gfx90a（MI200）。

HIPCC ?= hipcc
ARCH ?= gfx1100

# -Xptxas 是 nvcc 的；HIP 用 --offload-arch。
HIPCCFLAGS ?= -std=c++14 -O3 --offload-arch=$(ARCH)

TARGET := build/bench
SRCS := bench.cpp fused_ops.hip.cpp
HDRS := fused_ops.h

$(TARGET): $(SRCS) $(HDRS)
	@mkdir -p build
	$(HIPCC) $(HIPCCFLAGS) -o $@ $(SRCS)

# 跑基准。输出里带 ### SUMMARY ### 段，供脚本解析回写文档。
bench: $(TARGET)
	@./$(TARGET)

clean:
	rm -rf build

.PHONY: bench clean
