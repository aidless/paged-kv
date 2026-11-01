# CUDA 基准构建
#
# 本机没有 nvcc（NVIDIA 驱动被 nouveau 占用，装驱动需 sudo 密码 + 重启），
# 所以这套 kernel 的实际编译与运行放在 Kaggle 的 T4 上（见 build_kaggle_kernel.py）。
# 这个 Makefile 供任何有 CUDA toolkit 的机器直接用。

NVCC ?= nvcc

# T4 = sm_75。换卡时用 ARCH=sm_80 / sm_86 / sm_90。
ARCH ?= sm_75

# -Xptxas -v 打印寄存器用量与 shared memory 占用：occupancy 出问题时这是第一个线索。
NVCCFLAGS ?= -std=c++14 -O3 -arch=$(ARCH) -Xptxas -v -lineinfo

TARGET := build/bench
SRCS := bench.cu fused_ops.cu
HDRS := fused_ops.h

$(TARGET): $(SRCS) $(HDRS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

# 跑基准。输出里带 ### SUMMARY ### 段，供脚本解析回写文档。
bench: $(TARGET)
	@./$(TARGET)

clean:
	rm -rf build

.PHONY: bench clean
