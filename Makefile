# =====================================================================
# Makefile for CUTLASS GEMM 学习示例
#
# 用法：
#   make CUTLASS_DIR=/path/to/cutlass         # 指定 CUTLASS 源码路径
#   make 01_gemm_sm80 && ./01_gemm_sm80
#   make 02_gemm_sm90 && ./02_gemm_sm90       # 需要 H100/H20
#
# 默认从仓库的 bazel 缓存里取（可以直接 make 不带参数）。
# =====================================================================

CUTLASS_DIR ?= /root/hzy/rtp-llm/bazel-rtp-llm/external/cutlass_h_moe
NVCC ?= nvcc

CUTLASS_INC := -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include

# CUTLASS 重模板，所以要把 constexpr 限制放宽，并打开 lambda 扩展
COMMON_FLAGS := -O3 -std=c++17 \
                --expt-relaxed-constexpr --expt-extended-lambda \
                -Xcompiler -Wno-strict-aliasing

.PHONY: all clean run80 run90
all: 01_gemm_sm80 02_gemm_sm90

01_gemm_sm80: 01_gemm_sm80.cu
	$(NVCC) -arch=sm_80 $(COMMON_FLAGS) $(CUTLASS_INC) -o $@ $<

# SM90 需要 sm_90a (a = architecture-specific features，包含 WGMMA/TMA)
02_gemm_sm90: 02_gemm_sm90.cu
	$(NVCC) -arch=sm_90a $(COMMON_FLAGS) $(CUTLASS_INC) -o $@ $<

run80: 01_gemm_sm80
	./01_gemm_sm80

run90: 02_gemm_sm90
	./02_gemm_sm90

clean:
	rm -f 01_gemm_sm80 02_gemm_sm90
