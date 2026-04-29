# CUTLASS / cute 学习路径

## 目录结构

```
cutlass_learn/
  Makefile
  01_gemm_sm80.cu       ← CUTLASS 2.x device::Gemm（已完成）
  02_gemm_sm90.cu       ← CUTLASS 3.x CollectiveBuilder（已完成）
  03_cute_layout.cu     ← cute 核心：Layout = (Shape, Stride)
  04_cute_tensor_tile.cu← Tensor + local_tile + partition
  05_cute_tiled_copy_mma.cu ← TiledCopy + TiledMMA（连接 SMEM 和硬件指令）
  SageAttention/        ← 目标代码库
```

---

## 阶段 1：cute 实操

**目标**：能读懂 SageAttention3 里所有的 cute 表达式，不卡壳。

具体的"卡壳点"清单：

| 代码片段（来自 SageAttention） | 需要懂的概念 | 在哪学 |
|---|---|---|
| `Layout<Shape<_16,_4>, Stride<_16,_4>>` | Shape/Stride 嵌套、编译期常量 | 03 |
| `make_stride(make_stride(_0{}, _1{}), _0{}, _4{})` | zero-stride (broadcast)、嵌套 | 03 |
| `make_tensor(ptr, layout)(i, j)` | Tensor 索引 = inner_product(coord, stride) | 04 |
| `local_tile(mQ, TileShape{}, coord)` | 从大 tensor 切出 tile | 04 |
| `local_partition(tile, thr_layout, thr_id)` | 分配给一个 thread | 04 |
| `group_modes<0, 3>(tensor)` | 合并多个 mode | 04 |
| `partition_S(gQ)` / `partition_D(sQ)` | TiledCopy 的 src/dst 分片 | 05 |
| `partition_fragment_A(sQ)` | TiledMMA 从 SMEM 切 register fragment | 05 |
| `make_zip_tensor(tSrQ, tSrSFQ)` | 配对 (data, scale) 给 BlockScaled MMA | 05 |

---

## 编译说明

```bash
cd cutlass_learn
CUTLASS_DIR=/root/hzy/rtp-llm/bazel-rtp-llm/external/cutlass_h_moe

# 03, 04 不需要 GPU（host-only print）
nvcc -arch=sm_80 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
     -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/tools/util/include \
     03_cute_layout.cu -o 03_cute_layout
./03_cute_layout

# 05 需要 GPU
nvcc -arch=sm_90a ... 05_cute_tiled_copy_mma.cu
```

---

## 阶段 1 三个概念层

```
Layout (Shape, Stride)          ← 03_cute_layout.cu
    │
    ▼
Tensor = ptr + Layout           ← 04_cute_tensor_tile.cu
local_tile / local_partition
    │
    ▼
TiledCopy / TiledMMA            ← 05_cute_tiled_copy_mma.cu
partition_S / partition_D
partition_fragment_A/B/C
make_zip_tensor
```

每一层都建立在上一层之上。SageAttention 的 mainloop_tma_ws.h 是三层全部叠加的产物。
