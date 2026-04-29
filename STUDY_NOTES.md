# CUTLASS / cute / SageAttention 学习笔记

> 整理自实际学习过程，按"从基础到目标"排列。
> 目标：吃透 FA3 + SageAttention3 的实现。

---

## 目录

1. [学习路径](#学习路径)
2. [CUTLASS 2.x SM80：抽象金字塔](#cutlass-2x-sm80)
3. [CUTLASS 3.x SM90：CollectiveBuilder 范式](#cutlass-3x-sm90)
4. [cute：Layout 代数](#cute)
5. [SageAttention3：FA3 + MX-FP4](#sageattention3)
6. [关键文件索引](#关键文件索引)
7. [自测题](#自测题)
8. [重要建议](#重要建议)

---

## 学习路径

```
阶段 0  论文（1-2 天）
  FA1 §3 → FA2 §3 → FA3 全文 → SageAttn 1/2 §3-4 → SageAttn3 全文

阶段 1  cute 实操（3-5 天）  ← 已完成三个示例文件
  03_cute_layout.cu          Layout = (Shape, Stride)
  04_cute_tensor_tile.cu     Tensor + local_tile + partition
  05_cute_tiled_copy_mma.cu  TiledCopy + TiledMMA + ZipTensor

阶段 2  FA3 Hopper 通读（4-7 天）
  flash-attention/hopper/ 目录，顺序：
  launch → kernel_ws → kernel_traits → softmax → mainloop → epilogue

阶段 3  SageAttention3（5-7 天）
  只看与 FA3 的 6 个 diff，其余结构相同
```

**总时间估算（懂 FA 算法的起点）**：12-19 天读懂，加自己写 mini 版则月级。

---

## CUTLASS 2.x SM80

### 抽象金字塔

```
Device::Gemm      一次 host 调用，整个 grid
  └── Kernel      一个 __global__ 函数，一个 grid
        └── Threadblock   一个 CTA，处理 ThreadblockShape 的 tile
              └── Warp    一个 warp，处理 WarpShape 的 tile
                    └── MMA Instruction   一条 mma.sync，处理 InstructionShape
```

### 三个 GemmShape 的物理含义

```cpp
ThreadblockShape <128, 128, 32>   // CTA 算 128×128 输出，每次 K 步进 32
WarpShape        < 64,  64, 32>   // 一个 warp 算 64×64，推导：(128/64)² = 4 warps/CTA
InstructionShape < 16,   8, 16>   // 一条 mma.sync.m16n8k16 的硬件形状，不可变
```

`InstructionShape` 由硬件决定，不能随意选。`ThreadblockShape` 和 `WarpShape` 要满足：
- `ThreadblockShape` 能被 `WarpShape` 整除
- `WarpShape` 能被 `InstructionShape` 整除
- `kNWarps = (TbM/WarpM) × (TbN/WarpN)` 通常等于 4 或 8

### Stages = 软件流水深度

`Stages=3` 表示 SMEM 里同时存 3 份 K-tile 的 A/B：
- stage k 在 `mma.sync` 时，stage k+1 在 `cp.async` 加载
- 把全局内存延迟（数百 cycles）藏在算力后面
- `Stages` 越大延迟藏得越深，但 SMEM 占用越多，occupancy 越低
- A100 典型最优：`Stages=3` 或 `Stages=4`

### Swizzle 解决 Bank Conflict

- A100 SMEM：32 个 bank，每个 bank 4 字节，一个 warp 同时访问同 bank 不同地址 → 串行化
- fp16 naive 存法：同一列的元素全在同一 bank → `ldmatrix` 触发 conflict
- Swizzle（XOR 模式）：`bank_idx ^= (row_idx & 0x7) << k`，散列后同列元素分散在不同 bank
- SM80 里 CUTLASS 根据 (dtype, layout) 自动选 `TensorOpMultiplicand` tag，不需要手写

### EpilogueOp

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, /*VectorWidth=*/8, ElementAccumulator, ElementCompute>;
// D = alpha * acc + beta * C
// VectorWidth=8：一次写回 8 个 fp16 = 16 bytes = STG.E.128
```

### B 为什么是 col-major（TN GEMM）

A row-major (M×K) + B col-major (K×N)：两者都在 K 方向连续，SMEM load 和 `ldmatrix` 最优。给 B 传 row-major 时 CUTLASS 会加隐式转置，性能损失。业界标准称为 "TN GEMM"（T=A transposed-implicit, N=B non-transposed）。

### 调用三步

```cpp
gemm.can_implement(args);   // 运行期参数合法性检查
gemm.initialize(args, ws);  // 把 args 写到 device-resident params
gemm();                     // launch grid
```

---

## CUTLASS 3.x SM90

### 范式对比

| | SM80 (2.x) | SM90 (3.x) |
|---|---|---|
| 描述方式 | 4 个 GemmShape | TileShape + ClusterShape + KernelSchedule |
| 数据搬运 | `cp.async` per-thread | TMA（一条指令搬一个 tile） |
| MMA 指令 | `mma.sync.m16n8k16`（warp 级，同步） | `wgmma.mma_async.m64nXk16`（warp-group 级，异步） |
| 软件流水 | 所有 warp 一起 load+compute | Producer/Consumer warp 分工，mbarrier 同步 |
| SMEM 复用 | 手写 union | `StageCountAutoCarveout` 自动算，`union` 自动生成 |
| 跨 SM | 无 | Cluster + DSMEM |
| 几何抽象 | 隐式（在模板里） | cute Layout（编译期代数）|

### CollectiveBuilder 声明顺序（epilogue 必须先写）

```cpp
// 1. 先构造 CollectiveEpilogue（因为 mainloop 需要知道 epilogue 的 SMEM 大小）
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    Sm90, OpClassTensorOp, TileShape, ClusterShape, EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    EpilogueSchedule>::CollectiveOp;

// 2. 再构造 CollectiveMainloop，传入 epilogue 的 SharedStorage 大小
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    Sm90, OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc, TileShape, ClusterShape,
    StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>,
    KernelSchedule>::CollectiveOp;
```

### Cluster

- H100 引入：多个 CTA 组成 cluster，保证同时驻留在相邻 SM
- 好处：可访问 DSMEM（跨 SM 共享 SMEM）、集体 TMA（多 CTA 共享一次 TMA load）
- `ClusterShape<1,1,1>` = 不用 cluster，最保守最稳定
- `ClusterShape<2,1,1>` = N 方向相邻两个 CTA 共享 A 的 TMA，节省一半 A 带宽

### Warp Specialization（KernelTmaWarpSpecializedCooperative）

```
一个 CTA（kBlockM=128 时 12 warps = 3 warp-groups）：

WarpGroup 0 = Producer（4 warps）
  warp 0: Mainloop — 持续 TMA load Q/K/V
  warp 1: Epilogue — TMA store O（等 mma 完成后）
  warp 2/3: 让出寄存器（warpgroup_reg_dealloc<24>）

WarpGroup 1 = Consumer 0（4 warps）  ← 算 MMA + softmax
WarpGroup 2 = Consumer 1（4 warps）  ← 协作算同一 tile（Cooperative 模式）
```

寄存器策略：
- `warpgroup_reg_dealloc<24>()` — producer 让出寄存器给 consumer
- `warpgroup_reg_alloc<232>()` — consumer 申请大量寄存器存累加器

### TMA vs cp.async

| | cp.async (SM80) | TMA (SM90) |
|---|---|---|
| 粒度 | 每个 thread 搬 16 bytes | 一条指令搬整个 tile |
| 边界处理 | 需要 predicate | 硬件自动处理 |
| Swizzle | 软件决定 SMEM 地址 | TMA descriptor 里指定 swizzle mode |
| 同步 | `cp.async.wait_group` | mbarrier arrive/wait |
| 谁发指令 | 所有参与 warp | 只有 producer warp |

### WGMMA vs mma.sync

| | mma.sync (SM80) | wgmma (SM90) |
|---|---|---|
| 粒度 | warp（32 thread）| warp-group（4 warps = 128 thread）|
| 同步性 | 同步 | 异步（需 commit_group + wait_group）|
| A 操作数 | 必须在 register | 可直接从 SMEM 读 |
| 最小 M | 16 | 64（这是 SM90 处理小 M 的痛点）|
| 吞吐 | 基准 | 约 2× |

### Swizzle 在 SM90 的位置

SM90 swizzle 出现在**两处**，必须保持一致：
1. **cute SmemLayout**：`Swizzle<B,M,S> ∘ Layout<...>` 描述 wgmma 读 SMEM 时的地址
2. **TMA descriptor**：`swizzleMode = SWIZZLE_128B` 等，描述 TMA 写 SMEM 时的地址

两者由 `CollectiveBuilder` 自动配齐，手写时必须手动保持一致。

### Epilogue：什么时候要手写 SmemLayout

| 场景 | 是否需要手写 SmemLayout |
|---|---|
| 默认 `LinearCombination`（alpha/beta） | 否，CollectiveBuilder 自动 |
| Bias broadcast（标准 per-row/per-col） | 否，用 EVT 内置节点 `Sm90RowBroadcast` |
| 非标准 broadcast（如 per-group scale）| 是，写自定义 visitor 节点 |
| 多输出 tensor（如 O + LSE） | 是 |
| 非平凡 LayoutD | 是 |
| MoE finalize fused epilogue | 是，完全自定义 |

### StageCountAutoCarveout + SharedStorage Union

```cpp
// SMEM 里 mainloop 和 epilogue 共用同一段（分时不同时用）
union {
    Mainloop::SharedStorage mainloop;   // 多 stage Q/K/V buffer
    Epilogue::SharedStorage epilogue;   // 累加器 staging
} smem;
// 总占用 = max(mainloop_size, epilogue_size)，不是相加

// StageCountAutoCarveout 的含义：
// "我（mainloop）在算 pipeline stage 数时，先给 epilogue 留 X bytes，
// 剩下的全用来装 pipeline，能塞几个 stage 就几个"
```

---

## cute

### 核心：Layout = (Shape, Stride)

```
offset(i, j, ...) = inner_product(coord, stride)
                  = i * stride[0] + j * stride[1] + ...
```

Shape 和 Stride 均可嵌套，嵌套时递归展开：
```cpp
// row-major (4,8)
make_layout(make_shape(_4{}, _8{}), make_stride(_8{}, _1{}));
// col-major (4,8)
make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _4{}));
// 嵌套（描述 warp 内 thread 的 tile 结构）
make_layout(make_shape(make_shape(_2{},_4{}), make_shape(_4{},_2{})),
            make_stride(make_stride(_1{},_8{}), make_stride(_2{},_32{})));
```

### 编译期常量 vs 运行期值

```cpp
_1{}, _8{}, _128{}   = cute::Int<1>{}, Int<8>{}, Int<128>{}  // 编译期
int(4), n            = 运行期
Int<kBlockN/64>{}    = 模板实例化后确定，编译期（如果 kBlockN 是 constexpr）
```

全编译期的 Layout → offset 计算在编译期折叠 → 零运行期开销。

### zero-stride = broadcast

```cpp
// 8 个逻辑元素全映射到同一物理地址
make_layout(make_shape(_8{}), make_stride(_0{}));

// SageAttention LayoutSFP：16 个 thread 共享一个 scale
make_layout(make_shape(make_shape(_16{}, _4{}), Int<kBlockN/64>{}),
            make_stride(make_stride(_0{},  _1{}), _4{}));
// stride[0][0] = _0 => 这 16 个 thread 的 scale 映射到同一物理 slot
```

### 重要 API

| API | 含义 | SageAttention 用例 |
|---|---|---|
| `tile_to_shape(atom, shape)` | 把 atom layout 平铺到 shape | SmemLayoutK = tile_to_shape(SmemLayoutAtomK, (N, K, Stages)) |
| `coalesce(layout)` | 化简，合并连续维度 | 内部用，不直接写 |
| `filter_zeros(layout)` | 去掉 stride=0 的维度 | 算 SF tensor 的 stage stride |
| `local_tile(tensor, tile, coord)` | 取第 coord 个 tile | gQ = local_tile(mQ, TileShape, (m_block, _0)) |
| `local_partition(tensor, thr_layout, id)` | 把 tile 分给 thread id | partition_S/D 内部用 |
| `group_modes<B,E>(tensor)` | 合并 mode B~E-1 | group_modes<0,3>(partition_S(gK)) |
| `recast<T>(tensor)` | reinterpret_cast 元素类型 | recast<float4>(acc) 向量化写 |
| `make_zip_tensor(t1, t2)` | 配对 (data, scale) | make_zip_tensor(tSrQ, tSrSFQ) |
| `as_position_independent_swizzle_tensor(t)` | 调整 swizzle tensor 基址 | partition_S(as_position_independent...(sQ)) |

### TiledCopy：分布式 SMEM Copy

```cpp
// 创建
auto tiled_copy = make_tiled_copy(CopyAtom{}, thread_layout, value_layout);
auto thr_copy   = tiled_copy.get_thread_slice(thread_idx);

// 分片
Tensor src_frag = thr_copy.partition_S(src_tensor);  // S = Source（从哪读）
Tensor dst_frag = thr_copy.partition_D(dst_tensor);  // D = Destination（写到哪）

// 执行
copy(tiled_copy, src_frag, dst_frag);
```

`partition_S` 返回的 Tensor 第 0 个 mode = CopyAtom 形状（一次搬运的元素），后续 mode = 循环次数。

### TiledMMA：分布式 MMA

```cpp
TiledMmaQK tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

// 创建 fragment（寄存器）
Tensor tSrA = thr_mma.partition_fragment_A(smem_A);  // A 的 register fragment
Tensor tSrB = thr_mma.partition_fragment_B(smem_B);  // B 的 register fragment
Tensor tSrC = partition_fragment_C(tiled_mma, tile_MN);  // C 累加器

// 执行 MMA（编译期展开成硬件指令）
cute::gemm(tiled_mma, tSrA, tSrB, tSrC);
```

`partition_fragment_A` ≠ `partition_S`：
- `partition_S` 给 **TiledCopy** 用，描述 copy 的 source 分片
- `partition_fragment_A` 给 **TiledMMA** 用，描述 MMA 的 A register 分片

### BlockScaled MMA（SageAttention 独有）

```cpp
// 普通 MMA
cute::gemm(tiled_mma, tSrA, tSrB, tSrC);

// BlockScaled MMA（SM120，每 16 个 fp4 对应 1 个 ue4m3 scale）
cute::gemm(tiled_mma,
           make_zip_tensor(tSrA_data, tSrA_scale),   // data + scale 配对
           make_zip_tensor(tSrB_data, tSrB_scale),
           tSrC);
// cute 自动展开成 SM120_16x32x64_TN_VS_NVFP4 指令
```

### Swizzle<B, M, S> 参数

- `B`（Base）：XOR 翻转的 bit 数，= log2(swizzle pattern 数)
- `M`（MBase）：从地址第几个 bit 开始取"行号"
- `S`（Shift）：行号 shift 多少位再 XOR 进列号

`Swizzle<3,3,3>` = fp16 @ SMEM 最常用配置，适用 8×8 的 swizzle 单元。

---

## SageAttention3

### 定位

**SageAttention3 = FA3（FlashAttention-3）+ MX-FP4 量化 + SM120（RTX 5090）**

FA3 是"基底"，SageAttention3 的代码结构、文件命名、warp 分工方式几乎与 FA3 完全一致。两者 diff 集中在 6 处，见下文。

### 数据格式

| Tensor | dtype | 含义 |
|---|---|---|
| Q, K, V（输入） | `uint8`（packed fp4） | 每字节 2 个 e2m1 fp4 值，host 端预量化 |
| sfq, sfk, sfv | `fp8_e4m3fn`（ue4m3） | 每 16 个 fp4 对应 1 个 scale（MX-FP4 格式）|
| delta_s | `float32` | per-tile 量化误差修正项，host 端预算 |
| out（输出） | `bf16` / `fp16` | 标准浮点输出 |

MX-FP4（Microscaling FP4）：每 16 个元素共享 1 个 block scale，硬件（SM120 BlockScaled MMA）原生支持。

### 与 FA3 的 6 个 diff

| # | 内容 | 文件 + 行号 |
|---|---|---|
| 1 | dtype 从 bf16 换成 fp4+scale | `api.cu:226`, `kernel_traits.h:75-94` |
| 2 | MMA atom 换成 BlockScaled | `kernel_traits.h:113` `SM120_16x32x64_TN_VS_NVFP4` |
| 3 | 多了 SF tensor 的 SMEM/TMA path | `blockscaled_layout.h`, `mainloop_tma_ws.h:105-125` |
| 4 | **delta_s 修正项**（核心创新）| `mainloop_tma_ws.h:691-704`, `api.cu:50` |
| 5 | online softmax 融合 fp4 量化 | `softmax_fused.h` 全文 |
| 6 | P 量化 + SFP shfl 重排 | `mainloop_tma_ws.h:750-797` |

### delta_s：SageAttention 的核心 trick

```
fp4 量化 QK^T 有误差：
  QK^T_fp4 = QK^T_truth + ε（量化误差）

SageAttention 在 host 端预算 delta_s：
  delta_s ≈ E[QK^T_truth] - E[QK^T_fp4]（per-tile fp32 矩阵）

Kernel 里每次 QK MMA 前：
  acc = delta_s          ← add_delta_s lambda 把 fp32 修正项写进累加器
  acc += QK_fp4_gemm     ← BlockScaled MMA 继续累加

效果：fp4 计算速度 + fp32 级别修正精度，几乎零额外运行时成本
（delta_s 只是一次 TMA load + fp32 向量加法）
```

代码位置：`mainloop_tma_ws.h:691-704`（`add_delta_s` lambda），`mainloop_tma_ws.h:718`（调用处）。

### Online Softmax + 量化融合

`softmax_fused.h` 的 `online_softmax_with_quant` 同时做三件事：
1. reduce 出 `row_max`（跨 thread `shfl_xor_sync`）
2. 算 `exp2(x * scale - max_scaled)` 归一化
3. 算 `AbsMaxP`（softmax 后 P 的 absmax，用于下一步 fp4 量化的 scale）

关键常量：
```cpp
fp8_scalexfp4_scale_log2 = log2(1 / (448 * 6))
// = log2(1 / ue4m3_max / fp4_e2m1_max)
// 把 P 的 dynamic range 直接映射到 fp4 的表达范围，
// 一行常量加法把量化 rescale 融进 softmax 的 max scaling
```

### Warp 角色分工

```
kBlockM=128 → kNWarps=12 = 3 warp-groups

WarpGroup 0 = Producer（4 warps）
  warp 0 (Mainloop): TMA load Q + SF_Q（1 stage） + K + SF_K + delta_s（3 stages） + V + SF_V（3 stages）
  warp 1 (Epilogue): wait barrier_o → TMA store O → arrive barrier_o

WarpGroup 1 = Consumer 0（4 warps）
WarpGroup 2 = Consumer 1（4 warps）
  两者 cooperative 协作算同一个 (M, N) 输出 tile 的 mma + softmax
```

`kernel_ws.h:140` `warpgroup_reg_dealloc<24>` / `warpgroup_reg_alloc<232>` — 把寄存器从 producer 让给 consumer。

### Pipeline 结构

```
Q:  PipelineTmaAsync<1>      # Q 整个 kernel 只 load 一次，不需要多 stage
K:  PipelineTmaAsync<3>      # K + SF_K + delta_s 三份数据共享同一 mbarrier
V:  PipelineTmaAsync<3>      # V + SF_V 共享同一 mbarrier
```

TmaTransactionBytes 把三份数据的字节数加起来，`producer_acquire` 一次 commit 三个 TMA copy 到同一 barrier。

### Tile Scheduler

`StaticPersistentTileScheduler`：
- 每个 CTA 是 persistent worker，处理多个 (batch, head, m_block) 三元组
- grid_dim ≈ SM 数（RTX 5090 约 170 SM）
- causal 模式用 atomic-counter semaphore 做反向调度（长序列尾部先做，平衡负载）

### SM120 vs SM100 vs SM90

| GPU 系列 | Arch | 新硬件特性 |
|---|---|---|
| H100 / H20 | sm_90 | TMA, WGMMA, Cluster, `cp.async` pipeline |
| B100 / B200 | sm_100 | TMEM（累加器专用片上存储）, 更大 WGMMA tile |
| RTX 5090 | sm_120 | SM100 的 BlockScaled MMA，但无 TMEM |

SageAttention3 目标是 sm_120（RTX 5090），所以：
- 用 BlockScaled MMA（`SM120_16x32x64_TN_VS_NVFP4`）
- 不用 TMEM（sm_120 没有）
- TMA 复用 SM90 的（没有改变）

---

## 关键文件索引

### 本学习目录

| 文件 | 内容 |
|---|---|
| `01_gemm_sm80.cu` | CUTLASS 2.x device::Gemm，SM80，带注释 |
| `02_gemm_sm90.cu` | CUTLASS 3.x CollectiveBuilder，SM90，带注释 |
| `03_cute_layout.cu` | cute Layout 基础（host-only，可 print） |
| `04_cute_tensor_tile.cu` | Tensor + local_tile + partition（host-only）|
| `05_cute_tiled_copy_mma.cu` | TiledCopy + TiledMMA + ZipTensor（注释为主）|

### SageAttention3 文件（推荐阅读顺序）

| 顺序 | 文件 | 核心内容 | 难度 |
|---|---|---|---|
| 1 | `api.cu` | Python 入口，dtype 验证，mha_fwd | ⭐ |
| 2 | `params.h` | 参数结构体，所有 host→device 传参 | ⭐ |
| 3 | `launch.h` | host 端 launch，grid/cluster，kernel 指针 | ⭐⭐ |
| 4 | `kernel_ws.h` | kernel 入口，warp 角色，producer/consumer | ⭐⭐ |
| 5 | `softmax_fused.h` | online softmax + 融合量化，纯算法 | ⭐⭐⭐ |
| 6 | `tile_scheduler.h` | persistent scheduler | ⭐⭐ |
| 7 | `epilogue_tma_ws.h` | TMA store O，较标准 | ⭐⭐⭐ |
| 8 | `kernel_traits.h` | cute 类型密集区，所有 SmemLayout 定义 | ⭐⭐⭐⭐ |
| 9 | `blockscaled_layout.h` | SF tensor layout 推导 | ⭐⭐⭐⭐ |
| 10 | `cute_extension.h` | partition_fragment_SFA/SFB 等扩展 | ⭐⭐⭐⭐ |
| 11 | `utils.h` | fp4/ue4m3 转换的内联 PTX | ⭐⭐⭐ |
| 12 | `mainloop_tma_ws.h` | **核心**（908 行），最后读 | ⭐⭐⭐⭐⭐ |

### 外部资源

| 资源 | 用途 |
|---|---|
| [CUTLASS examples/cute/tutorial/](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial) | cute 官方入门例子 |
| [CUTLASS docs/cute/](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute) | Layout/Tensor/算法 文档 |
| [Dao-AILab/flash-attention hopper/](https://github.com/Dao-AILab/flash-attention) | FA3 Hopper kernel（阶段 2 目标）|
| PTX ISA 文档 `mbarrier.*` 章节 | mbarrier arrive/wait 语义 |
| PTX ISA 文档 `wgmma.*` 章节 | WGMMA register operand layout |
| PTX ISA 文档 `tcgen05.mma.blockscaled` | SM100/SM120 BlockScaled MMA |
| NVIDIA OCP MXFP4 spec | MX-FP4 格式定义 |

---

## 自测题

读完三个 cute 文件后用这 5 题验证自己的理解。

**Q1**  
`make_layout(make_shape(_4{}, _8{}), make_stride(_0{}, _1{}))` 的 `size` 和 `cosize` 分别是多少？为什么 `cosize < size`？

> A: size=32, cosize=8。stride[0]=0 意味着 4 个"行"全映射到同一物理位置，所以只需要 8 个物理元素。

**Q2**  
`local_tile(mQ, make_shape(_128{}, _64{}), make_coord(3, 0))` 返回的 Tensor 的 `gQ(2, 5)` 对应 `mQ` 的哪个逻辑坐标？

> A: mQ 的 (3×128+2, 0×64+5) = (386, 5)。

**Q3**  
`group_modes<0, 3>(tensor)` 之前 shape 是 `(a, b, c, d)`，之后 shape 是什么？

> A: `(a*b*c, d)`，前三个 mode 合并成一个。

**Q4**  
`partition_fragment_A(sQ)` 和 `partition_S(sQ)` 有什么本质区别？

> A: `partition_fragment_A` 给 **TiledMMA** 用，返回按 MMA register layout 组织的寄存器 fragment；`partition_S` 给 **TiledCopy** 用，返回按 Copy 搬运方式组织的 source 分片。两者的 layout 不同，用 `retile_D` 在两种视图之间转换。

**Q5**  
SageAttention 里 `make_zip_tensor(tSrQ, tSrSFQ)` 不用 zip 直接传 `tSrQ` 会怎样？

> A: `cute::gemm` 会选普通 fp4 MMA 而非 BlockScaled MMA，忽略 scale factor，计算结果错误（或编译期报错，因为 TiledMMA 的 atom 类型与 operand 类型不匹配）。

---

## 重要建议

### 关于阅读顺序

1. **不要直接读 `mainloop_tma_ws.h`**：它同时叠了 cute + Pipeline + TMA + BlockScaled MMA + softmax + quantize 五套东西，必须把每套单独学了再读。

2. **FA3 是 SageAttention 的"基底"**：FA3 代码结构和 SageAttention3 几乎一一对应。先把 FA3 读 3 遍，SageAttention 再看只剩 diff。

3. **cute 必须动手写**：看懂不等于会用。一定要在 cute tutorial 里实际写 `make_layout`、`local_tile`、`partition`，看 `print_layout` 的输出，才能真正建立直觉。

### 关于调试工具

```cpp
// cute 的 print 系列是最重要的调试工具
cute::print(layout);           // 打印 Layout 的 (shape, stride)
cute::print_layout(layout);    // 打印可视化网格（坐标→偏移）
cute::print(tensor.layout());  // 打印 Tensor 的 layout

// 任何看不懂的 cute 表达式，先 print，再理解
```

### 关于 PTX ISA 文档

下面几组指令的章节要读熟，是理解 CUTLASS 高层 API 的基础：
- `mbarrier.*`（arrive, wait, init）
- `cp.async.bulk.tensor`（TMA）
- `wgmma.mma_async.*`（SM90 WGMMA，register layout 表很重要）
- `tcgen05.mma.blockscaled.*`（SM100/SM120 BlockScaled MMA）

### 关于 Epilogue 手写时机

**99% 的情况不需要手写 SmemLayout**。用 EVT（Epilogue Visitor Tree）节点 compose。只在以下情况才直接写：
- 自定义 visitor 节点（非标准 broadcast）
- 多输出 tensor
- 非平凡 LayoutD

### 关于 cute 模板报错

cute 模板编译错误可能产生 200+ 行的 `templated from` 链。处理方法：
1. 找最底层的 `static_assert` 或 `error:`
2. 通常是 Layout 不匹配（size / stride 不符合 MMA/Copy atom 要求）
3. 用 `cute::print` 打出实际 layout，对照文档要求的格式
