代码全部在 SageAttention/sageattention3_blackwell/sageattn3/blackwell/，约 3800 行。本质是一个 MX-FP4 量化版的 FlashAttention-3，跑在 Blackwell（sm_120/121，对应 RTX 5090）上。先讲创新点，再讲实现。

1. 算法大图
标准 FA3：


Q, K, V (bf16) → softmax(Q @ K^T) @ V → O (bf16)
SageAttention3：


Q, K, V 已经 host 端预量化为：
   data: fp4 (e2m1)，2 bit 指数 + 1 bit 尾数
   scale: fp8 (ue4m3)，每 16 个 fp4 共享一个

         + 校正项 delta_s: fp32 (M, N) 矩阵

→ MX-FP4 attention → O (bf16/fp16)
关键创新（也是这篇 paper 的核心）有两个：

(a) MX-FP4 (Microscaling FP4)：每 16 个 fp4 元素配一个 fp8 scale。Blackwell 硬件直接支持这个格式的 MMA —— 所谓 BlockScaled MMA，一条指令就把 (data, scale) 配对喂进去算。

(b) delta_s 校正项：fp4 精度太低，直接量化 attention 误差大。SageAttention 的 trick 是 预先在 host 端算一个 (M, N) 的 fp32 修正矩阵 delta_s，在每个 K 块的 QK^T 之后、softmax 之前加上去。代码里就一行：


// mainloop_tma_ws.h:691-704
auto add_delta_s = [&](auto& acc) {
    auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
    auto acc_float4 = recast<float4>(acc);
    // ... 把 sDS 里的 fp32 直接累加到 mma 累加器 acc 上
};
然后在 QK GEMM 之前调用 mainloop_tma_ws.h:718 add_delta_s(tSrS) —— 等价于 acc = acc + delta_s，在 fp32 精度下补偿 fp4 量化的偏置项。

注意 delta_s 是 host 端预先按 token 算好的（看 api.cu:50 的 delta_s 参数），所以 kernel 只 load + add 一下，几乎零成本。这是 "8-bit attention" 系列里 SageAttention 一直在做的核心思想：把高精度 cost 移到 offline，runtime 只跑低精度 + 微小修正。

2. Blackwell 硬件特性使用
先把代码里出现的几个新东西定位清楚：

硬件特性	代码出现位置	说明
BlockScaled MMA (sm_120)	kernel_traits.h:113 cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4	硬件指令 tcgen05.mma.blockscaled（sm_100）/ 对应 sm_120 变体。一次输入 (A_fp4, A_scale_fp8, B_fp4, B_scale_fp8) 四个 tensor。
MXFP4 数据类型	kernel_traits.h:75 cutlass::nv_float4_t<cutlass::float_e2m1_t>	NVIDIA MXFP4 微缩格式
UE4M3 scale 类型	kernel_traits.h:93 cutlass::float_ue4m3_t	4-bit 无符号指数 + 3-bit 尾数，专门用作 MX 格式的 block scale
TMA	SM90_TMA_LOAD	Blackwell 没改 TMA，复用 SM90 的
没用 TMEM：B100/B200 (sm_100) 引入了 Tensor Memory（专门给 mma 累加器用的片上存储），但这个 kernel 的目标是 sm_120 (RTX 5090)，sm_120 不支持 TMEM。所以累加器还是在寄存器里。这也是为什么 api.cu:222 检查 is_sm120 || is_sm121 而不是 sm_100。

3. Warp 角色编排（kernel_ws.h）
跟 FA3 一脉相承，但角色更细。kBlockM=128 时一个 CTA 12 warps 分成 3 个 warp-group：


WarpGroup 0 = Producer (4 warps)
   ├── Warp 0 (Mainloop): TMA load Q/K/V + scales + delta_s
   └── Warp 1 (Epilogue): TMA store O，等 mma 完成
   └── Warp 2/3: idle (registers donated to consumers)
WarpGroup 1 = Consumer 0 (4 warps)  ← 算 mma + softmax
WarpGroup 2 = Consumer 1 (4 warps)  ← 算 mma + softmax (cooperative，与 C0 协作算同一 tile)
kernel_ws.h:140-167 是分发逻辑：


if (warp_group_role == WarpGroupRole::Producer) {
    cutlass::arch::warpgroup_reg_dealloc<24>();   // 让出寄存器
    if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // 持续 TMA load
        for (each work_tile) { collective_mainloop.load(...); }
    } else if (producer_warp_role == ProducerWarpRole::Epilogue) {
        // 持续 TMA store
        for (each work_tile) { barrier_o.wait(); ...tma_store(...); barrier_o.arrive(); }
    }
} else {
    cutlass::arch::warpgroup_reg_alloc<232>();    // 申请大量寄存器
    for (each work_tile) {
        collective_mainloop.mma(...);              // QK -> softmax -> PV
        barrier_o.wait();
        collective_epilogue.mma_store(...);        // 累加器 -> SMEM
        barrier_o.arrive();
    }
}
关键 trick：warpgroup_reg_dealloc<24>() + warpgroup_reg_alloc<232>()。Hopper/Blackwell 的 warpgroup-register-reallocation 指令——producer 不需要那么多寄存器（只发 TMA 指令），把寄存器配额还给 consumer，让 consumer 多攒累加器。这是 fa3 里学来的硬技巧。

4. 主循环（mainloop_tma_ws.h）
mma() 函数（L573-L903）是整个 kernel 的核心。流程：


for n_block in [n_block_max-1, ..., 0]:
   wait K + SF_K + delta_s ready in SMEM        ← consumer_wait(pipeline_k)
   add_delta_s(tSrS)                            ← acc = delta_s（直接初始化）
   for k_block in K_subtiles:
       gemm_qk(zip(Q,SFQ), zip(K,SFK), tSrS)    ← BlockScaled MMA（带 scale）
   apply_causal_mask(tSrS)
   online_softmax_with_quant(tSrS, AbsMaxP)     ← softmax + 同时算 P 的 absmax
   quantize(tSrS → tOrP fp4 + tOrSFP)           ← P 量化成 fp4
   wait V + SF_V ready                          ← consumer_wait(pipeline_v)
   for v_block:
       gemm_pv(zip(P,SFP), zip(V,SFV), tOrO)    ← BlockScaled MMA
   rescale_o(tOrO_store, tOrO)                  ← online softmax 的 max 变化时重缩放
softmax_fused.finalize(tOrO_store)              ← 除以 sum
几个值得讲的细节：

4.1 BlockScaled GEMM 的调用方式

// L721
cute::gemm(tiled_mma_qk,
           make_zip_tensor(tSrQ(_,_,k), tSrSFQ(_,_,k)),    // A: data + scale 配对
           make_zip_tensor(tSrK(_,_,k), tSrSFK(_,_,k)),    // B: data + scale 配对
           tSrS);                                           // C: fp32 acc
make_zip_tensor 是 cute 的语法糖：把 (data tensor, scale tensor) 打包成一个 tensor 对象，里面每个"元素"实际是 (fp4_packed, ue4m3) 配对。cute::gemm 看到 tiled_mma_qk 的 atom 是 SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4，就知道展开成硬件 BlockScaled MMA 指令。

这是 cute 抽象的精妙之处——数据 + scale 在编译期就是一个对象，逻辑上跟普通 GEMM 写法一样，物理上展开到硬件指令。

4.2 在线 softmax 与 量化融合
softmax_fused.h:40-137 online_softmax_with_quant 同时做三件事：

跨 thread shuffle reduce 算 row_max
算 exp2(x * softmax_scale - max_scaled)
同时算 AbsMaxP（softmax 后 P 的 absmax，用作下一步 fp4 量化的 scale）
巧妙的地方在 L70-72：


const float max_scaled = (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
fp8_scalexfp4_scale_log2 = log2(1/(448 * 6)) —— 把 fp8 ue4m3 的最大值（448）和 fp4 的最大值（6）一起塞进 max 项，这样后续做 exp2 时 P 的范围天然就映射到 fp4 表达范围。用一行常量加法，把 quantize 的 rescale 融进了 softmax 的 max scaling。

4.3 P 的 fp4 packing 与 SFP 重排
mainloop_tma_ws.h:750-797 quantize lambda：

把 8 个 fp32 packed 成 4 个 fp4（一个 uint32）：utils.h 里 packed_float_to_e2m1 用一段 PTX 内联汇编做硬件 fp32→fp4 转换
AbsMaxP 转 ue4m3：packed_float_to_ue4m3
然后用 __shfl_xor_sync(..., 2) 在 quad 内的 thread pair 间交换 SFP，把 scale 排成 PV MMA 期望的 (lane, mma_m) layout
这段是整个 kernel 最 mma-layout-依赖 的部分——每个 thread 的 SFP 在哪个 register slot，必须跟 BlockScaled MMA 指令对 SF 操作数的要求精确匹配。

4.4 SFP 的硬件 layout（kernel_traits.h:160-171）

using LayoutSFP = decltype(make_layout(
    make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBlockN / 64>{}),
    make_stride(make_stride(_0{}, _1{}), _0{}, _4{})
));
这是个 嵌套 + 含 zero-stride 的 cute Layout：

Stride<_0, _1> 的 _0 表示 16 个 thread 共享同一份 scale（lane 内 broadcast）
Stride _4 表示 N 方向相邻 64 个元素的 scale 步长 4
这个布局的目的就是匹配 sm_120 BlockScaled MMA 对 SF 操作数的 register layout 要求
这种"layout 用 zero-stride 表达 broadcast / 用嵌套表达 thread-quad 关系"是 cute 的核心表达力，FA3 / DeepGEMM / SageAttention 都重度依赖。

5. Pipeline 编排
3 个独立 pipeline，各自管自己的 mbarrier：


// kernel_traits.h:60-62 SharedStorage 定义
typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;        // Q 只 load 一次
typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;  // K 多 stage（=3）
typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;  // V 多 stage（=3）
每个 K-block 的 K + SF_K + delta_s 三份数据共享同一 mbarrier——producer 一次 commit 三次 TMA copy 到同个 barrier transaction，consumer 一次 wait：


// mainloop_tma_ws.h:516-523
pipeline_k.producer_acquire(smem_pipe_write_k);
copy(tma_load_K.with(*pipeline_k.producer_get_barrier(...), mcast),    tKgK(_,n), tKsK(_,idx));
copy(tma_load_SFK.with(*pipeline_k.producer_get_barrier(...), mcast),  tKgSFK(_,n), tKsSFK(_,idx));
copy(tma_load_DS.with(*pipeline_k.producer_get_barrier(...), mcast),   tDSgDS(_,n), tDSsDS(_,idx));
++smem_pipe_write_k;
TmaTransactionBytesK (L146-149) 把三者 byte 数加起来，TMA 一次发 3 个 copy 但只算 1 个 transaction。

6. Tile Scheduler
tile_scheduler.h 用的是 StaticPersistentTileScheduler：

一个 CTA 是 persistent worker，处理多个 (batch, head, m_block) 三元组
grid_dim 取 SM 个数（170 in launch.h:89，对应 GB300 / RTX 5090 的 SM 数）
causal mask 时用 atomic-counter（tile_count_semaphore）做反向调度——长序列尾部先做（计算重的 tile 先调度），短的后做，平衡负载
7. 总结：相比 FA3 的差异
维度	FA3 (Hopper)	SageAttention3 (Blackwell)
Q/K/V dtype	bf16/fp16	fp4 (e2m1) + fp8 (ue4m3) scale
MMA 指令	wgmma.async	sm_120 BlockScaled MMA
精度补偿	无（bf16 自身够）	host 预算 delta_s (fp32) 在 acc 加
Softmax	online softmax → bf16 P	online softmax → fp4 P + ue4m3 SFP，融合在一个函数
Pipeline	Q/K/V 各自 stage	Q 1 stage + K(含 SF_K, delta_s) 3 stage + V(含 SF_V) 3 stage
Warp 角色	1 producer + 2 consumer	同样，但 producer 内部分 Mainloop / Epilogue
寄存器策略	warpgroup reg realloc	同样
TMA 用法	SM90 TMA	复用 SM90 TMA
TMEM	n/a	不用（sm_120 没有）
要进一步看哪一块？三个候选：

packed_float_to_e2m1 / packed_float_to_ue4m3 的 PTX 实现（utils.h）—— 看硬件 fp4/ue4m3 类型转换怎么写
epilogue_tma_ws.h —— TMA store O 比较标准，没 SageAttention 独有的
cute_extension.h + partition_fragment_SFA/B —— BlockScaled MMA 的 cute 框架扩展，能看到 cute 怎么把 (data, scale) 作为 zip tensor 表达
这些代码是用什么写的
四样东西堆叠出来的：

1. CUDA C++17 (NVCC 编译)
底层语言。所有 .cu 文件本质是 C++ 源码，用 nvcc 编译。代码用了大量 C++17 特性：

if constexpr 编译期分支
decltype(auto) / 类型推导
模板模板参数
fold expressions
2. CUTLASS 3.x (NVIDIA 模板库)
整个高层抽象都是 CUTLASS 提供的：

cutlass::PipelineTmaAsync<> —— mbarrier-based 多 stage pipeline
cutlass::arch::warpgroup_reg_alloc/dealloc —— 寄存器重分配 PTX 包装
cutlass::ClusterLaunchParams + launch_kernel_on_cluster —— cluster 启动
cutlass::FastDivmod —— 把动态除法转成乘法+移位
cutlass::float_e2m1_t / float_ue4m3_t / nv_float4_t —— Blackwell 的 fp4/ue4m3 类型
CUTLASS 你可以理解为"把 GEMM/Attention 这种重模板代码模式化的库"——它不是黑盒 kernel（不像 cuBLAS），而是给你一堆模板组件，让你拼自己的 kernel。

3. cute (CUTLASS 内的几何代数 DSL)
代码里所有 cute::Layout、cute::Tensor、cute::make_zip_tensor、cute::gemm、partition_fragment_*、local_tile、tile_to_shape、get_layoutSFA_TV 都是 cute 的。

cute 是 CUTLASS 3.x 引入的"小语言"：用 (Shape, Stride) 双层结构在编译期描述任意 layout（包括 swizzle、broadcast、嵌套 tile），用 local_tile / partition / composition 等代数运算切 tile，整套表达式完全在编译期展开成原生指针运算。

简单说：cute 让你用像数学公式一样的代码写 GPU 数据布局，不用手算偏移。SageAttention 这套 SF 的复杂 layout（带 zero-stride broadcast、嵌套 thread-quad）没有 cute 几乎写不出来。

4. 内联 PTX 汇编
最底层的硬件指令——尤其是新格式（fp4/ue4m3）的 packing/unpacking——CUTLASS 还没全包好，作者自己写了 PTX。例如 utils.h 里的 packed_float_to_e2m1 应该是这样的：


asm volatile(
    "cvt.rn.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4, %5, %6, %7, %8};\n"
    : "=r"(out)
    : "f"(a0), "f"(a1), "f"(a2), "f"(a3), "f"(a4), "f"(a5), "f"(a6), "f"(a7)
);
（具体形式没看，可以打开看，但模式是这样：��条 PTX 指令把 8 个 fp32 转换并 pack 成 1 个 uint32 = 4 个 fp4 × 2 = 8 个 fp4）

类似的还有 flash::ptx_exp2（用 ex2.approx.f32 单条指令算 exp2，比 libdevice 快）、shfl_xor_sync 包装等。

5. pybind11 (Python ↔ C++ 桥)
api.cu:338-341：


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
}
配合 setup.py 用 torch.utils.cpp_extension.BuildExtension 编译，Python 端 import sageattn3 就能直接调。

一个直观对比
如果你拿 PyTorch + Triton 写一个 fp4 attention：

数据布局 → 用 numpy/torch tensor 的 stride
pipeline → Triton 编译器自动 pipeline
mma → tl.dot
mbarrier → 没有，靠编译器
TMA → 没有，自动 cp.async
fp4 → Triton 还没原生支持
每一项都"够用"但不是最优。SageAttention3 这种代码就是"每一项都要榨干硬件"的产物，所以 必须用 CUDA C++ + CUTLASS + cute + 内联 PTX 这套精确控制硬件的栈。换言之：能用 CUTLASS 的项目，就是性能要求最极致的项目。

按代码量分布：约 90% 是 cute / CUTLASS 模板拼装（using xxx = decltype(...)、make_layout(...)、tile_to_shape(...)），5% 是真正的算法逻辑（softmax、quantize 这些 lambda），5% 是 Python binding 和 host 端胶水。