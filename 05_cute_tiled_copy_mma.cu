// =====================================================================
//  05_cute_tiled_copy_mma.cu  ——  cute 第三层：TiledCopy + TiledMMA
//
//  前置：03 (Layout) + 04 (Tensor/Tile)
//
//  这是阶段 1 最难的一个文件，覆盖 SageAttention mainloop 里
//  剩余的所有 cute 核心用法。
//
//  编译（需要 SM90+ GPU）：
//    nvcc -arch=sm_90a -std=c++17 --expt-relaxed-constexpr \
//         -I/root/hzy/rtp-llm/bazel-rtp-llm/external/cutlass_h_moe/include \
//         -I/root/hzy/rtp-llm/bazel-rtp-llm/external/cutlass_h_moe/tools/util/include \
//         05_cute_tiled_copy_mma.cu -o 05_cute_tiled_copy_mma
//    ./05_cute_tiled_copy_mma
//
//  读完这个文件你能看懂 SageAttention mainloop_tma_ws.h 里
//  所有涉及 TiledCopy / TiledMMA / make_zip_tensor 的代码。
// =====================================================================

#include <cstdio>
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

using namespace cute;

// =====================================================================
// 第 1 章  TiledCopy：从 GMEM → SMEM 的分布式 Copy
// =====================================================================
//
// 在 GPU kernel 里，把一个 tile 从 global memory 搬到 shared memory
// 是所有 thread 协作完成的——每个 thread 搬一小块。
// TiledCopy 就是这个"协作方案"的描述。
//
// 概念结构：
//   CopyAtom     = 单个 thread 执行的最小 copy 操作
//                  （e.g. ld.global.v4.f32，一次 load 4×f32 = 16 bytes）
//   TiledCopy    = 把 CopyAtom 在 thread 之间平铺，覆盖整个 tile
//                  通过 make_tiled_copy(atom, thread_layout, value_layout) 创建
//
// API：
//   auto thr_copy = tiled_copy.get_thread_slice(thread_idx)
//   => 返回当前 thread 的"分片视图"
//
//   Tensor tSrc = thr_copy.partition_S(src_tensor)   // S = Source (from)
//   Tensor tDst = thr_copy.partition_D(dst_tensor)   // D = Destination (to)
//   => 从整个 src/dst tensor 中切出当前 thread 负责的部分
//
//   copy(tiled_copy, tSrc, tDst)
//   => 所有 thread 协作把 src 搬到 dst

// =====================================================================
// 第 2 章  TiledCopy 的 partition_S / partition_D 在做什么
// =====================================================================
//
// partition_S/D 的物理含义：
//   给定 TiledCopy 描述的"N 个 thread 每人搬 M 个元素"的方案，
//   partition_S(src) 返回一个 Tensor，其中：
//     - 第 0 个 mode 是 "copy atom 的形状"（单次搬运几个元素）
//     - 后续 mode 是 "这个 thread 要搬几批"（repeat count）
//   这样 for(int i=0; i<size<1>(tSrc); ++i) copy_atom(tSrc(_,i), tDst(_,i))
//   就完整搬完这个 thread 的份额。
//
// SageAttention mainloop_tma_ws.h:486-491 的代码：
//
//   auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
//   Tensor tQgQ = block_tma_q.partition_S(gQ);   // S = source in gmem
//   Tensor tQsQ = block_tma_q.partition_D(sQ);   // D = dest in smem
//
//   这里 tma_load_Q 是一个特殊的 TiledCopy——TMA copy。
//   partition_S 的结果 tQgQ 描述"TMA 要从 gmem 哪里搬数据"，
//   partition_D 的结果 tQsQ 描述"搬到 smem 哪里"。
//   然后 copy(tma, tQgQ, tQsQ) 发射一条 TMA 指令。

// =====================================================================
// 第 3 章  TiledMMA：把硬件 MMA 指令平铺给 warp/warp-group
// =====================================================================
//
// GPU 上 矩阵乘法分三层：
//   MMA Atom（单条硬件指令）
//   TiledMMA（把 atom 平铺给多个 thread/warp/warp-group，覆盖整个 tile）
//   cute::gemm(tiled_mma, A_frag, B_frag, C_frag)（调用）
//
// 创建方式：
//   auto tiled_mma = make_tiled_mma(
//       MmaAtom{},          // 硬件 atom（e.g. SM80_16x8x16_F32F16F16F32_TN）
//       AtomLayout{},       // atom 在 warp/thread 内的排列（几行几列几个 atom）
//       PermTile{}          // 总 tile 大小的排列
//   );
//
// 关键 API：
//   auto thr_mma = tiled_mma.get_thread_slice(thread_idx)
//   Tensor tSrQ = thr_mma.partition_fragment_A(smem_Q)
//     => 当前 thread 负责的 A 寄存器 fragment（从 SMEM Q 中切出）
//   Tensor tSrK = thr_mma.partition_fragment_B(smem_K)
//   Tensor tSrS = partition_fragment_C(tiled_mma, tile_MN)
//     => C 的累加器 fragment
//
//   cute::gemm(tiled_mma, tSrQ, tSrK, tSrS)
//     => 发射 MMA 指令，tSrS += tSrQ × tSrK

// =====================================================================
// 第 4 章  BlockScaled MMA 和 make_zip_tensor（SageAttention 独有）
// =====================================================================
//
// SM120 (RTX 5090) 引入了 BlockScaled MMA 指令：
//   一次 MMA 需要 4 个操作数：A_data, A_scale, B_data, B_scale
//   其中 A_scale / B_scale 是"每 16 个 fp4 元素共享一个 ue4m3 scale"
//
// cute 用 make_zip_tensor 把 (data, scale) 配对成一个联合 Tensor：
//   make_zip_tensor(tSrQ, tSrSFQ)
//   => 一个 ZipTensor，每个"元素"是 (fp4_element, ue4m3_scale) 的配对
//
// 然后 cute::gemm 看到 TiledMMA 是 BlockScaled 类型，
// 就知道展开成 SM120_16x32x64_TN_VS_NVFP4 指令（带 scale 的硬件 MMA）。
//
// SageAttention mainloop_tma_ws.h:721-722：
//   cute::gemm(tiled_mma_qk,
//              make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
//              make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
//              tSrS);

// =====================================================================
// 第 5 章  Fragment Layout：partition_fragment_A/B/C 返回什么
// =====================================================================
//
// partition_fragment_A(sQ) 返回的 Tensor 的 shape 是什么？
//
// 以 SM80 mma.sync.m16n8k16 为例：
//   一个 warp (32 thread) 参与计算 16×8 的输出 tile，
//   使用 16×16 的 A 和 16×8 的 B。
//
//   A fragment per thread：每个 thread 持有 A 的 2 个 fp16 = 1 次 ldmatrix.x2
//   shape = (MMA_M=2, MMA_K=2)，按 WGMMA/mma 规范排布在 register 里
//
// TiledMMA 把多个 atom 平铺后：
//   shape = (MMA_M * atom_repeat_M, MMA_K * atom_repeat_K, num_tiles)
//         = (8, 4, num_K_tiles) 之类的嵌套形式
//
// 这就是为什么 SageAttention mainloop 里 for(k_block) 的循环体里
// 用 tSrQ(_, _, k_block) 索引第 k_block 个 K tile 的 fragment。

// =====================================================================
// 第 6 章  SageAttention mma() 函数的 cute 代码全解
// =====================================================================

void print_sage_mma_walkthrough() {
    printf("\n=== SageAttention mainloop::mma() cute 代码逐行解析 ===\n\n");

    printf(
"// ---- 从 SMEM Tensor 创建 MMA fragment ----\n"
"\n"
"// sQ:   SMEM 里 Q 的 Tensor，layout = SmemLayoutQ（带 swizzle）\n"
"// sK:   SMEM 里 K 的 Tensor，layout = SmemLayoutK（多 stage）\n"
"// sSFQ: SMEM 里 Q 的 scale factor Tensor\n"
"\n"
"TiledMmaQK tiled_mma_qk;    // QK^T 计算用的 TiledMMA，\n"
"                             // atom = SM120_16x32x64_TN_VS_NVFP4\n"
"auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);\n"
"\n"
"// partition_fragment_A：从 SMEM 的 Q 中推算出\n"
"// 当前 thread 在 mma 中负责的 A 寄存器 fragment\n"
"// tSrQ 的 shape 大致是 (Atom_M, Atom_K, num_K_tiles)\n"
"// 每个 tSrQ(_, _, k) 是一个 K-tile 的 A 寄存器片段\n"
"Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);\n"
"\n"
"// 同理，K 的 B fragment\n"
"Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));\n"
"\n"
"// SF (scale factor) 的 fragment——专门为 BlockScaled MMA 设计\n"
"// partition_fragment_SFA/SFB 是 SageAttention 自己的 cute_extension.h 里加的扩展\n"
"Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);\n"
"Tensor tSrSFK = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);\n"
"\n"
"// C 累加器 fragment（fp32，由 partition_fragment_C 创建）\n"
"Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShape_MNK{}));\n"
"\n"
"// ---- 从 SMEM 到 Register 的 ldmatrix copy ----\n"
"\n"
"// make_tiled_copy_A：根据 TiledMMA 的 A 需求创建对应的 TiledCopy\n"
"// 使用 SM75_U32x4_LDSM_N（即 ldmatrix.sync.aligned.x4.m8n8）\n"
"auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);\n"
"auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);\n"
"\n"
"// as_position_independent_swizzle_tensor：\n"
"// 把 swizzle tensor 的基址调整成 position-independent 的\n"
"// （TMA 写入 SMEM 时用的绝对地址，ldmatrix 读时用相对 bank 地址，\n"
"//  两种寻址方式需要统一，这个 helper 做这个转换）\n"
"Tensor tSsQ = smem_thr_copy_Q.partition_S(\n"
"    as_position_independent_swizzle_tensor(sQ));\n"
"Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);\n"
"// retile_D：把 tSrQ（按 MMA layout 组织）的视图重排成 copy 期望的形状\n"
"\n"
"// ---- 主循环：copy + MMA 交织 ----\n"
"\n"
"// 先 copy K 的 block_0 到 register（ldmatrix）\n"
"copy(smem_tiled_copy_K, tSsK(_, _, block_0), tSrK_copy_view(_, _, block_0));\n"
"\n"
"// 主循环\n"
"for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {\n"
"    // 发射 BlockScaled MMA：\n"
"    //   tSrS += zip(tSrQ, tSrSFQ) × zip(tSrK, tSrSFK)\n"
"    //   zip_tensor 让 cute 知道这是 (data, scale) 配对，\n"
"    //   从而展开成 SM120_16x32x64_TN_VS_NVFP4 指令\n"
"    cute::gemm(tiled_mma_qk,\n"
"               make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),\n"
"               make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),\n"
"               tSrS);\n"
"    // 如果不是最后一个 block，预取下一个 K block 到 register\n"
"    if (k_block < size<2>(tSrQ) - 1) {\n"
"        copy(smem_tiled_copy_K, tSsK(_, _, k_block+1),\n"
"             tSrK_copy_view(_, _, k_block+1));\n"
"    }\n"
"}\n"
    );

    printf("\n[关键理解]\n");
    printf("cute::gemm 不是在你调用时才发射指令——\n");
    printf("它被编译器展开成一系列 mma.sync / wgmma.async / tcgen05.mma 调用。\n");
    printf("具体哪种指令取决于 tiled_mma_qk 的 atom 类型（SM80/SM90/SM120）。\n");
    printf("SageAttention 里 atom = SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4，\n");
    printf("所以展开成 Blackwell 的 BlockScaled MMA 指令。\n");
}

// =====================================================================
// 第 7 章  make_zip_tensor 的 Layout 语义
// =====================================================================
//
// make_zip_tensor(t_data, t_scale) 要求：
//   - t_data 和 t_scale 的 shape 有关联（data 的某个维度 = scale 的维度 × SFVecSize）
//   - 两个 Tensor 的 mode 一一对应
//
// 返回的 ZipTensor 里每个"元素"是 tuple<data_elem, scale_elem>。
//
// 当 cute::gemm 看到 ZipTensor 作为 MMA 的 A 或 B 操作数，
// 它知道这是 BlockScaled 格式，并从 Tensor 的 layout 中
// 读取 data 和 scale 的内存布局，检查与 MMA atom 要求是否匹配。
// 如果不匹配，编译期报错（static_assert）。
//
// 这就是为什么 blockscaled_layout.h 里的 SmemLayoutSFQ/K/V 的构造
// 要非常精确地匹配 SM120_16x32x64_TN_VS_NVFP4 的 SF operand layout——
// make_zip_tensor 在编译期验证。

void print_zip_tensor_explanation() {
    printf("\n=== make_zip_tensor 和 BlockScaled MMA 的关系 ===\n\n");
    printf(
"在普通 fp16 MMA 里：\n"
"  cute::gemm(tiled_mma, tSrA, tSrB, tSrC)  // tSrA 是普通 Tensor\n"
"\n"
"在 BlockScaled MMA（SageAttention）里：\n"
"  cute::gemm(tiled_mma,\n"
"             make_zip_tensor(tSrQ,   tSrSFQ),   // data + scale 配对\n"
"             make_zip_tensor(tSrK,   tSrSFK),\n"
"             tSrS)\n"
"\n"
"物理上，SM120 的 BlockScaled MMA 指令接受 5 个操作数：\n"
"  .imm = {A_data_regs, A_sf_regs, B_data_regs, B_sf_regs, C_acc_regs}\n"
"\n"
"cute 的 make_zip_tensor 让你在代码里仍然用"两个独立 Tensor"描述 A 和 SF，\n"
"cute::gemm 负责把它们拆分并填进正确的指令操作数槽。\n"
"\n"
"如果你看到 SageAttention cute_extension.h 里有：\n"
"  get_layoutSFA_TV(tiled_mma)  // 返回 A scale factor 的 thread-value layout\n"
"  get_layoutSFB_TV(tiled_mma)  // 返回 B scale factor 的 thread-value layout\n"
"这两个函数从 TiledMMA 里提取出 SF operand 的 TV layout（thread × value），\n"
"用来创建匹配的 TiledCopy，以确保 SMEM → Register 的 SF copy 是正确的布局。\n"
    );
}

// =====================================================================
// 第 8 章  整合：读懂 SageAttention 的 mma 函数入口
// =====================================================================
//
// 把前面所有概念对应到 SageAttention mainloop_tma_ws.h:573 开始的
// template mma() 函数，按顺序标注每一段的 cute 概念：

void print_mma_function_map() {
    printf("\n=== SageAttention mma() 函数 — cute 概念地图 ===\n\n");
    printf(
"STEP 1: 创建 SMEM Tensor [Chapter 03: Layout + Tensor]\n"
"  Tensor sQ  = make_tensor(make_smem_ptr(smem_q.begin()), SmemLayoutQ{});\n"
"  Tensor sSFQ = make_tensor(make_smem_ptr(smem_SFQ.begin()), SmemLayoutSFQ{});\n"
"\n"
"STEP 2: 创建 MMA thread 分片 [Chapter 05 §3]\n"
"  TiledMmaQK tiled_mma_qk;\n"
"  auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);\n"
"\n"
"STEP 3: 分配 MMA fragment [Chapter 05 §3, §5]\n"
"  Tensor tSrQ   = thread_mma_qk.partition_fragment_A(sQ);\n"
"  Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk); // SF 专用\n"
"  Tensor tSrS   = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShape));\n"
"\n"
"STEP 4: 创建 SMEM→Register 的 copy tiler [Chapter 05 §2]\n"
"  auto smem_thr_copy_Q = make_tiled_copy_A(...).get_thread_slice(thread_idx);\n"
"  Tensor tSsQ  = smem_thr_copy_Q.partition_S(as_position_independent...sQ);\n"
"  Tensor tSrQ_cv = smem_thr_copy_Q.retile_D(tSrQ);\n"
"\n"
"STEP 5: add_delta_s [无 cute，纯 recast 访问]\n"
"  auto acc_float4 = recast<float4>(tSrS);  // [Chapter 04 §4]\n"
"  acc_float4(...) = delta_s_data;\n"
"\n"
"STEP 6: pipeline wait + ldmatrix copy [Chapter 05 §2]\n"
"  consumer_wait(pipeline_k, smem_pipe_read_k);\n"
"  copy(smem_tiled_copy_K, tSsK(...), tSrK_copy_view(...));  // ldmatrix\n"
"  copy(smem_tiled_copy_SFK, ...);  // SF copy\n"
"\n"
"STEP 7: BlockScaled MMA [Chapter 05 §4]\n"
"  cute::gemm(tiled_mma_qk,\n"
"             make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),\n"
"             make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)),\n"
"             tSrS);\n"
"\n"
"STEP 8: Online softmax + quantize P to fp4 [无 cute，纯算术]\n"
"  softmax_fused.online_softmax_with_quant(tSrS, AbsMaxP, ...);\n"
"  // quantize lambda 把 fp32 tSrS -> fp4 tOrP + ue4m3 tOrSFP\n"
"\n"
"STEP 9: PV BlockScaled MMA [Chapter 05 §4]\n"
"  cute::gemm(tiled_mma_pv,\n"
"             make_zip_tensor(tOrP(_, _, v), tOrSFP(_, _, v)),\n"
"             make_zip_tensor(tOrVt(_, _, v), tOrSFVt(_, _, v)),\n"
"             tOrO);\n"
    );
}

int main() {
    printf("==============================================\n");
    printf("  05_cute_tiled_copy_mma.cu\n");
    printf("  TiledCopy + TiledMMA + BlockScaled MMA\n");
    printf("==============================================\n");

    printf("\n[本文件主要是注释性讲解，不产生数值输出]\n");
    printf("[配合阅读 SageAttention mainloop_tma_ws.h 使用]\n");

    print_sage_mma_walkthrough();
    print_zip_tensor_explanation();
    print_mma_function_map();

    printf("\n==============================================\n");
    printf("  阶段 1 完成后的自我检验问题：\n");
    printf("\n");
    printf("  Q1. make_layout(Shape<_4,_8>, Stride<_0,_1>) 的\n");
    printf("      size 和 cosize 分别是多少？为什么 cosize < size？\n");
    printf("\n");
    printf("  Q2. local_tile(mQ, (128,64), make_coord(m_block, _0{}))\n");
    printf("      返回的 Tensor 的 shape 是什么？\n");
    printf("      gQ(2, 3) 对应原始 mQ 的哪个位置？\n");
    printf("\n");
    printf("  Q3. group_modes<0,3>(tensor) 之前 shape 是 (a,b,c,d)，\n");
    printf("      之后 shape 是什么？\n");
    printf("\n");
    printf("  Q4. SageAttention 里 make_zip_tensor(tSrQ, tSrSFQ)\n");
    printf("      的作用是什么？不用 zip 直接传 tSrQ 会怎样？\n");
    printf("\n");
    printf("  Q5. partition_fragment_A(sQ) 和 partition_S(sQ) 的区别是什么？\n");
    printf("      哪个给 copy 用？哪个给 mma 用？\n");
    printf("==============================================\n");

    return 0;
}
