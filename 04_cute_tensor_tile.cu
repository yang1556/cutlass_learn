// =====================================================================
//  04_cute_tensor_tile.cu  ——  cute 第二层：Tensor + Tile + Partition
//
//  前置：读懂了 03_cute_layout.cu（知道 Layout 是什么）
//
//  本文件主要在 host 端演示（部分需要 device pointer 的例子用 cudaMalloc）。
//
//  编译运行：
//    nvcc -arch=sm_80 -std=c++17 --expt-relaxed-constexpr \
//         -I/root/hzy/rtp-llm/bazel-rtp-llm/external/cutlass_h_moe/include \
//         04_cute_tensor_tile.cu -o 04_cute_tensor_tile && ./04_cute_tensor_tile
//
//  读完这个文件你能看懂 SageAttention mainloop_tma_ws.h 里：
//    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(shape_Q);
//    Tensor gQ = local_tile(mQ(...), TileShape, coord);
//    Tensor tQgQ = block_tma_q.partition_S(gQ);
//    Tensor tOrO = partition_fragment_C(tiled_mma_pv, ...);
//    auto t = group_modes<0,3>(tensor);
// =====================================================================

#include <cstdio>
#include <vector>
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

// =====================================================================
// 第 1 章  Tensor = Pointer + Layout
// =====================================================================
//
// cute 的 Tensor 不是"数组"，而是"指针 + Layout"的配对。
//
//   make_tensor(ptr, layout)
//   => Tensor 对象，用 (i,j,...) 索引时等价于 ptr[layout(i,j,...)]
//
// Tensor 有两种存储位置的"味道"：
//   gmem_ptr / make_gmem_ptr  ← 全局内存指针
//   smem_ptr / make_smem_ptr  ← 共享内存指针
//   make_rmem_ptr             ← 寄存器（实际上就是普通指针）
//
// 但这些"味道"只在 cute 的类型系统里有区别，实际运行时没差别。
// 它们的作用是让编译器推断出正确的 copy atom（用 cp.async 还是 ldmatrix）。

void example_1_tensor_basics() {
    printf("\n======= Example 1: Tensor 基础 =======\n");

    // 1a. 最简单的 host 端 Tensor（用普通数组）
    float data[4 * 8];
    for (int i = 0; i < 32; ++i) data[i] = float(i);

    // row-major (4, 8) tensor
    auto layout = make_layout(make_shape(_4{}, _8{}), make_stride(_8{}, _1{}));
    auto tensor = make_tensor(data, layout);

    printf("[1a] tensor(row=2, col=3) = %.0f  (expected %.0f)\n",
           float(tensor(2, 3)), float(data[2*8 + 3]));
    // tensor(2, 3) = data[layout(2,3)] = data[2*8 + 3*1] = data[19] = 19

    // 1b. 用 () 操作做切片（slice）
    //   tensor(2, _) = 第 2 行，所有列 = 一个 1D tensor
    auto row2 = tensor(2, _);   // _ 是 cute 的"取全部"标记，等价于 Underscore{}
    printf("[1b] tensor(2, _) 即第 2 行:\n  ");
    for (int j = 0; j < 8; ++j) printf("%.0f ", float(row2(j)));
    printf("\n");   // 应该输出 16 17 18 19 20 21 22 23

    // 1c. 注意：cute 的切片不复制数据，只是改变 pointer + layout
    printf("[1c] &tensor(2,3) = %p, &data[19] = %p  (same? %s)\n",
           &tensor(2,3), &data[19],
           (&tensor(2,3) == &data[19]) ? "YES" : "NO");
    // 它们应该是同一地址。这是 cute 的"零拷贝视图"特性。
}

// =====================================================================
// 第 2 章  local_tile：从大 Tensor 切出一个 Tile
// =====================================================================
//
// 在 GPU kernel 里，一个大矩阵被所有 CTA 共同处理，每个 CTA 只负责一块。
// local_tile 就是"给我第 (m_block, k_block) 个 tile 的视图"。
//
// 用法：
//   local_tile(big_tensor, tile_shape, coord)
//   => 一个 Tensor，逻辑上是 big_tensor 里第 coord 个 tile，大小 = tile_shape
//
// SageAttention mainloop_tma_ws.h:473-474：
//   Tensor gQ = local_tile(mQ(_, _, bidh, bidb),
//                           select<0, 2>(TileShape_MNK{}),
//                           make_coord(m_block, _0{}));
//   => 取第 m_block 个 Q tile，大小 (kBlockM, kHeadDim)

void example_2_local_tile() {
    printf("\n======= Example 2: local_tile =======\n");

    // 模拟一个 (seqlen=512, headdim=64) 的 Q 矩阵
    constexpr int seqlen = 512, headdim = 64;
    constexpr int kBlockM = 128, kBlockK = 64;

    // 用 host 端数组模拟
    std::vector<float> Q_data(seqlen * headdim, 0.f);
    for (int i = 0; i < seqlen * headdim; ++i) Q_data[i] = float(i);

    // Global tensor：(seqlen, headdim) row-major
    auto mQ = make_tensor(Q_data.data(),
                          make_layout(make_shape (seqlen, headdim),
                                      make_stride(headdim, 1)));

    // local_tile：取第 m_block=1 个 tile（行 128~255，列全部）
    int m_block = 1;
    auto gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                         make_coord(m_block, 0));

    printf("[2a] gQ = local_tile(mQ, (128,64), (1,0))\n");
    printf("  gQ.shape  = (%d, %d)\n", (int)size<0>(gQ), (int)size<1>(gQ));
    // 应该是 (128, 64)

    // gQ(0, 0) 应该对应 Q_data[128 * 64 + 0 * 1] = Q_data[8192]
    printf("  gQ(0, 0)  = %.0f  (expected %.0f)\n",
           float(gQ(0, 0)), float(128 * headdim + 0));
    // 应该是 8192

    printf("  gQ(1, 0)  = %.0f  (expected %.0f)\n",
           float(gQ(1, 0)), float(129 * headdim + 0));
    // 应该是 8256

    // 2b. local_tile 的第三个参数可以有 _ 表示"保留所有 tile"
    //   local_tile(mK, TileShape, make_coord(_, _0{}))
    //   => 所有 N 方向的 tile，按 block 编号索引
    //   这就是 SageAttention 里 gK / gVt 的取法——返回的 tensor 多一个"block" 维度
    auto gK_all = local_tile(mQ,
                             make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                             make_coord(_, 0));
    // gK_all 的 shape 是 (kBlockM, kBlockK, num_tiles)
    int num_tiles = seqlen / kBlockM;
    printf("\n[2b] local_tile with _ in coord:\n");
    printf("  gK_all.shape = (%d, %d, %d)  num_tiles=%d\n",
           (int)size<0>(gK_all), (int)size<1>(gK_all), (int)size<2>(gK_all),
           num_tiles);
    // gK_all(_, _, n_block) 就是第 n_block 个 tile
}

// =====================================================================
// 第 3 章  local_partition：把 Tile 分配给各个 Thread
// =====================================================================
//
// local_tile 切出一个 CTA 要处理的 tile，
// local_partition 继续把这个 tile 分配给 CTA 里的各个 thread。
//
// 用法：
//   local_partition(tile_tensor, thread_layout, thread_id)
//   => 当前 thread (thread_id) 应该处理的那些元素的视图
//
// thread_layout 描述"CTA 里 thread 的二维/多维排列方式"。
// 比如 32x4 表示 32 个 thread 在列方向，4 个 thread 在行方向。

void example_3_local_partition() {
    printf("\n======= Example 3: local_partition =======\n");

    // 模拟一个 (8, 16) 的 tile，分给 (4, 4) 排列的 16 个 thread
    float tile_data[8 * 16];
    for (int i = 0; i < 128; ++i) tile_data[i] = float(i);

    auto tile = make_tensor(tile_data,
                            make_layout(make_shape(_8{}, _16{}),
                                        make_stride(_16{}, _1{})));  // (8,16) row-major

    // thread layout：4行 x 4列 的 thread 排布
    auto thr_layout = make_layout(make_shape(_4{}, _4{}),
                                  make_stride(_4{}, _1{}));
    // thread 0 = (row=0, col=0)，thread 1 = (row=0, col=1)，...
    // thread 4 = (row=1, col=0)，...

    // 取 thread 5 的分片（row=1, col=1）
    int thread_id = 5;
    auto thr5_tile = local_partition(tile, thr_layout, thread_id);

    printf("[3a] tile (8,16) 分给 (4,4) 的 thread，thread 5 的部分:\n");
    printf("  shape = (%d, %d)  (每个 thread 负责 8/4 x 16/4 = 2x4 个元素)\n",
           (int)size<0>(thr5_tile), (int)size<1>(thr5_tile));

    printf("  元素: ");
    for (int i = 0; i < size<0>(thr5_tile); ++i)
        for (int j = 0; j < size<1>(thr5_tile); ++j)
            printf("%.0f ", float(thr5_tile(i, j)));
    printf("\n");

    // 3b. SageAttention 里 partition 的典型用法
    printf("\n[3b] SageAttention 里的 partition 模式:\n");
    printf("  smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx)\n");
    printf("  tSsQ = smem_thr_copy_Q.partition_S(sQ)\n");
    printf("  => 每个 thread 只访问 sQ 里"属于自己"的那些位置\n");
    printf("  这和 local_partition 是同一个概念，只是包装在 TiledCopy 里\n");
}

// =====================================================================
// 第 4 章  group_modes + recast：mode 操作
// =====================================================================
//
// cute Tensor 的 shape 可以有多个 mode（维度）。有时候需要合并或重解释。
//
// group_modes<B, E>(tensor)：把 mode B 到 E-1 合并成一个 mode
//   SageAttention mainloop:493
//     Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
//   partition_S 返回 shape (CopyAtomShape, RestM, RestK, Tiles)，
//   group_modes<0,3> 把前 3 个维度合并，得到 (merged, Tiles)
//   => tKgK(_, n_block) 就是第 n_block 个 tile 的 TMA copy 视图
//
// recast<T>(tensor)：把 tensor 的元素类型 reinterpret 成 T
//   SageAttention mainloop:692
//     auto tSsDS_stage = recast<float4>(sDS(...));
//   => 把 float 的 SMEM tensor 重解释成 float4，一次读 4 个 float

void example_4_mode_ops() {
    printf("\n======= Example 4: group_modes + recast =======\n");

    // 4a. group_modes
    float data[2 * 3 * 4 * 5];
    for (int i = 0; i < 120; ++i) data[i] = float(i);

    auto t = make_tensor(data,
                         make_layout(make_shape(_2{}, _3{}, _4{}, _5{}),
                                     make_stride(_60{}, _20{}, _5{}, _1{})));
    printf("[4a] 原始 tensor shape: (%d,%d,%d,%d)\n",
           (int)size<0>(t), (int)size<1>(t), (int)size<2>(t), (int)size<3>(t));

    auto t_grouped = group_modes<0, 3>(t);  // 合并 mode 0,1,2
    printf("  group_modes<0,3> 后 shape: (%d, %d)\n",
           (int)size<0>(t_grouped), (int)size<1>(t_grouped));
    // size<0> = 2*3*4 = 24, size<1> = 5

    // 4b. recast
    float data4[16];
    for (int i = 0; i < 16; ++i) data4[i] = float(i);

    auto t_float = make_tensor(data4, make_layout(make_shape(_16{}), make_stride(_1{})));
    auto t_float4 = recast<float4>(t_float);
    printf("\n[4b] recast<float4>: 16 个 float -> %d 个 float4\n",
           (int)size(t_float4));
    // size = 16 / 4 = 4

    printf("  t_float4(1) = {%.0f, %.0f, %.0f, %.0f}\n",
           t_float4(1).x, t_float4(1).y, t_float4(1).z, t_float4(1).w);
    // 应该是 {4, 5, 6, 7}

    printf("\n[4c] SageAttention 里的 recast 用途:\n");
    printf("  add_delta_s lambda 里：\n");
    printf("  recast<float4>(acc)  => 把 fp32 accumulator 当 float4 来写，\n");
    printf("  一次写入 4 个 fp32，比逐个写 4 倍快（向量化 STG.E.128）\n");
}

// =====================================================================
// 第 5 章  整合：模拟 SageAttention mainloop 的 tensor 切片过程
// =====================================================================
//
// 把前面几章的知识串联起来，模拟 SageAttention mainloop_tma_ws.h
// 里的 Q tensor 切片过程（简化版，不涉及 TMA，只看 cute 部分）

void example_5_mainloop_simulation() {
    printf("\n======= Example 5: 模拟 mainloop Q tensor 切片 =======\n");

    // 参数（对应 kernel_traits.h 默认配置）
    constexpr int seqlen_q = 512, headdim = 128;
    constexpr int kBlockM = 128, kBlockK = 128;
    constexpr int num_heads = 8, batch = 2;

    // ---- Global memory tensor Q: (seqlen_q, headdim, num_heads, batch) ----
    // 在 SageAttention 里 stride_Q = (headdim, 1, seqlen_q*headdim, num_heads*seqlen_q*headdim)
    // 这里用简化版：batch 和 head 维度用小数组
    std::vector<float> Q_host(seqlen_q * headdim * num_heads * batch, 1.f);

    // (seqlen, d, head, batch) row-major stride
    int q_row_stride = headdim;
    int q_head_stride = seqlen_q * headdim;
    int q_batch_stride = num_heads * seqlen_q * headdim;

    auto mQ = make_tensor(
        Q_host.data(),
        make_layout(
            make_shape (seqlen_q, headdim, num_heads, batch),
            make_stride(q_row_stride, 1, q_head_stride, q_batch_stride)
        )
    );

    printf("[5a] mQ.shape = (%d, %d, %d, %d)\n",
           seqlen_q, headdim, num_heads, batch);

    // ---- Step 1：取特定 head 和 batch 的 Q ----
    int bidh = 2, bidb = 1;
    auto mQ_hb = mQ(_, _, bidh, bidb);  // shape: (seqlen_q, headdim)
    printf("[5b] mQ(_, _, bidh=%d, bidb=%d).shape = (%d, %d)\n",
           bidh, bidb, (int)size<0>(mQ_hb), (int)size<1>(mQ_hb));

    // ---- Step 2：local_tile 取 m_block 的 tile ----
    int m_block = 2;
    auto gQ = local_tile(mQ_hb,
                         make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                         make_coord(m_block, 0));
    printf("[5c] gQ = local_tile(mQ_hb, (128,128), (%d,0)).shape = (%d, %d)\n",
           m_block, (int)size<0>(gQ), (int)size<1>(gQ));

    // ---- Step 3：验证 gQ 指向正确的内存位置 ----
    // gQ(0,0) 应该对应 Q_host 的第几个元素？
    // row = m_block * kBlockM = 2 * 128 = 256
    // head = bidh = 2, batch = bidb = 1
    // offset = bidb * q_batch_stride + bidh * q_head_stride + 256 * q_row_stride + 0
    size_t expected_offset = size_t(bidb) * q_batch_stride
                           + size_t(bidh) * q_head_stride
                           + size_t(m_block * kBlockM) * q_row_stride;
    size_t actual_offset = &gQ(0, 0) - Q_host.data();
    printf("[5d] gQ(0,0) 在 Q_host 的偏移: actual=%zu, expected=%zu, %s\n",
           actual_offset, expected_offset,
           actual_offset == expected_offset ? "✓" : "✗");

    printf("\n[5e] 对比 SageAttention mainloop_tma_ws.h:473-474 的代码:\n");
    printf("  Tensor gQ = local_tile(mQ(_, _, bidh, bidb),\n");
    printf("                          select<0, 2>(TileShape_MNK{}),\n");
    printf("                          make_coord(m_block, _0{}));\n");
    printf("  我们的模拟代码做的是完全一样的事情，\n");
    printf("  只是 select<0,2>(TileShape_MNK{}) = (kBlockM, kHeadDim)。\n");
}

int main() {
    printf("==============================================\n");
    printf("  04_cute_tensor_tile.cu — Tensor + local_tile + partition\n");
    printf("==============================================\n");

    example_1_tensor_basics();
    example_2_local_tile();
    example_3_local_partition();
    example_4_mode_ops();
    example_5_mainloop_simulation();

    printf("\n==============================================\n");
    printf("  下一步：05_cute_tiled_copy_mma.cu\n");
    printf("  学习：TiledCopy（TMA/SMEM copy）+ TiledMMA（硬件 MMA 调度）\n");
    printf("        这两个是 SageAttention mainloop 里 cute 最难的部分\n");
    printf("==============================================\n");
    return 0;
}
