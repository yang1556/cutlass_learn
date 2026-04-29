// =====================================================================
//  03_cute_layout.cu  ——  cute 核心概念第一层：Layout = (Shape, Stride)
//
//  这个文件在 host 端运行（不需要 GPU）。每个 Example 会用
//  cute::print_layout() 打印一个可视化的 Layout，让你直接"看到"
//  每个逻辑坐标对应的内存偏移。
//
//  编译运行：
//    nvcc -arch=sm_80 -std=c++17 --expt-relaxed-constexpr \
//         -I/root/hzy/rtp-llm/bazel-rtp-llm/external/cutlass_h_moe/include \
//         03_cute_layout.cu -o 03_cute_layout && ./03_cute_layout
//
//  读完这个文件你能看懂：
//    SageAttention kernel_traits.h 里所有 using SmemLayout* = ... 的声明
// =====================================================================

#include <cstdio>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

// =====================================================================
// 第 1 章  Layout 的本质
// =====================================================================
//
// cute 里一切都建立在 Layout 上。Layout = (Shape, Stride)，两个嵌套元组。
//
//   Shape  描述"有多少个元素"，可以多维、嵌套。
//   Stride 描述"相邻两个元素在内存中差多少"，与 Shape 结构一致。
//
// 给定一个多维逻辑坐标 (i, j, ...)，它对应的物理内存偏移是：
//
//   offset = i * stride[0] + j * stride[1] + ...
//
// 如果 Shape/Stride 是嵌套的（hierarchical），就做递归的 inner_product。
//
// 关键点：cute 的 Layout 是"双层"的——外层结构描述 logical 坐标系，
// 内层递归展开成物理偏移。这比传统的"row-major/col-major"表达力强很多。

// =====================================================================
// 第 2 章  编译期常量 vs 运行期值
// =====================================================================
//
// cute 大量使用编译期常量。写法：
//
//   _1{}    等价于 cute::Int<1>{}   ← 编译期 1
//   _8{}    等价于 cute::Int<8>{}   ← 编译期 8
//   int(4)  或者 4                  ← 运行期 4
//
// 如果 Shape 和 Stride 全是编译期常量，整个 Layout 的 offset 计算在
// 编译期被展开为常量——零运行期开销。
//
// SageAttention 里：
//   Int<kBlockN / 64>{}  表示"运行期不知道但 template 实例化后确定"的值
//   _1{}, _0{}, _4{}     表示固定的编译期常量
//
// 混用时：编译期部分被 fold，运行期部分留到运行时算。

void example_1_basic_layouts() {
    printf("\n======= Example 1: 基础 Layout =======\n");

    // -----------------------------------------------
    // 1a. Row-major (4, 8) 矩阵
    //     4 行 × 8 列，行内相邻元素地址差 1，跨行地址差 8
    //     offset(i, j) = i * 8 + j * 1
    // -----------------------------------------------
    auto layout_row = make_layout(make_shape (_4{}, _8{}),
                                  make_stride(_8{}, _1{}));

    printf("\n[1a] Row-major (4x8): stride=(8,1)\n");
    print_layout(layout_row);
    // 输出会是一个 4x8 的网格，格子里填的是内存偏移：
    //   0  1  2  3  4  5  6  7
    //   8  9 10 11 12 13 14 15
    //  16 17 18 19 20 21 22 23
    //  24 25 26 27 28 29 30 31

    // -----------------------------------------------
    // 1b. Col-major (4, 8) 矩阵
    //     相邻行地址差 1，相邻列地址差 4
    //     offset(i, j) = i * 1 + j * 4
    // -----------------------------------------------
    auto layout_col = make_layout(make_shape (_4{}, _8{}),
                                  make_stride(_1{}, _4{}));

    printf("\n[1b] Col-major (4x8): stride=(1,4)\n");
    print_layout(layout_col);
    // 输出：
    //   0  4  8 12 16 20 24 28
    //   1  5  9 13 17 21 25 29
    //   2  6 10 14 18 22 26 30
    //   3  7 11 15 19 23 27 31

    // 两个 Layout 逻辑上都是"(4,8)的矩阵"，但物理布局不同。
    // cute 的 Tensor 用不同 Layout 包装同一段内存，就能在不改数据的
    // 情况下"切换"行主序/列主序的视角。

    // -----------------------------------------------
    // 1c. 验证：用 () 运算符查询坐标→偏移
    // -----------------------------------------------
    printf("\nlayout_row(2, 3) = %d  (expected 19)\n", (int)layout_row(2, 3));
    printf("layout_col(2, 3) = %d  (expected 14)\n", (int)layout_col(2, 3));
}

void example_2_nested_layouts() {
    printf("\n======= Example 2: 嵌��� Layout =======\n");

    // -----------------------------------------------
    // cute 最强大的地方：Shape 和 Stride 可以嵌套（hierarchical）。
    // 嵌套的目的是在一个 Layout 里同时描述多个层次的结构。
    //
    // 典型用途：GPU 编程中 warp 内的 thread 排布：
    //   - 一个 CTA 有 (warp_y × warp_x) 个 warp
    //   - 每个 warp 有 (lane_y × lane_x) 个 lane
    //   => 嵌套 layout: Shape<(lane_y, warp_y), (lane_x, warp_x)>
    //                   Stride<(s0, s1), (s2, s3)>
    // -----------------------------------------------

    // 2a. Shape<(2,4), (4,2)>  ←  M 维度里有 2×4 的子结构
    //                               N 维度里有 4×2 的子结构
    //     Stride<(1,8), (2,32)>
    //     offset(i, j) = inner_product(flatten(i), flatten(stride_i))
    //                    其中 i 是嵌套坐标
    //
    // 实际上这个 layout 把 (8, 8) 的矩阵用一种"tile 优先"的顺序排布：
    // 先在 2×4 的小 tile 内遍历，再跨 tile。
    auto layout_nested = make_layout(
        make_shape (make_shape (_2{}, _4{}), make_shape (_4{}, _2{})),
        make_stride(make_stride(_1{}, _8{}), make_stride(_2{}, _32{}))
    );

    printf("\n[2a] Nested Layout Shape<(2,4),(4,2)> Stride<(1,8),(2,32)>\n");
    print_layout(layout_nested);
    // print_layout 会把嵌套 shape "展平"后打印。
    // 理解它的关键：看相邻元素在内存中的距离。

    // 2b. 为什么 SageAttention 大量用嵌套 Layout？
    //
    // 以 TiledMmaQK 为例（kernel_traits.h:108-116）：
    //   AtomLayoutMNK = Layout<Shape<_8, _1, _1>>
    //   这表示：一个 warp-group 把 MMA atom 在 M 维度复制 8 份、
    //           在 N/K 维度不复制。
    //
    // 以 SmemLayoutAtomQ 为例（kernel_traits.h:129）：
    //   由 sm120_rr_smem_selector 自动选出，形如
    //   Swizzle<B,M,S> ∘ Layout<Shape<(M, ...), (K, ...)>, Stride<...>>
    //   嵌套是为了描述"swizzle 后的 SMEM tile"。

    // 2c. 用 rank / depth / size 查询 layout 属性
    printf("\n[2c] layout_nested 的属性:\n");
    printf("  rank  = %d  (几个维度)\n",  (int)rank(layout_nested));
    printf("  size  = %d  (总元素数)\n",  (int)size(layout_nested));
    printf("  cosize = %d (值域大小，即需要多少内存)\n",
                                          (int)cosize(layout_nested));
    // cosize 是"最大偏移 + 1"，不一定等于 size（有时会有空洞）
}

void example_3_zero_stride_broadcast() {
    printf("\n======= Example 3: zero-stride (broadcast) =======\n");

    // -----------------------------------------------
    // Stride 某个维度为 0 表示"广播"：
    // 这个维度上所有坐标映射到同一个物理元素。
    //
    // 这是 cute 里表达 broadcast 的标准方式，不需要额外复制数据。
    //
    // 在 SageAttention 里：
    //   LayoutSFP = Layout<
    //       Shape <(16, 4), 1, (kBlockN/64)>,
    //       Stride<( 0, 1), 0, 4>
    //   >
    //   stride 第一个子维度是 _0 ——
    //   意思是：16 个 thread 共享同一个 scale factor（同一物理地址）。
    //   这正好匹配 BlockScaled MMA 的 SF operand layout：
    //   每 16 个 fp4 data 对应 1 个 ue4m3 scale，
    //   所以持有这 16 个 data 的 16 个 thread，scale 只有 1 份。
    // -----------------------------------------------

    // 3a. 简单例子：8 个逻辑元素，全部广播到同一物理元素
    auto layout_broadcast = make_layout(make_shape (_8{}),
                                        make_stride(_0{}));
    printf("[3a] 全广播 size=8 stride=0:\n");
    print_layout(layout_broadcast);
    // 输出: 0 0 0 0 0 0 0 0   ← 全部映射到地址 0

    // 3b. SFP-like layout：部分维度广播，部分不广播
    //   Shape<(16, 4), kBlockN/64>
    //   Stride<(0, 1), 4>
    //   => 16 个元素共享同一个 scale（stride=0），4 个 scale 紧密排列（stride=1）
    //   => 相邻 64 个 data 元素的 scale 差 4（stride=4）
    constexpr int kBlockN = 128;
    auto layout_sfp = make_layout(
        make_shape (make_shape (_16{}, _4{}), Int<kBlockN / 64>{}),
        make_stride(make_stride(_0{},  _1{}), _4{})
    );
    printf("\n[3b] SFP-like layout (kBlockN=128):\n");
    print_layout(layout_sfp);
    // 注意：size=16*4*2=128，但 cosize 远小于 128——因为大量"共享"

    printf("  size   = %d\n", (int)size(layout_sfp));
    printf("  cosize = %d\n", (int)cosize(layout_sfp));

    // -----------------------------------------------
    // 理解了这个，你就能看懂 SageAttention 里为什么
    // quantize() lambda 要用 __shfl_xor_sync 重排 SFP：
    //
    //   blockscaled MMA 期望：lane 0~15 持有同一个 scale，lane 16~31 持有另一个
    //   但 quantize() 计算时每个 thread 各自算出自己那 8 个 data 的 absmax
    //   => 要把 16 个 lane 的 absmax reduce 成 1 个，再广播回去
    //   => shfl_xor_sync 在 quad 内完成这个 reduce+broadcast
    // -----------------------------------------------
}

void example_4_layout_algebra() {
    printf("\n======= Example 4: Layout 代数运算 =======\n");

    // -----------------------------------------------
    // cute 提供一套"代数"在 layout 上操作，类比线性代数：
    //
    //   composition (∘)：把两个 layout 串联，像函数复合
    //   complement      ：补全 layout（较少直接用）
    //   product         ：笛卡尔积
    //   coalesce        ：化简（合并可合并的维度）
    //   filter_zeros    ：去掉 stride=0 的维度（只看非广播维度）
    //
    // 这些运算让 cute 的 API（tile_to_shape、partition、local_tile）
    // 能在编译期把复杂的 layout 操作约简成简单的偏移计算。
    // -----------------------------------------------

    // 4a. tile_to_shape：把一个"atom layout"平铺到更大的 shape
    //   atom 是最小重复单元，tile_to_shape 把它重复铺满 total_shape
    //
    // 类比：如果 atom 是一块 8×8 的瓷砖，tile_to_shape 把它铺到 64×64 的地板
    auto atom = make_layout(make_shape (_8{}, _8{}),
                            make_stride(_8{}, _1{}));   // row-major 8x8
    auto tiled = tile_to_shape(atom, make_shape(_32{}, _32{}));  // 铺到 32x32
    printf("[4a] tile_to_shape: 8x8 atom -> 32x32\n");
    printf("  atom  size=%d cosize=%d\n", (int)size(atom),  (int)cosize(atom));
    printf("  tiled size=%d cosize=%d\n", (int)size(tiled), (int)cosize(tiled));
    // tiled 的 shape 是 32×32，但内部结构保留了 8×8 tile 的组织方式

    // 4b. coalesce：化简 layout
    //   如果相邻维度的 stride 关系满足"可合并"条件，合并成一维
    auto layout_2d = make_layout(make_shape (_4{}, _8{}),
                                 make_stride(_8{}, _1{}));  // (4,8) row-major
    auto layout_1d = coalesce(layout_2d);
    printf("\n[4b] coalesce (4,8)row-major:\n");
    printf("  before: size=%d rank=%d\n", (int)size(layout_2d), (int)rank(layout_2d));
    printf("  after:  size=%d rank=%d\n", (int)size(layout_1d), (int)rank(layout_1d));
    // row-major (4,8) 本来就是内存连续的，coalesce 后退化成 1D Layout<32, 1>

    // 4c. filter_zeros：去掉广播维度
    auto layout_with_bcast = make_layout(
        make_shape (_4{}, _8{}, _2{}),
        make_stride(_0{}, _1{}, _8{})   // 第一维是广播
    );
    auto layout_filtered = filter_zeros(layout_with_bcast);
    printf("\n[4c] filter_zeros:\n");
    printf("  before rank=%d  after rank=%d\n",
           (int)rank(layout_with_bcast), (int)rank(layout_filtered));
    // 去掉 stride=0 的维度后 rank 从 3 降到 2
    //
    // SageAttention 里用 filter_zeros 算 SF tensor 的 SMEM stage stride：
    //   append(stride(atom), size(filter_zeros(atom)))
    //   意思是：下一个 stage 紧跟在上一个 stage 的非广播元素后面

    printf("\n[重要结论]\n");
    printf("tile_to_shape / coalesce / filter_zeros 在 cute 里是编译期操作——\n");
    printf("只要参数是编译期常量（Int<N>），结果也是编译期常量。\n");
    printf("SageAttention 里大量 'using Foo = decltype(tile_to_shape(...))'\n");
    printf("就是把这些编译期结果固化成类型，零运行时开销。\n");
}

void example_5_smem_swizzle() {
    printf("\n======= Example 5: SMEM Swizzle =======\n");

    // -----------------------------------------------
    // 前面讲了 swizzle 解决 bank conflict 的原理。
    // cute 里 Swizzle<B, M, S> 的参数含义：
    //
    //   B = Base：翻转多少 bit（2^B 个 pattern）
    //   M = MBase：从地址的第几 bit 开始取"行号"用于 XOR
    //   S = Shift：把"行号" shift 多少位再 XOR 进"列号"
    //
    // 物理上，bank = (byte_addr / 4) % 32
    // 给定元素类型为 fp16（2 字节），Swizzle<3,3,3> 的效果是：
    //   把 byte_addr 的 bit [5:3]（即 element[7:4]）XOR 进 bit [8:6]
    //   结果：相邻 8 行的同一列位于不同 bank，消灭 conflict。
    //
    // cute 里 Swizzle 是一个"函数"，可以 compose 进 Layout：
    //   Layout<..., ...> ∘ Swizzle<B,M,S>
    //   = 带 swizzle 的 layout
    //
    // 使用时，CUTLASS 提供 as_position_independent_swizzle_tensor()
    // 让 smem pointer 对齐到 swizzle 的基址。
    //
    // SageAttention 里的 SmemLayoutAtomQ 就是一个带 swizzle 的 layout，
    // 由 sm120_rr_smem_selector<Element, K> 自动根据 dtype 和 K 维度选出。
    // -----------------------------------------------

    // 注：Swizzle 的实际使用需要结合 GPU smem，这里只打印它的 layout 结构
    using Swizzle3 = Swizzle<3, 3, 3>;  // fp16 @ SMEM 最常用的配置
    auto swizzle_layout = composition(
        Swizzle3{},
        make_layout(make_shape (_8{}, _8{}), make_stride(_8{}, _1{}))
    );

    printf("[5] Swizzle<3,3,3> ∘ Layout<(8,8), (8,1)>:\n");
    print_layout(swizzle_layout);
    // 你会看到：同一列的 8 个元素，在内存里不再连续排列，
    // 而是按 XOR 规律散落在不同位置——这就是 swizzle 消除 bank conflict 的方式。

    printf("\n[关键结论]\n");
    printf("SageAttention kernel_traits.h 里所有 SmemLayout* 都是\n");
    printf("这种 swizzle ∘ base_layout 的产物。你看到的 decltype(sm120_rr_smem_selector<>())\n");
    printf("返回的正是一个带 Swizzle 的 Layout——不需要你手写 B/M/S 参数，\n");
    printf("CUTLASS 根据 dtype + K 维度自动选出最优配置。\n");
}

int main() {
    printf("==============================================\n");
    printf("  03_cute_layout.cu — cute Layout 基础\n");
    printf("==============================================\n");
    printf("\n目标：读完此文件后能看懂 SageAttention 里\n");
    printf("  kernel_traits.h 所有 using SmemLayout* = ... 的声明\n");

    example_1_basic_layouts();
    example_2_nested_layouts();
    example_3_zero_stride_broadcast();
    example_4_layout_algebra();
    example_5_smem_swizzle();

    printf("\n==============================================\n");
    printf("  下一步：04_cute_tensor_tile.cu\n");
    printf("  学习：Tensor = ptr + Layout，local_tile，partition\n");
    printf("==============================================\n");
    return 0;
}
