// =====================================================================
//  02_gemm_sm90.cu  ——  最小可运行的 SM90 (H100/H20) CUTLASS 3.x GEMM
//
//  目标：D = alpha * (A @ B) + beta * C
//        A: (M, K)  bf16  row-major
//        B: (K, N)  bf16  col-major   (TN 约定，跟 SM80 例子一致)
//        C/D: (M, N)  bf16  row-major
//
//  与 01_gemm_sm80 对比，CUTLASS 3.x 的范式有四点本质变化：
//    (a) 不再手写 ThreadblockShape/WarpShape/InstructionShape，
//        改成 TileShape + ClusterShape + KernelSchedule，
//        让 CollectiveBuilder 自动选最合适的 collective op。
//    (b) Mainloop 和 Epilogue 是两个独立的 "Collective"，分别构建后拼装。
//    (c) 数据搬运不再是 cp.async + ldmatrix，而是 TMA（一条指令搬一个 tile）。
//    (d) MMA 不再是 mma.sync.m16n8k16，而是 wgmma.async（异步 + warp-group 级）。
//        软件流水的实现方式也跟着变成 "warp specialization"
//        （专门一组 warp 做 TMA load，另一组做 WGMMA）。
//
//  所有这些都在 KernelSchedule = KernelTmaWarpSpecializedCooperative
//  这一个标签里"打包决定"。CollectiveBuilder 看到这个 tag 自动展开成
//  ~3000 行 cute 元编程代码（CUTLASS 内部）。我们只需要写顶层声明。
// =====================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// ---- CUTLASS 3.x headers --------------------------------------------
//   GemmUniversalAdapter   —— 等价于 SM80 的 device::Gemm，host 侧封装
//   GemmUniversal kernel   —— 模板 kernel，由 Mainloop+Epilogue 组装
//   collective::CollectiveBuilder  —— 自动构建 Mainloop / Epilogue
//   cute/tensor.hpp        —— cute 命名空间，包含 Shape/Stride/Layout
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

// ---- 1. 类型定义 ----------------------------------------------------
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;   // TN 约定，理由见 SM80 第 5 章
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// 讲解第 1 章：对齐
//   TMA 要求 16 字节对齐。bf16 = 2 字节 → 一次最少搬 8 个元素。
//   公式：Alignment = 128 bits / sizeof_bits<T> = 8 (bf16) / 16 (int8)
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementC>::value;

// 讲解第 2 章：TileShape / ClusterShape
//   TileShape <128,128,64>: 一个 CTA 算 128×128 输出，K-tile 是 64
//      ↑ 注意比 SM80 的 K-tile 32 大了一倍——TMA 一次搬一个完整 tile 更划算
//   ClusterShape <1,1,1>: 不组成多 SM cluster，最朴素配置
//      （cluster 的好处见讲解第 3 章；这里关掉以聚焦最小可运行版）
using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_1,   cute::_1,   cute::_1>;

// 讲解第 4-5 章：Schedule 决定 mainloop / epilogue 的实现策略
//   KernelTmaWarpSpecializedCooperative
//     - "Tma":           用 TMA 搬数据
//     - "WarpSpecialized": producer warp（搬数据）/ consumer warp（算）分离
//     - "Cooperative":   两组 consumer warp-group 共同算一个 tile（更高带宽利用）
//   还有一个变体 "PingPong"，两组 warp-group 各算自己的 tile（更适合中等 K）
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

// ---- 2. CollectiveEpilogue ------------------------------------------
//
// CollectiveBuilder 的 11 个模板参数看着吓人，但分组其实很清楚：
//   <Arch, OpClass>                    —— 在哪种 SM 上、用 Tensor Core
//   <TileShape, ClusterShape, EpiTile> —— tile 几何
//   <ElemAcc, ElemCompute>             —— 累加 / 标量精度
//   <ElemC, LayoutC, AlignC>           —— 输入 C
//   <ElemD, LayoutD, AlignD>           —— 输出 D
//   <Schedule>                         —— 上面的 epilogue schedule
//
// 它产物 ::CollectiveOp 是一个类型，封装了：
//   - 累加器 → SMEM staging → 写回 D 的整套搬运
//   - alpha/beta scaling
//   - 复用 SMEM（StageCountAutoCarveout 在 mainloop 那边会用到）
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,    // 子 tile 大小自动选
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,                      // 输入 C
    ElementC, LayoutD, AlignmentD,                      // 输出 D（同 dtype）
    EpilogueSchedule
>::CollectiveOp;

// ---- 3. CollectiveMainloop ------------------------------------------
//
// 类似的 12 个参数，分组：
//   <Arch, OpClass>
//   <ElemA, LayoutA, AlignA>
//   <ElemB, LayoutB, AlignB>
//   <ElemAcc>
//   <TileShape, ClusterShape>
//   <StageCount, KernelSchedule>
//
// StageCountAutoCarveout = "请帮我算 epilogue 占用了多少 SMEM，
// 剩下的 SMEM 全部用来做多 stage pipeline，stage 数自动选最大"
//   ↑ 这就是 SM80 例子里 Stages=3 的"自动版"
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

// ---- 4. 装配 GemmKernel + GemmUniversalAdapter ----------------------
//
// 注意 ProblemShape 是 4 维的：(M, N, K, L)，L 是 batch
// 普通 GEMM L=1，相当于 batch=1 的 batched GEMM
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ---- 5. 朴素 reference（host）---------------------------------------
void reference_gemm(const std::vector<__nv_bfloat16>& A, int lda,
                    const std::vector<__nv_bfloat16>& B, int ldb,
                    std::vector<__nv_bfloat16>& D, int ldd,
                    int M, int N, int K, float alpha, float beta) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                float a = __bfloat162float(A[m * lda + k]);   // row-major
                float b = __bfloat162float(B[n * ldb + k]);   // col-major
                acc += a * b;
            }
            float c_old = __bfloat162float(D[m * ldd + n]);
            D[m * ldd + n] = __float2bfloat16(alpha * acc + beta * c_old);
        }
    }
}

// ---- 6. main --------------------------------------------------------
int main() {
    int M = 1024, N = 1024, K = 1024;     // 选较大尺寸，TMA/WGMMA 才发挥
    int L = 1;                            // batch
    float alpha = 1.0f, beta = 0.0f;

    // host data
    std::vector<__nv_bfloat16> hA(M * K), hB(K * N), hD(M * N);
    std::vector<__nv_bfloat16> hRef(M * N, __float2bfloat16(0.f));
    for (int i = 0; i < M * K; ++i) hA[i] = __float2bfloat16((rand() % 17 - 8) / 8.0f);
    for (int i = 0; i < K * N; ++i) hB[i] = __float2bfloat16((rand() % 17 - 8) / 8.0f);
    for (int i = 0; i < M * N; ++i) hD[i] = __float2bfloat16(0.f);

    // device data
    ElementA *dA; ElementB *dB; ElementC *dC, *dD;
    cudaMalloc(&dA, M * K * sizeof(ElementA));
    cudaMalloc(&dB, K * N * sizeof(ElementB));
    cudaMalloc(&dC, M * N * sizeof(ElementC));
    cudaMalloc(&dD, M * N * sizeof(ElementC));
    cudaMemcpy(dA, hA.data(), M * K * sizeof(ElementA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), K * N * sizeof(ElementB), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(ElementC));

    // 讲解第 6 章：cute 的 Stride
    //   StrideA 是 cute::Stride<int64_t, _1, int64_t>（举例），表示
    //     - 沿 M 方向走 1 步在内存中跨多少元素 = K
    //     - 沿 K 方向走 1 步 = 1（连续）
    //     - 沿 batch 方向走 1 步 = M*K
    //   make_cute_packed_stride 是辅助函数：传入 (M, K, L)，自动算出 (K, _1, M*K)
    //   而对 col-major B，传入 (N, K, L)，自动算出 (_1, K, K*N)
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // 讲解第 7 章：Arguments 是嵌套的，结构对应 (mode, problem, mainloop, epilogue, hw, scheduler)
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        { dA, stride_A, dB, stride_B },               // mainloop args
        {                                              // epilogue args
            { alpha, beta },                           //   thread args (LinearCombination)
            dC, stride_C,                              //   C ptr + stride
            dD, stride_D                               //   D ptr + stride
        },
        hw_info
    };

    Gemm gemm_op;
    size_t ws = gemm_op.get_workspace_size(args);
    void* d_ws = nullptr;
    if (ws) cudaMalloc(&d_ws, ws);

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "can_implement failed: " << cutlassGetStatusString(status) << "\n";
        return 1;
    }
    status = gemm_op.initialize(args, d_ws);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "initialize failed\n"; return 1;
    }
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "run failed\n"; return 1;
    }
    cudaDeviceSynchronize();

    // 验对
    cudaMemcpy(hD.data(), dD, M * N * sizeof(ElementC), cudaMemcpyDeviceToHost);
    reference_gemm(hA, K, hB, K, hRef, N, M, N, K, alpha, beta);

    float max_err = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float a = __bfloat162float(hD[i]);
        float b = __bfloat162float(hRef[i]);
        max_err = std::fmax(max_err, std::fabs(a - b));
    }
    bool pass = max_err < 1e-1f;
    std::cout << "[02_gemm_sm90] M=" << M << " N=" << N << " K=" << K
              << "  max_abs_err=" << max_err
              << "  " << (pass ? "PASS" : "FAIL") << "\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    if (d_ws) cudaFree(d_ws);
    return pass ? 0 : 1;
}
