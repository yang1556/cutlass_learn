// =====================================================================
//  01_gemm_sm80.cu  ——  最小可运行的 SM80 (A100) CUTLASS 2.x GEMM 例子
//
//  目标：D = alpha * (A @ B) + beta * C
//        A: (M, K)  fp16  row-major
//        B: (K, N)  fp16  col-major   <—— 注意 B 取列主序，原因见下文
//        C/D: (M, N)  fp16  row-major
//
//  这一版用 cutlass::gemm::device::Gemm，是 CUTLASS 2.x 最高层封装：
//  你只描述"我想要一个什么形状的 GEMM、用什么精度、用什么 tile 大小"，
//  CUTLASS 帮你搭好整个 kernel，包括：
//     - 全局内存 ↔ shared memory 的搬运（cp.async）
//     - 软件流水（多 stage）
//     - shared memory 的 swizzle（避免 bank conflict）
//     - 调用 mma.sync.m16n8k16 累加
//     - epilogue（量化、scale、写回）
//
//  代码很短（不到 150 行），但每一行都对应一个值得讲的概念。注释会指
//  向"讲解第 N 章"，请配合对话里的讲解阅读。
// =====================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// ---- CUTLASS 头文件 -------------------------------------------------
//   device::Gemm  —— 最高层封装，host 端可以像调用一个普通函数一样调它
//   layout::*     —— 用 tag 类型区分 row-major / col-major
//   numeric_types —— cutlass::half_t 是对 __half 的薄包装
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

// ---- 1. 类型与形状定义 ----------------------------------------------
// 所有 using 都是编译期决定的——CUTLASS 是模板元编程的极致体现，
// 整个 kernel 是被 C++ 模板"组装"出来的，运行时几乎没有分支。
//
// 讲解点：抽象金字塔（讲解第 1 章）
//   Device   —— 整个 GEMM 一次调用（host）
//   Kernel   —— 启动一次 grid（一个 grid 算 D）
//   Threadblock —— 一个 CTA 算 D 的一个 ThreadblockShape tile
//   Warp     —— 一个 warp 算 CTA tile 内的一个 WarpShape tile
//   MMA      —— 一条 mma.sync 指令算 InstructionShape

using ElementA = cutlass::half_t;          // A 元素类型
using ElementB = cutlass::half_t;          // B 元素类型
using ElementC = cutlass::half_t;          // C/D 元素类型
using ElementAccumulator = float;          // 累加器精度（fp32）
using ElementCompute = float;              // 标量 alpha/beta 的精度

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;   // 见讲解第 5 章
using LayoutC = cutlass::layout::RowMajor;

// 讲解第 2 章：三个 GemmShape 的物理含义
//   ThreadblockShape <128,128,32>  ——  一个 CTA 算结果矩阵的 128×128，
//                                       每次沿 K 步进 32 累加
//   WarpShape        <64,64,32>    ——  一个 warp 算 CTA tile 内的 64×64
//                                       每个 CTA 有 (128/64)*(128/64) = 4 warps
//   InstructionShape <16,8,16>     ——  一条 mma.sync.m16n8k16 算 16×8 输出
//                                       一个 warp 一次发射 (64/16)*(64/8) = 32 条
using ThreadblockShape  = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape         = cutlass::gemm::GemmShape< 64,  64, 32>;
using InstructionShape  = cutlass::gemm::GemmShape< 16,   8, 16>;

// Tensor Core 路径标识（区别于 SIMT cuda core）
using OpClass = cutlass::arch::OpClassTensorOp;
using ArchTag = cutlass::arch::Sm80;

// EpilogueOp = "怎么把 fp32 累加器写回 fp16 输出 + 做 alpha/beta scaling"
//   D = alpha * acc + beta * C
// 模板参数 8 = 向量化宽度：一次写回 8 个 fp16 = 16 字节 = 一次 STG.E.128
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, /*VectorWidth=*/8, ElementAccumulator, ElementCompute>;

// 调度器（决定 (m_tile, n_tile) → CTA grid 坐标的映射；保持默认即可）
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// 讲解第 3 章：Stages = 软件流水深度
//   Stages=3 表示 SMEM 同时存 3 份 K-tile 的 A/B：
//   stage k 在做 mma.sync 时，stage k+1 在 cp.async 加载，stage k+2 已加载完
//   这是一个 producer-consumer 三级 pipeline，让全局内存延迟被算力盖住
constexpr int Stages = 3;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp, SwizzleThreadBlock,
    Stages>;

// ---- 2. 朴素 reference GEMM（host）用于验对 ------------------------
void reference_gemm(const std::vector<__half>& A, int lda_rm,
                    const std::vector<__half>& B, int ldb_cm,
                    std::vector<__half>& D, int ldd_rm,
                    int M, int N, int K, float alpha, float beta) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                // A row-major:  A[m,k] = A[m*lda + k]
                // B col-major:  B[k,n] = B[n*ldb + k]
                float a = __half2float(A[m * lda_rm + k]);
                float b = __half2float(B[n * ldb_cm + k]);
                acc += a * b;
            }
            float c_old = __half2float(D[m * ldd_rm + n]);
            D[m * ldd_rm + n] = __float2half(alpha * acc + beta * c_old);
        }
    }
}

// ---- 3. main：分配 / 调用 / 验对 ------------------------------------
int main() {
    int M = 512, N = 512, K = 512;
    float alpha = 1.0f, beta = 0.0f;

    // host 数据
    std::vector<__half> hA(M * K), hB(K * N), hD(M * N), hRef(M * N, __float2half(0.f));
    for (int i = 0; i < M * K; ++i) hA[i] = __float2half((rand() % 17 - 8) / 8.0f);
    for (int i = 0; i < K * N; ++i) hB[i] = __float2half((rand() % 17 - 8) / 8.0f);
    for (int i = 0; i < M * N; ++i) hD[i] = __float2half(0.f);

    // device 数据
    cutlass::half_t *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, M * K * sizeof(cutlass::half_t));
    cudaMalloc(&dB, K * N * sizeof(cutlass::half_t));
    cudaMalloc(&dC, M * N * sizeof(cutlass::half_t));
    cudaMalloc(&dD, M * N * sizeof(cutlass::half_t));
    cudaMemcpy(dA, hA.data(), M * K * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), K * N * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(cutlass::half_t));

    // 构造 Arguments：尺寸、指针、leading dimension、alpha/beta
    //   - lda = K（A row-major：每行有 K 个元素）
    //   - ldb = K（B col-major：每列有 K 个元素，列内连续）
    //   - ldc = N（C/D row-major：每行有 N 个元素）
    typename Gemm::Arguments args(
        {M, N, K},                                 // problem size
        {dA, K},                                   // ref_A: ptr + ld
        {dB, K},                                   // ref_B
        {dC, N},                                   // ref_C
        {dD, N},                                   // ref_D
        {alpha, beta}                              // epilogue scalars
    );

    // workspace 用于 split-K 等场景；这个例子不用 split-K，所以是 0
    Gemm gemm_op;
    size_t ws = gemm_op.get_workspace_size(args);
    void* d_ws = nullptr;
    if (ws) cudaMalloc(&d_ws, ws);

    // 三步走：can_implement → initialize → ()
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "can_implement failed: "
                  << cutlassGetStatusString(status) << "\n";
        return 1;
    }
    status = gemm_op.initialize(args, d_ws);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "initialize failed\n"; return 1;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "run failed\n"; return 1;
    }
    cudaDeviceSynchronize();

    // 拷回比对
    cudaMemcpy(hD.data(), dD, M * N * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);
    reference_gemm(hA, K, hB, K, hRef, N, M, N, K, alpha, beta);

    float max_err = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float a = __half2float(hD[i]);
        float b = __half2float(hRef[i]);
        max_err = std::fmax(max_err, std::fabs(a - b));
    }
    bool pass = max_err < 1e-1f;          // fp16 容差宽松
    std::cout << "[01_gemm_sm80] M=" << M << " N=" << N << " K=" << K
              << "  max_abs_err=" << max_err
              << "  " << (pass ? "PASS" : "FAIL") << "\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    if (d_ws) cudaFree(d_ws);
    return pass ? 0 : 1;
}
