#include <torch/extension.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int RANK = 16;
constexpr int TILE_COLS = 128;
constexpr int TILE_ROWS = 2;
constexpr int BLOCK_SIZE = 256;

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.device() == X.device() && W.device() == A.device() && W.device() == B.device(),
                "all inputs must be on the same CUDA device");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "all inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be rank-2 tensors");

    const int64_t d = W.size(0);
    TORCH_CHECK(d >= 3584 && d <= 4608, "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d, "W must be d x d");
    TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == RANK, "A must be d x 16");
    TORCH_CHECK(B.size(0) == d && B.size(1) == RANK, "B must be d x 16");
}

__device__ __forceinline__ float tf32_round_bits(float v) {
    unsigned int x = __float_as_uint(v);
    unsigned int exp = x & 0x7f800000u;
    if (exp == 0x7f800000u) {
        return v;
    }
    unsigned int lsb = (x >> 13) & 1u;
    x += 0x00000fffu + lsb;
    x &= 0xffffe000u;
    return __uint_as_float(x);
}

__device__ __forceinline__ float maybe_tf32(float v, bool use_tf32) {
    return use_tf32 ? tf32_round_bits(v) : v;
}

__global__ __launch_bounds__(BLOCK_SIZE)
void fused_rank16_update_kernel(float* __restrict__ Y,
                                const float* __restrict__ A,
                                const float* __restrict__ Z,
                                int d,
                                bool use_tf32) {
    __shared__ float As[TILE_ROWS][RANK];
    __shared__ float Zs[RANK][TILE_COLS];

    const int tid = threadIdx.x;
    const int col_base = blockIdx.x * TILE_COLS;
    const int row_base = blockIdx.y * TILE_ROWS;

    if (tid < TILE_ROWS * RANK) {
        const int lr = tid / RANK;
        const int k = tid - lr * RANK;
        const int row = row_base + lr;
        As[lr][k] = (row < d) ? maybe_tf32(A[row * RANK + k], use_tf32) : 0.0f;
    }

    for (int idx = tid; idx < RANK * TILE_COLS; idx += BLOCK_SIZE) {
        const int k = idx / TILE_COLS;
        const int lc = idx - k * TILE_COLS;
        const int col = col_base + lc;
        Zs[k][lc] = (col < d) ? maybe_tf32(Z[k * d + col], use_tf32) : 0.0f;
    }

    __syncthreads();

    const int lr = tid / TILE_COLS;
    const int lc = tid - lr * TILE_COLS;
    const int row = row_base + lr;
    const int col = col_base + lc;

    if (lr < TILE_ROWS && row < d && col < d) {
        float acc = 0.0f;

        const float a0  = As[lr][0];
        const float a1  = As[lr][1];
        const float a2  = As[lr][2];
        const float a3  = As[lr][3];
        const float a4  = As[lr][4];
        const float a5  = As[lr][5];
        const float a6  = As[lr][6];
        const float a7  = As[lr][7];
        const float a8  = As[lr][8];
        const float a9  = As[lr][9];
        const float a10 = As[lr][10];
        const float a11 = As[lr][11];
        const float a12 = As[lr][12];
        const float a13 = As[lr][13];
        const float a14 = As[lr][14];
        const float a15 = As[lr][15];

        acc = fmaf(a0,  Zs[0][lc],  acc);
        acc = fmaf(a1,  Zs[1][lc],  acc);
        acc = fmaf(a2,  Zs[2][lc],  acc);
        acc = fmaf(a3,  Zs[3][lc],  acc);
        acc = fmaf(a4,  Zs[4][lc],  acc);
        acc = fmaf(a5,  Zs[5][lc],  acc);
        acc = fmaf(a6,  Zs[6][lc],  acc);
        acc = fmaf(a7,  Zs[7][lc],  acc);
        acc = fmaf(a8,  Zs[8][lc],  acc);
        acc = fmaf(a9,  Zs[9][lc],  acc);
        acc = fmaf(a10, Zs[10][lc], acc);
        acc = fmaf(a11, Zs[11][lc], acc);
        acc = fmaf(a12, Zs[12][lc], acc);
        acc = fmaf(a13, Zs[13][lc], acc);
        acc = fmaf(a14, Zs[14][lc], acc);
        acc = fmaf(a15, Zs[15][lc], acc);

        Y[row * d + col] += acc;
    }
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    c10::cuda::CUDAGuard device_guard(W.device());

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    auto Y = at::matmul(Wc, Xc);
    auto Z = at::matmul(Bc.transpose(0, 1).contiguous(), Xc).contiguous();

    const int d = static_cast<int>(Wc.size(0));
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((d + TILE_COLS - 1) / TILE_COLS,
                    (d + TILE_ROWS - 1) / TILE_ROWS);

    const bool use_tf32 = at::globalContext().allowTF32CuBLAS();

    fused_rank16_update_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Z.data_ptr<float>(),
        d,
        use_tf32);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}