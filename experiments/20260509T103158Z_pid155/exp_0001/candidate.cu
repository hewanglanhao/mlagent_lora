#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

constexpr int RANK = 16;
constexpr int TILE_COLS = 128;
constexpr int TILE_ROWS = 8;
constexpr int THREADS = 256;

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

__global__ __launch_bounds__(THREADS)
void rank16_update_tile128x8_kernel(float* __restrict__ Y,
                                    const float* __restrict__ A,
                                    const float* __restrict__ Z,
                                    int d) {
    __shared__ float sZ[RANK * TILE_COLS];
    __shared__ float sA[TILE_ROWS * RANK];

    const int tid = threadIdx.x;
    const int col_base = blockIdx.x * TILE_COLS;
    const int row_base = blockIdx.y * TILE_ROWS;

    for (int idx = tid; idx < RANK * TILE_COLS; idx += THREADS) {
        const int k = idx / TILE_COLS;
        const int tc = idx - k * TILE_COLS;
        const int col = col_base + tc;
        sZ[idx] = (col < d) ? Z[k * d + col] : 0.0f;
    }

    for (int idx = tid; idx < TILE_ROWS * RANK; idx += THREADS) {
        const int tr = idx / RANK;
        const int k = idx - tr * RANK;
        const int row = row_base + tr;
        sA[idx] = (row < d) ? A[row * RANK + k] : 0.0f;
    }

    __syncthreads();

    const int tc = tid & (TILE_COLS - 1);
    const int row_pair = tid >> 7;
    const int col = col_base + tc;

    if (col >= d) {
        return;
    }

    const int lr0 = row_pair;
    const int lr1 = row_pair + 2;
    const int lr2 = row_pair + 4;
    const int lr3 = row_pair + 6;

    const int row0 = row_base + lr0;
    const int row1 = row_base + lr1;
    const int row2 = row_base + lr2;
    const int row3 = row_base + lr3;

    float acc0 = (row0 < d) ? Y[row0 * d + col] : 0.0f;
    float acc1 = (row1 < d) ? Y[row1 * d + col] : 0.0f;
    float acc2 = (row2 < d) ? Y[row2 * d + col] : 0.0f;
    float acc3 = (row3 < d) ? Y[row3 * d + col] : 0.0f;

#define DO_RANK_K(K)                                      \
    do {                                                  \
        const float z = sZ[(K) * TILE_COLS + tc];          \
        acc0 += sA[lr0 * RANK + (K)] * z;                 \
        acc1 += sA[lr1 * RANK + (K)] * z;                 \
        acc2 += sA[lr2 * RANK + (K)] * z;                 \
        acc3 += sA[lr3 * RANK + (K)] * z;                 \
    } while (0)

    DO_RANK_K(0);
    DO_RANK_K(1);
    DO_RANK_K(2);
    DO_RANK_K(3);
    DO_RANK_K(4);
    DO_RANK_K(5);
    DO_RANK_K(6);
    DO_RANK_K(7);
    DO_RANK_K(8);
    DO_RANK_K(9);
    DO_RANK_K(10);
    DO_RANK_K(11);
    DO_RANK_K(12);
    DO_RANK_K(13);
    DO_RANK_K(14);
    DO_RANK_K(15);

#undef DO_RANK_K

    if (row0 < d) Y[row0 * d + col] = acc0;
    if (row1 < d) Y[row1 * d + col] = acc1;
    if (row2 < d) Y[row2 * d + col] = acc2;
    if (row3 < d) Y[row3 * d + col] = acc3;
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    const c10::cuda::CUDAGuard device_guard(W.device());

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    auto Y = at::mm(Wc, Xc);

    auto Bt = Bc.transpose(0, 1).contiguous();
    auto Z = at::mm(Bt, Xc).contiguous();

    const int d = static_cast<int>(Wc.size(0));

    const dim3 block(THREADS);
    const dim3 grid((d + TILE_COLS - 1) / TILE_COLS,
                    (d + TILE_ROWS - 1) / TILE_ROWS);

    rank16_update_tile128x8_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Z.data_ptr<float>(),
        d);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}