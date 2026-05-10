#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int RANK = 16;
constexpr int COL_TILE = 128;
constexpr int ROWS_PER_BLOCK = 4;

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
    TORCH_CHECK(d >= 3584 && d <= 4608,
                "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d,
                "W must be d x d");
    TORCH_CHECK(X.size(0) == d && X.size(1) == d,
                "X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == RANK,
                "A must be d x 16");
    TORCH_CHECK(B.size(0) == d && B.size(1) == RANK,
                "B must be d x 16");
}

__global__ void rank16_update_b128_kernel(float* __restrict__ Y,
                                           const float* __restrict__ A,
                                           const float* __restrict__ Z,
                                           int d) {
    __shared__ float sA[ROWS_PER_BLOCK][RANK];
    __shared__ float sZ[RANK][COL_TILE];

    const int tx = threadIdx.x;
    const int col = blockIdx.x * COL_TILE + tx;
    const int row_base = blockIdx.y * ROWS_PER_BLOCK;

    if (tx < RANK) {
        #pragma unroll
        for (int rr = 0; rr < ROWS_PER_BLOCK; ++rr) {
            const int row = row_base + rr;
            sA[rr][tx] = (row < d) ? A[row * RANK + tx] : 0.0f;
        }
    }

    if (tx < COL_TILE) {
        #pragma unroll
        for (int k = 0; k < RANK; ++k) {
            sZ[k][tx] = (col < d) ? Z[k * d + col] : 0.0f;
        }
    }

    __syncthreads();

    if (col < d) {
        #pragma unroll
        for (int rr = 0; rr < ROWS_PER_BLOCK; ++rr) {
            const int row = row_base + rr;
            if (row < d) {
                float acc = Y[row * d + col];

                acc = __fmaf_rn(sA[rr][0],  sZ[0][tx],  acc);
                acc = __fmaf_rn(sA[rr][1],  sZ[1][tx],  acc);
                acc = __fmaf_rn(sA[rr][2],  sZ[2][tx],  acc);
                acc = __fmaf_rn(sA[rr][3],  sZ[3][tx],  acc);
                acc = __fmaf_rn(sA[rr][4],  sZ[4][tx],  acc);
                acc = __fmaf_rn(sA[rr][5],  sZ[5][tx],  acc);
                acc = __fmaf_rn(sA[rr][6],  sZ[6][tx],  acc);
                acc = __fmaf_rn(sA[rr][7],  sZ[7][tx],  acc);
                acc = __fmaf_rn(sA[rr][8],  sZ[8][tx],  acc);
                acc = __fmaf_rn(sA[rr][9],  sZ[9][tx],  acc);
                acc = __fmaf_rn(sA[rr][10], sZ[10][tx], acc);
                acc = __fmaf_rn(sA[rr][11], sZ[11][tx], acc);
                acc = __fmaf_rn(sA[rr][12], sZ[12][tx], acc);
                acc = __fmaf_rn(sA[rr][13], sZ[13][tx], acc);
                acc = __fmaf_rn(sA[rr][14], sZ[14][tx], acc);
                acc = __fmaf_rn(sA[rr][15], sZ[15][tx], acc);

                Y[row * d + col] = acc;
            }
        }
    }
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    auto Y = at::matmul(Wc, Xc).contiguous();
    auto Z = at::matmul(Bc.transpose(0, 1).contiguous(), Xc).contiguous();

    const int d = static_cast<int>(Wc.size(0));

    const dim3 block(COL_TILE);
    const dim3 grid((d + COL_TILE - 1) / COL_TILE,
                    (d + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    rank16_update_b128_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Z.data_ptr<float>(),
        d
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}