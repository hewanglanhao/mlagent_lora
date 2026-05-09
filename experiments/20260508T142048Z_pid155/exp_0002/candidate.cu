#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "all inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be rank-2 tensors");
    const int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

__global__ void fused_rank16_update_kernel(
    float* __restrict__ Y,
    const float* __restrict__ A,
    const float* __restrict__ Z,
    int d) {

    // Block handles 8 rows and 128 columns.
    int row_tile = blockIdx.y;
    int col_tile = blockIdx.x;
    int row_start = row_tile * 8;
    int col_start = col_tile * 128;

    int tid = threadIdx.x;
    int row_in_block = tid / 16;   // 0..7
    int col_group = tid % 16;      // 0..15

    int global_row = row_start + row_in_block;
    int global_col_base = col_start + col_group * 8;

    __shared__ float As[8][16];
    __shared__ float Zs[16][128];

    // Load A tile (8x16): each thread loads one element
    int a_row = row_start + (tid / 16);
    int a_col = tid % 16;
    if (a_row < d && a_col < 16) {
        As[a_row - row_start][a_col] = A[a_row * 16 + a_col];
    } else {
        As[a_row - row_start][a_col] = 0.0f;
    }

    // Load Z tile (16x128): each thread loads 16 floats (scalar)
    for (int i = 0; i < 16; ++i) {
        int linear = tid * 16 + i;
        int k = linear / 128;
        int c = linear % 128;
        int global_c = col_start + c;
        if (global_c < d && k < 16) {
            Zs[k][c] = Z[k * d + global_c];
        } else {
            Zs[k][c] = 0.0f;
        }
    }

    __syncthreads();

    // Compute rank-16 update for this thread's row and columns
    if (global_row < d) {
        float acc[8] = {0.0f};
        int c_base = col_group * 8;

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float av = As[row_in_block][k];
            acc[0] += av * Zs[k][c_base + 0];
            acc[1] += av * Zs[k][c_base + 1];
            acc[2] += av * Zs[k][c_base + 2];
            acc[3] += av * Zs[k][c_base + 3];
            acc[4] += av * Zs[k][c_base + 4];
            acc[5] += av * Zs[k][c_base + 5];
            acc[6] += av * Zs[k][c_base + 6];
            acc[7] += av * Zs[k][c_base + 7];
        }

        // Scalar add to Y (read-modify-write)
        float* y_row = Y + global_row * d;
        for (int i = 0; i < 8; ++i) {
            int col = global_col_base + i;
            if (col < d) {
                y_row[col] += acc[i];
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

    // Ensure contiguous tensors for pointer access and cuBLAS
    W = W.contiguous();
    X = X.contiguous();
    A = A.contiguous();
    B = B.contiguous();

    const int64_t d = W.size(0);

    // Step 1: Y = W @ X using cuBLAS via ATen
    auto Y = torch::mm(W, X);

    // Step 2: Z = B.T @ X using cuBLAS via ATen
    auto Z = torch::mm(B.t(), X);

    // Step 3: Fused rank-16 update Y += A @ Z
    const int rows_per_block = 8;
    const int cols_per_block = 128;
    dim3 grid((d + cols_per_block - 1) / cols_per_block,
              (d + rows_per_block - 1) / rows_per_block);
    dim3 block(128);

    fused_rank16_update_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(),
        A.data_ptr<float>(),
        Z.data_ptr<float>(),
        static_cast<int>(d)
    );

    AT_CUDA_CHECK(cudaGetLastError());

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W @ X + A @ (B.T @ X)");
}