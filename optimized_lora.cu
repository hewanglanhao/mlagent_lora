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

    // Block handles 16 rows and 128 columns.
    int row_tile = blockIdx.y;
    int col_tile = blockIdx.x;
    int row_start = row_tile * 16;
    int col_start = col_tile * 128;

    int tid = threadIdx.x;
    int row_in_block = tid / 16;   // 0..15
    int col_group = tid % 16;      // 0..15

    int global_row = row_start + row_in_block;
    int global_col_base = col_start + col_group * 8;

    __shared__ float As[16][16];
    __shared__ float Zs[16][128];

    // Load A tile (16x16): each thread loads one element
    int a_row = row_start + (tid / 16);
    int a_col = tid % 16;
    if (a_row < d && a_col < 16) {
        As[a_row - row_start][a_col] = A[a_row * 16 + a_col];
    } else {
        As[a_row - row_start][a_col] = 0.0f;
    }

    // Load Z tile (16x128): each thread loads 8 floats (2 float4)
    for (int i = 0; i < 8; ++i) {
        int linear = tid * 8 + i;
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

        // Add to Y (read-modify-write) with vectorized float4 stores
        float* y_row = Y + global_row * d;
        for (int i = 0; i < 8; i += 4) {
            int col = global_col_base + i;
            if (col < d) {
                if (col + 3 < d) {
                    float4 old = *reinterpret_cast<const float4*>(&y_row[col]);
                    old.x += acc[i];
                    old.y += acc[i+1];
                    old.z += acc[i+2];
                    old.w += acc[i+3];
                    *reinterpret_cast<float4*>(&y_row[col]) = old;
                } else {
                    // scalar tail
                    for (int j = i; j < 8 && (global_col_base + j) < d; ++j) {
                        y_row[global_col_base + j] += acc[j];
                    }
                }
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

    // Step 1: Y = W @ X (cuBLAS via ATen)
    auto Y = at::matmul(Wc, Xc);

    // Step 2: Z = B.T @ X (cuBLAS via ATen)
    auto Z = at::matmul(Bc.transpose(0, 1).contiguous(), Xc);

    // Ensure contiguous output buffers
    Y = Y.contiguous();
    Z = Z.contiguous();

    int64_t d = Wc.size(0);

    // Launch fused rank-16 update kernel
    dim3 block(256);
    dim3 grid((d + 127) / 128, (d + 15) / 16);

    auto stream = at::cuda::getCurrentCUDAStream();
    fused_rank16_update_kernel<<<grid, block, 0, stream>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Z.data_ptr<float>(),
        static_cast<int>(d));

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (fused rank-16 update)");
}