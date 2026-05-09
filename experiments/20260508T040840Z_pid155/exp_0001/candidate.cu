#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_DIM(x, dim) TORCH_CHECK(x.dim() == dim, #x " must be " #dim "-dimensional")

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    CHECK_CUDA(W); CHECK_CUDA(X); CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_FLOAT(W); CHECK_FLOAT(X); CHECK_FLOAT(A); CHECK_FLOAT(B);
    CHECK_DIM(W, 2); CHECK_DIM(X, 2); CHECK_DIM(A, 2); CHECK_DIM(B, 2);
    TORCH_CHECK(W.size(0) == W.size(1), "W must be square");
    TORCH_CHECK(X.size(0) == X.size(1) && X.size(0) == W.size(0), "X must be same size as W");
    TORCH_CHECK(A.size(0) == W.size(0) && A.size(1) == 16, "A must be d x 16");
    TORCH_CHECK(B.size(0) == W.size(0) && B.size(1) == 16, "B must be d x 16");
}

// Tile size parameters
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_L = 16;
constexpr int RANK = 16;

__global__ void fused_lora_kernel(
    const float* __restrict__ Y_input,   // W @ X result, shape (d, d)
    const float* __restrict__ A,         // d x 16
    const float* __restrict__ B,         // d x 16
    const float* __restrict__ X,         // d x d
    float* __restrict__ Y_output,
    int d) {

    // Block index and global tile coordinates
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row0 = by * TILE_M;
    int col0 = bx * TILE_N;

    // Linear thread index (0..255)
    int tid = threadIdx.x;
    int local_row = tid / TILE_N;
    int local_col = tid % TILE_N;

    int row_global = row0 + local_row;
    int col_global = col0 + local_col;

    // Shared memory for M (B^T @ X) tile: [RANK][TILE_N]
    __shared__ float s_M[RANK][TILE_N];

    // Initialize s_M to zero
    if (local_row < RANK) {
        s_M[local_row][local_col] = 0.0f;
    }
    __syncthreads();

    // Accumulate M over L dimension
    for (int l0 = 0; l0 < d; l0 += TILE_L) {
        int l_global = l0 + local_row;
        if (l_global < d && local_row < TILE_L) {
            float b_val = B[l_global * RANK + local_col];
            float x_val = X[l_global * d + col_global];
            float prod = b_val * x_val;

            // Write to shared memory for reduction across rows
            __shared__ float s_prod[TILE_L][RANK];
            s_prod[local_row][local_col] = prod;
            __syncthreads();

            // Reduce within each column group (same local_col)
            if (local_row == 0) {
                float sum = 0.0f;
                for (int r = 0; r < TILE_L && (l0 + r) < d; ++r) {
                    sum += s_prod[r][local_col];
                }
                s_M[local_col][local_col] += sum;  // k = local_col, column index = local_col
            }
            __syncthreads();
        }
    }

    // Compute LoRA contribution for the output tile and add to Y_input
    if (row_global < d && col_global < d) {
        float lo_sum = 0.0f;
        // Load A row (16 floats)
        float a_row[16];
        for (int k = 0; k < 16; ++k) {
            a_row[k] = A[row_global * 16 + k];
        }

        for (int k = 0; k < 16; ++k) {
            lo_sum += a_row[k] * s_M[k][local_col];
        }

        // Load W @ X value
        float y_val = Y_input[row_global * d + col_global];
        // Write final result
        Y_output[row_global * d + col_global] = y_val + lo_sum;
    }
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);
    W = W.contiguous();
    X = X.contiguous();
    A = A.contiguous();
    B = B.contiguous();

    int64_t d = W.size(0);

    // Compute W @ X using cuBLAS (ATen)
    torch::Tensor Y = at::matmul(W, X).contiguous();

    // Launch fused kernel for LoRA update (in-place add to Y)
    dim3 grid((d + TILE_N - 1) / TILE_N, (d + TILE_M - 1) / TILE_M);
    dim3 block(256); // TILE_M * TILE_N = 256

    fused_lora_kernel<<<grid, block>>>(
        Y.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        static_cast<int>(d)
    );

    // Synchronize to catch errors
    CUDA_CHECK(cudaGetLastError());

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused LoRA forward (W@X + A@(B.T@X))");
}