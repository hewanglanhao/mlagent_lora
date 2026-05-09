#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors.");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "All inputs must be float32.");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "All inputs must be 2D tensors.");
    int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be square d x d.");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16.");
}

// Kernel: compute T = B^T @ X,  T is 16 x d
// Each thread handles one element (k, j) of T
__global__ void lowrank_btx_kernel(
    const float* __restrict__ B,   // d x 16 row-major
    const float* __restrict__ X,   // d x d row-major
    float* __restrict__ T,         // 16 x d row-major
    int64_t d) {
    int64_t total = 16 * d;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t k = idx / d;          // rank index 0..15
    int64_t j = idx % d;          // column index in X
    float sum = 0.0f;
    const float* B_col = B + k;   // B[k] column access: stride 16
    const float* X_col = X + j;   // X column: stride d
    for (int64_t i = 0; i < d; ++i) {
        sum += (*B_col) * (*X_col);
        B_col += 16;   // next row, same column k
        X_col += d;    // next row, same column j
    }
    T[k * d + j] = sum;
}

// Kernel: compute delta = A @ T, add result to Y
// Each thread handles one element (i, j) of Y
__global__ void lowrank_ax_add_kernel(
    const float* __restrict__ A,   // d x 16 row-major
    const float* __restrict__ T,   // 16 x d row-major
    float* __restrict__ Y,         // d x d row-major (in-place addition)
    int64_t d) {
    int64_t total = d * d;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t i = idx / d;
    int64_t j = idx % d;
    float sum = 0.0f;
    const float* A_row = A + i * 16;      // row i of A
    const float* T_col = T + j;           // column j of T (stride d)
    #pragma unroll 16
    for (int k = 0; k < 16; ++k) {
        sum += A_row[k] * T_col[k * d];
    }
    Y[i * d + j] += sum;
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Make contiguous copies
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = Wc.size(0);

    // Step 1: full GEMM via cuBLAS
    auto Y = at::matmul(Wc, Xc);  // d x d

    // Step 2: allocate T = B^T @ X  (16 x d)
    auto T = torch::empty({16, d}, Wc.options());

    const float* Bptr = Bc.data_ptr<float>();
    const float* Xptr = Xc.data_ptr<float>();
    float* Tptr = T.data_ptr<float>();
    float* Yptr = Y.data_ptr<float>();
    const float* Aptr = Ac.data_ptr<float>();

    const int block_size = 256;

    // Launch kernel for B^T @ X
    int64_t total_btx = 16 * d;
    dim3 grid_btx((total_btx + block_size - 1) / block_size);
    lowrank_btx_kernel<<<grid_btx, block_size>>>(Bptr, Xptr, Tptr, d);

    // Launch kernel for A @ T and add to Y
    int64_t total_ax = d * d;
    dim3 grid_ax((total_ax + block_size - 1) / block_size);
    lowrank_ax_add_kernel<<<grid_ax, block_size>>>(Aptr, Tptr, Yptr, d);

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W @ X + A @ (B.T @ X)");
}