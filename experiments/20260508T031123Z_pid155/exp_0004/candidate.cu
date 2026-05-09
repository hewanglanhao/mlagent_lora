#include <torch/extension.h>

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

// Kernel: C = B.T @ X, where C is 16 x d, B is d x 16, X is d x d
// Each thread computes one element C[k][j] = sum_{l=0}^{d-1} B[l][k] * X[l][j]
__global__ void btx_kernel(const float* __restrict__ B,
                           const float* __restrict__ X,
                           float* __restrict__ C,
                           int d) {
    // thread index in 1D grid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 16 * d;
    if (tid >= total) return;

    // k in [0,15], j in [0, d-1]
    int k = tid % 16;
    int j = tid / 16;

    float sum = 0.0f;
    // loop over l
    for (int l = 0; l < d; ++l) {
        sum += B[l * 16 + k] * X[l * d + j];
    }
    C[k * d + j] = sum;
}

// Kernel: Y += A @ C, where A is d x 16, C is 16 x d, Y is d x d
// Each thread computes Y[i][j] += sum_{k=0}^{15} A[i][k] * C[k][j]
__global__ void ax_add_kernel(const float* __restrict__ A,
                              const float* __restrict__ C,
                              float* __restrict__ Y,
                              int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * d;
    if (tid >= total) return;

    int i = tid / d;
    int j = tid % d;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        sum += A[i * 16 + k] * C[k * d + j];
    }
    Y[i * d + j] += sum;
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Make tensors contiguous for pointer access
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    // Use ATen/cuBLAS for W @ X
    auto Y = at::matmul(Wc, Xc);

    int d = static_cast<int>(Wc.size(0));
    // Allocate temporary for C = B.T @ X (size: 16 x d)
    auto C = torch::empty({16, d}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int block_size = 512;

    // Launch kernel for C = B.T @ X
    int total_c = 16 * d;
    int grid_c = (total_c + block_size - 1) / block_size;
    btx_kernel<<<grid_c, block_size>>>(
        Bc.data_ptr<float>(),
        Xc.data_ptr<float>(),
        C.data_ptr<float>(),
        d
    );

    // Launch kernel for Y += A @ C
    int total_y = d * d;
    int grid_y = (total_y + block_size - 1) / block_size;
    ax_add_kernel<<<grid_y, block_size>>>(
        Ac.data_ptr<float>(),
        C.data_ptr<float>(),
        Y.data_ptr<float>(),
        d
    );

    // Optional: synchronize to catch errors (for safety, though not strictly needed)
    // cudaDeviceSynchronize() is called implicitly by PyTorch when returning.
    // But we can add a CUDA error check for robustness.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: W @ X + A @ (B.T @ X)");
}