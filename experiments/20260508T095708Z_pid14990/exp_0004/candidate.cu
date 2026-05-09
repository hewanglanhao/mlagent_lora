#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

// Input validation
void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "All inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "All inputs must be rank-2 tensors");
    const int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

// Custom kernel: Y += A @ T, where A is d x 16, T is 16 x d, Y is d x d.
// Each thread computes one output element using a grid-stride loop.
__global__ void rank16_add_kernel(float* __restrict__ Y,
                                  const float* __restrict__ A,
                                  const float* __restrict__ T,
                                  int d) {
    int total = d * d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        int row = i / d;
        int col = i % d;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += A[row * 16 + k] * T[k * d + col];
        }
        Y[i] += sum;
    }
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Ensure contiguous memory for pointer access
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = Wc.size(0);

    // Step 1: Y = W @ X using cuBLAS (via ATen)
    auto Y = at::matmul(Wc, Xc);

    // Step 2: T = B.T @ X (16 x d) using cuBLAS
    auto T = at::matmul(Bc.t().contiguous(), Xc);  // 16 x d

    // Step 3: Y += A @ T using custom rank-16 kernel
    const int block_size = 512;
    int total_elements = d * d;
    int grid_size = (total_elements + block_size - 1) / block_size;
    // Cap grid size to a reasonable maximum (e.g., 65535) to avoid launch failure
    if (grid_size > 65535) grid_size = 65535;

    rank16_add_kernel<<<grid_size, block_size, 0, 0>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        T.data_ptr<float>(),
        static_cast<int>(d));

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("rank16_add_kernel launch failed: ") + cudaGetErrorString(err));
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W @ X + A @ (B.T @ X)");
}