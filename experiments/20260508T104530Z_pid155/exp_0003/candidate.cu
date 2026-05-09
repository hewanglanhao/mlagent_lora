#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

// Custom kernel: adds A @ (B.T @ X) to Y in-place.
// Each thread computes one output element (row, col).
// Unrolled over rank r=16.
__global__ void rank16_update_kernel(
    float* __restrict__ Y,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ X,
    int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * d;
    if (idx >= total) return;

    int row = idx / d;
    int col = idx % d;

    float acc = 0.0f;

    // Fully unroll the rank-16 loop
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        float aik = A[row * 16 + k];
        float sum_blx = 0.0f;
        for (int l = 0; l < d; ++l) {
            sum_blx += B[l * 16 + k] * X[l * d + col];
        }
        acc += aik * sum_blx;
    }

    Y[idx] += acc;
}

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B)
{
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors.");
    TORCH_CHECK(W.scalar_type() == at::kFloat &&
                X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat &&
                B.scalar_type() == at::kFloat,
                "All inputs must be float32 tensors.");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "All inputs must be rank-2 tensors.");

    int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d.");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16.");
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B)
{
    check_inputs(W, X, A, B);

    // Ensure contiguous memory layouts for safe pointer access
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = Wc.size(0);

    // Compute W @ X using cuBLAS (via ATen)
    auto Y = at::matmul(Wc, Xc);

    // Launch custom kernel to add the rank-16 update
    const int block_size = 128;
    int total_elements = d * d;
    int grid_size = (total_elements + block_size - 1) / block_size;

    rank16_update_kernel<<<grid_size, block_size>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Bc.data_ptr<float>(),
        Xc.data_ptr<float>(),
        static_cast<int>(d)
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "rank16_update_kernel launch failed: ", cudaGetErrorString(err));

    // Synchronize to catch asynchronous errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "rank16_update_kernel execution failed: ", cudaGetErrorString(err));

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W @ X + A @ (B.T @ X) with custom rank-16 update");
}