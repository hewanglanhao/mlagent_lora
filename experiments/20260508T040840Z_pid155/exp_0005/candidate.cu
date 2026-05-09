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
    TORCH_CHECK(d >= 3584 && d <= 4608, "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

// Fused kernel: Y += A @ tmp, where tmp is 16 x d (row-major)
// Y is d x d, A is d x 16, tmp is 16 x d.
// Each thread computes one element of the output.
__global__ void fused_rank16_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ tmp,
    float* __restrict__ Y,
    const int64_t d) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d * d) return;
    const int64_t row = idx / d;
    const int64_t col = idx % d;

    // Fully unrolled loop over rank 16
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        acc += A[row * 16 + k] * tmp[k * d + col];
    }
    Y[idx] += acc;
}

} // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    // W @ X via cuBLAS (ATen)
    auto Y = at::matmul(Wc, Xc);

    // Compute B.T @ X as a small 16 x d matrix
    auto Bt = Bc.transpose(0, 1).contiguous(); // 16 x d
    auto tmp = at::matmul(Bt, Xc);             // 16 x d

    const int64_t d = Wc.size(0);
    const int64_t total_elements = d * d;

    // Launch fused kernel with block size 256
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_rank16_add_kernel<<<grid_size, block_size>>>(
        Ac.data_ptr<float>(),
        tmp.data_ptr<float>(),
        Y.data_ptr<float>(),
        d);

    // Check CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (fused rank-16 update)");
}