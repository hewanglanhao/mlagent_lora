#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

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

__global__ void rank16_add_scalar_kernel(float* __restrict__ y,
                                         const float* __restrict__ A,
                                         const float* __restrict__ T,
                                         int64_t d) {
    const int64_t total = d * d;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const int64_t row = idx / d;
        const int64_t col = idx - row * d;
        const float* a = A + row * 16;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc = fmaf(a[k], T[static_cast<int64_t>(k) * d + col], acc);
        }
        y[idx] += acc;
    }
}

void launch_rank16_add(torch::Tensor& y, const torch::Tensor& A, const torch::Tensor& T, int64_t d) {
    const int threads = 128;  // block size 128
    const int64_t total = d * d;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rank16_add_scalar_kernel<<<blocks, threads, 0, stream>>>(
        y.data_ptr<float>(),
        A.data_ptr<float>(),
        T.data_ptr<float>(),
        d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Ensure contiguous tensors for cuBLAS and custom kernel
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    const int64_t d = Wc.size(0);

    // Step 1: Y = W @ X using cuBLAS
    auto y = at::matmul(Wc, Xc);
    y = y.contiguous();  // ensure contiguous for custom kernel

    // Step 2: T = B^T @ X (shape 16 x d)
    auto T = at::matmul(Bc.t(), Xc);
    T = T.contiguous();

    // Step 3: Y += A @ T using custom scalar kernel
    launch_rank16_add(y, Ac, T, d);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W @ X + A @ (B^T @ X)");
}