#include <torch/extension.h>
#include <cuda_runtime.h>

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

__global__ void rank16_add_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ t,
    int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d || col >= d) return;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        sum += A[row * 16 + k] * t[k * d + col];
    }
    y[row * d + col] += sum;
}

void launch_rank16_add(
    torch::Tensor y,
    const torch::Tensor& A,
    const torch::Tensor& t,
    int64_t d) {
    dim3 block(16, 16);
    dim3 grid((d + 15) / 16, (d + 15) / 16);
    rank16_add_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(),
        A.data_ptr<float>(),
        t.data_ptr<float>(),
        static_cast<int>(d));
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

    // W @ X via cuBLAS
    auto y = at::matmul(Wc, Xc);
    y = y.contiguous();  // ensure contiguous layout for custom kernel

    // B.T @ X via cuBLAS -> 16 x d
    auto Bt = Bc.transpose(0, 1).contiguous();
    auto t = at::matmul(Bt, Xc);
    t = t.contiguous();

    // Fused rank-16 update: y += A @ t
    launch_rank16_add(y, Ac, t, Wc.size(0));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}