#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int RANK = 16;
constexpr int BLOCK_SIZE = 128;

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat &&
                X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat &&
                B.scalar_type() == at::kFloat,
                "all inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be rank-2 tensors");
    TORCH_CHECK(W.device() == X.device() &&
                W.device() == A.device() &&
                W.device() == B.device(),
                "all inputs must be on the same CUDA device");

    const int64_t d = W.size(0);
    TORCH_CHECK(d >= 3584 && d <= 4608, "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d, "W must be d x d");
    TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == RANK, "A must be d x 16");
    TORCH_CHECK(B.size(0) == d && B.size(1) == RANK, "B must be d x 16");
}

__global__ void rank16_add_kernel(
    float* __restrict__ Y,
    const float* __restrict__ A,
    const float* __restrict__ T,
    int64_t d) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = d * d;
    if (idx >= total) {
        return;
    }

    int64_t row = idx / d;
    int64_t col = idx - row * d;

    const float* arow = A + row * RANK;
    const float* tcol = T + col;

    float acc = 0.0f;
    acc += arow[0]  * tcol[0 * d];
    acc += arow[1]  * tcol[1 * d];
    acc += arow[2]  * tcol[2 * d];
    acc += arow[3]  * tcol[3 * d];
    acc += arow[4]  * tcol[4 * d];
    acc += arow[5]  * tcol[5 * d];
    acc += arow[6]  * tcol[6 * d];
    acc += arow[7]  * tcol[7 * d];
    acc += arow[8]  * tcol[8 * d];
    acc += arow[9]  * tcol[9 * d];
    acc += arow[10] * tcol[10 * d];
    acc += arow[11] * tcol[11 * d];
    acc += arow[12] * tcol[12 * d];
    acc += arow[13] * tcol[13 * d];
    acc += arow[14] * tcol[14 * d];
    acc += arow[15] * tcol[15 * d];

    Y[idx] += acc;
}

} // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    auto Y = at::matmul(Wc, Xc);
    auto T = at::matmul(Bc.transpose(0, 1).contiguous(), Xc).contiguous();

    const int64_t d = Wc.size(0);
    const int64_t total = d * d;
    const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    rank16_add_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        T.data_ptr<float>(),
        d);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}