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

constexpr int R = 16;

__global__ void btx_kernel(const float* __restrict__ B,
                           const float* __restrict__ X,
                           float* __restrict__ BTX,
                           int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = R * d;
    if (idx >= total) return;
    int k = idx / d;
    int j = idx % d;
    float sum = 0.0f;
    for (int l = 0; l < d; ++l) {
        sum += B[l * R + k] * X[l * d + j];
    }
    BTX[k * d + j] = sum;
}

__global__ void add_update_kernel(const float* __restrict__ A,
                                  const float* __restrict__ BTX,
                                  float* __restrict__ Y,
                                  int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * d;
    if (idx >= total) return;
    int i = idx / d;
    int j = idx % d;
    float update = 0.0f;
    for (int k = 0; k < R; ++k) {
        update += A[i * R + k] * BTX[k * d + j];
    }
    Y[i * d + j] += update;
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    int64_t d = Wc.size(0);

    // W @ X via cuBLAS
    auto Y = at::matmul(Wc, Xc);

    // Allocate temporary for B^T @ X (size 16 x d)
    auto BTX = at::empty({R, d}, Y.options());

    // Custom kernel: compute BTX = B^T @ X
    int64_t btx_elements = R * d;
    int block = 256;
    int grid = (btx_elements + block - 1) / block;
    btx_kernel<<<grid, block>>>(
        Bc.data_ptr<float>(), Xc.data_ptr<float>(),
        BTX.data_ptr<float>(), (int)d);

    // Custom kernel: Y += A @ BTX
    int64_t y_elements = d * d;
    grid = (y_elements + block - 1) / block;
    add_update_kernel<<<grid, block>>>(
        Ac.data_ptr<float>(), BTX.data_ptr<float>(),
        Y.data_ptr<float>(), (int)d);

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward with custom rank-16 update");
}