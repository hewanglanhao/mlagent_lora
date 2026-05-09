#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

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
    const int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
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

    auto Y = at::matmul(W, X);
    auto BT = B.transpose(0, 1).contiguous();
    auto M = at::matmul(BT, X);
    Y.add_(at::matmul(A, M));
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (W@X + A@(B.T@X))");
}