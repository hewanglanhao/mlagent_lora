#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

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

}

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    check_inputs(W, X, A, B);

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    const int64_t d = Wc.size(0);

    // Allocate output Y and temporary U (row-major shape {16, d})
    auto Y = torch::empty({d, d}, Wc.options());
    auto temp = torch::empty({16, d}, Wc.options());

    // Get cuBLAS handle and bind to current CUDA stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    cublasStatus_t status;

    // Y = W @ X  (row-major equivalent via column-major X * W)
    status = cublasSgemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         d, d, d,
                         &alpha,
                         Xc.data_ptr<float>(), d,
                         Wc.data_ptr<float>(), d,
                         &beta0,
                         Y.data_ptr<float>(), d);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed for W@X");

    // temp = B.T @ X  (row-major B is d x 16, transposed to 16 x d)
    status = cublasSgemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_T,
                         d, 16, d,
                         &alpha,
                         Xc.data_ptr<float>(), d,
                         Bc.data_ptr<float>(), 16,
                         &beta0,
                         temp.data_ptr<float>(), d);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed for B.T@X");

    // Y += A @ temp  (accumulate with beta=1)
    status = cublasSgemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         d, d, 16,
                         &alpha,
                         temp.data_ptr<float>(), d,
                         Ac.data_ptr<float>(), 16,
                         &beta1,
                         Y.data_ptr<float>(), d);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed for A@temp");

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W @ X + A @ (B.T @ X)");
}