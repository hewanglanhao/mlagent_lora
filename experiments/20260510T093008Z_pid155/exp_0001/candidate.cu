#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <limits>

namespace {

void check_cublas(cublasStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg, " failed with cuBLAS status ", static_cast<int>(status));
}

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.device() == X.device() && W.device() == A.device() && W.device() == B.device(),
                "all inputs must be on the same CUDA device");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "all inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be rank-2 tensors");
    TORCH_CHECK(W.is_contiguous() && X.is_contiguous() && A.is_contiguous() && B.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t d = W.size(0);
    TORCH_CHECK(d >= 3584 && d <= 4608, "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d, "W must be d x d");
    TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16, "A must be d x 16");
    TORCH_CHECK(B.size(0) == d && B.size(1) == 16, "B must be d x 16");
    TORCH_CHECK(d <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                "d exceeds cuBLAS int range");
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    const c10::cuda::CUDAGuard device_guard(W.device());
    const int d = static_cast<int>(W.size(0));
    constexpr int r = 16;

    auto Y = torch::empty_like(W);
    auto U = torch::empty({static_cast<int64_t>(d), static_cast<int64_t>(r)}, W.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    const float* Wp = W.data_ptr<float>();
    const float* Xp = X.data_ptr<float>();
    const float* Ap = A.data_ptr<float>();
    const float* Bp = B.data_ptr<float>();
    float* Yp = Y.data_ptr<float>();
    float* Up = U.data_ptr<float>();

    // Y_col = X_col * W_col
    check_cublas(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    d, d, d,
                    &alpha,
                    Xp, d,
                    Wp, d,
                    &beta0,
                    Yp, d),
        "SGEMM1");

    // U_col[d,16] = X_col * B_col^T
    check_cublas(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    d, r, d,
                    &alpha,
                    Xp, d,
                    Bp, r,
                    &beta0,
                    Up, d),
        "SGEMM2");

    // Y_col += U_col * A_col
    check_cublas(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    d, d, r,
                    &alpha,
                    Up, d,
                    Ap, r,
                    &beta1,
                    Yp, d),
        "SGEMM3");

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}