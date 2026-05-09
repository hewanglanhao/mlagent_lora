#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

namespace {

const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t _status = (call);                                       \
        TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                          \
                    "cuBLAS error: ", cublas_status_string(_status));          \
    } while (0)

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.layout() == torch::kStrided && X.layout() == torch::kStrided &&
                A.layout() == torch::kStrided && B.layout() == torch::kStrided,
                "all inputs must be strided tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "all inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be rank-2 tensors");
    TORCH_CHECK(W.get_device() == X.get_device() &&
                W.get_device() == A.get_device() &&
                W.get_device() == B.get_device(),
                "all inputs must be on the same CUDA device");

    const int64_t d = W.size(0);
    TORCH_CHECK(d >= 3584 && d <= 4608, "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

}  // namespace

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

    const int d = static_cast<int>(Wc.size(0));
    constexpr int r = 16;

    auto Y = torch::empty({d, d}, Wc.options());
    auto U = torch::empty({d, r}, Wc.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUBLAS(cublasSetStream(handle, stream.stream()));

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    const float* Wp = Wc.data_ptr<float>();
    const float* Xp = Xc.data_ptr<float>();
    const float* Ap = Ac.data_ptr<float>();
    const float* Bp = Bc.data_ptr<float>();
    float* Yp = Y.data_ptr<float>();
    float* Up = U.data_ptr<float>();

    // Row-major Y = W @ X is column-major Y^T = X^T @ W^T.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             d, d, d,
                             &alpha,
                             Xp, d,
                             Wp, d,
                             &beta0,
                             Yp, d));

    // U is allocated as {d,16}; cuBLAS views it as column-major d x 16:
    // U_col = X^T @ B, the transpose of the logical B.T @ X.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             d, r, d,
                             &alpha,
                             Xp, d,
                             Bp, r,
                             &beta0,
                             Up, d));

    // Accumulate row-major A @ (B.T @ X) via column-major Y^T += U_col @ A^T.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             d, d, r,
                             &alpha,
                             Up, d,
                             Ap, r,
                             &beta1,
                             Yp, d));

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward pure cuBLAS three-SGEMM");
}