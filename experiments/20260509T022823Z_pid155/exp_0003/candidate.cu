#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status__ = (call);                                      \
        TORCH_CHECK(status__ == CUBLAS_STATUS_SUCCESS,                         \
                    "cuBLAS error at ", __FILE__, ":", __LINE__,              \
                    " status=", static_cast<int>(status__));                   \
    } while (0)

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

    TORCH_CHECK(W.get_device() == X.get_device() &&
                W.get_device() == A.get_device() &&
                W.get_device() == B.get_device(),
                "all inputs must be on the same CUDA device");

    const int64_t d = W.size(0);
    TORCH_CHECK(d >= 3584 && d <= 4608,
                "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d,
                "W must be d x d");
    TORCH_CHECK(X.size(0) == d && X.size(1) == d,
                "X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16,
                "A must be d x 16");
    TORCH_CHECK(B.size(0) == d && B.size(1) == 16,
                "B must be d x 16");
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

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    CUBLAS_CHECK(cublasSetStream(handle, stream.stream()));

    const float* Wp = Wc.data_ptr<float>();
    const float* Xp = Xc.data_ptr<float>();
    const float* Ap = Ac.data_ptr<float>();
    const float* Bp = Bc.data_ptr<float>();
    float* Yp = Y.data_ptr<float>();
    float* Up = U.data_ptr<float>();

    /*
      Row-major tensor memory is interpreted by cuBLAS as column-major storage
      of the transpose.

      1) Y_rm = W_rm @ X_rm
         Y_cm = Y_rm^T = X_rm^T @ W_rm^T = X_cm @ W_cm.
    */
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             d, d, d,
                             &alpha,
                             Xp, d,
                             Wp, d,
                             &beta0,
                             Yp, d));

    /*
      2) U_rm = X_rm^T @ B_rm, shape d x 16.
         U storage as column-major is U_cm = U_rm^T, shape 16 x d.
         B storage as column-major is B_cm = B_rm^T, shape 16 x d.
         Therefore U_cm = B_cm @ X_cm^T = B_rm^T @ X_rm.
    */
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             r, d, d,
                             &alpha,
                             Bp, r,
                             Xp, d,
                             &beta0,
                             Up, r));

    /*
      3) Accumulate low-rank term:
         Y_rm += A_rm @ U_rm^T.
         In column-major view:
         Y_cm += U_rm @ A_rm^T = U_cm^T @ A_cm.
    */
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             d, d, r,
                             &alpha,
                             Up, r,
                             Ap, r,
                             &beta1,
                             Yp, d));

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}