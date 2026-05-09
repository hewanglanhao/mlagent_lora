#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

namespace {

#define CHECK_CUBLAS(expr)                                                       \
    do {                                                                         \
        cublasStatus_t _status = (expr);                                         \
        TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                            \
                    "cuBLAS error at ", __FILE__, ":", __LINE__,                 \
                    " status=", static_cast<int>(_status));                      \
    } while (0)

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.device() == X.device() && W.device() == A.device() && W.device() == B.device(),
                "all inputs must be on the same CUDA device");
    TORCH_CHECK(W.scalar_type() == at::kFloat &&
                X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat &&
                B.scalar_type() == at::kFloat,
                "all inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be rank-2 tensors");

    const int64_t d = W.size(0);
    TORCH_CHECK(d >= 3584 && d <= 4608,
                "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d,
                "W must be d x d");
    TORCH_CHECK(X.size(0) == d && X.size(1) == d,
                "X must be d x d matching W");
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

    const c10::cuda::CUDAGuard device_guard(W.device());

    torch::Tensor Wc = W.contiguous();
    torch::Tensor Xc = X.contiguous();
    torch::Tensor Ac = A.contiguous();
    torch::Tensor Bc = B.contiguous();

    const int d = static_cast<int>(Wc.size(0));
    constexpr int r = 16;

    torch::Tensor Y = torch::empty({d, d}, Wc.options());
    torch::Tensor U = torch::empty({d, r}, Wc.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    const float* Wp = Wc.data_ptr<float>();
    const float* Xp = Xc.data_ptr<float>();
    const float* Ap = Ac.data_ptr<float>();
    const float* Bp = Bc.data_ptr<float>();
    float* Yp = Y.data_ptr<float>();
    float* Up = U.data_ptr<float>();

    // Row-major C = W @ X is equivalent to column-major C^T = X^T_view @ W^T_view.
    // A row-major dxd tensor is viewed by cuBLAS as a column-major dxd transpose.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             d, d, d,
                             &alpha,
                             Xp, d,
                             Wp, d,
                             &beta0,
                             Yp, d));

    // U is allocated row-major {d,16}; cuBLAS views it as column-major {16,d}.
    // Fill U_view = B^T @ X without materializing B.T:
    // B row-major {d,16} is viewed as column-major {16,d} = B^T.
    // X row-major view is X^T, so op(X_view)=T gives X.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             r, d, d,
                             &alpha,
                             Bp, r,
                             Xp, d,
                             &beta0,
                             Up, r));

    // Accumulate low-rank term Y += A @ (B.T @ X).
    // In column-major view this is Y_view += op(U_view)=U_view^T @ A_view,
    // where U_view^T is row-major U = (B.T @ X)^T and A_view = A^T.
    CHECK_CUBLAS(cublasSgemm(handle,
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