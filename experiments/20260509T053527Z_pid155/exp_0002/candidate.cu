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
    int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

} // namespace

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
    auto Y = torch::empty({d, d}, Xc.options());

    // cuBLAS handle and stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    // SGEMM 1: Y = W @ X  (row-major)
    // column-major: C = X_col (d x d) @ W_col (d x d)   -> C is d x d col-major
    // lda = d (from X), ldb = d (from W), ldc = d
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, d, d,
                &alpha,
                Xc.data_ptr<float>(), d,
                Wc.data_ptr<float>(), d,
                &beta0,
                Y.data_ptr<float>(), d);

    // SGEMM 2: U (d x 16 row-major) holds column-major intermediate 16 x d
    // column-major: U_col = X_col (d x d) @ (B_col)^T (d x 16)   -> U_col is d x 16
    // with opB=T, B is treated as transposed from its column-major 16 x d representation
    // ldb = 16 (leading dimension of column-major B which is 16 x d)
    auto U = torch::empty({d, 16}, Xc.options());
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                d, 16, d,
                &alpha,
                Xc.data_ptr<float>(), d,
                Bc.data_ptr<float>(), 16,
                &beta0,
                U.data_ptr<float>(), d);

    // SGEMM 3: Y += A @ U   (row-major, interpreting U as row-major 16 x d)
    // column-major: Y_col += U_col (d x 16) @ A_col (16 x d) -> Y_col is d x d
    // U is stored row-major d x 16, column-major is 16 x d; we want to treat it as d x 16
    // The call below uses opA=N on U, which interprets U as column-major d x 16 (matching the required multiplication).
    lda = d, ldb = 16 (A is d x 16 row-major -> column-major 16 x d, leading dim 16)
    */
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, d, 16,
                &alpha,
                U.data_ptr<float>(), d,
                Ac.data_ptr<float>(), 16,
                &beta1,
                Y.data_ptr<float>(), d);

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (pure cuBLAS three SGEMM)");
}