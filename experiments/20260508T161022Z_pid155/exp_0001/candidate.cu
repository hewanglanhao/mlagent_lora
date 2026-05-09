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

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    const int64_t d = W.size(0);
    const int64_t r = 16;

    // Make contiguous to ensure pointer access is safe
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    // Output tensor
    auto Y = torch::empty({d, d}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Get cuBLAS handle and PyTorch CUDA stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one  = 1.0f;

    // Compute Y = W @ X  (main term)
    // Row-major: call cuBLAS with dimensions swapped per the standard pattern.
    // For row-major C = A * B (A: m x k, B: k x n, C: m x n):
    //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //               n, m, k, &alpha, B, n, A, k, &beta, C, n)
    // Here: A = W (d x d), B = X (d x d)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, d, d,                     // n, m, k
                &alpha,
                Xc.data_ptr<float>(), d,      // B, ldb = d
                Wc.data_ptr<float>(), d,      // A, lda = d
                &beta_zero,
                Y.data_ptr<float>(), d);      // C, ldc = d

    // Compute T = B.T @ X  (low‑rank intermediate)
    // B is d x r row‑major, X is d x d row‑major.
    // op(B) = T, op(X) = N.
    // Result T is r x d row‑major.
    // cuBLAS: C = alpha * op(A) * op(B) + beta * C.
    // Here op(A) = B^T (r x d), op(B) = X (d x d).
    auto T = torch::empty({r, d}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                d, r, d,                     // n, m, k   (n = cols of op(B) = d, m = rows of op(A) = r)
                &alpha,
                Bc.data_ptr<float>(), r,      // A (original B) lda = r (cols of B)
                Xc.data_ptr<float>(), d,      // B (original X) ldb = d
                &beta_zero,
                T.data_ptr<float>(), d);      // C, ldc = d (T has r rows, d cols)

    // Accumulate low‑rank term: Y = Y + A @ T  (beta = 1)
    // A is d x r, T is r x d.
    // Row‑major: call cuBLAS with order: C = A * T.
    // Use pattern: cublasSgemm(..., n, m, k, &alpha, B, n, A, k, &beta, C, n)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, d, r,                     // n, m, k
                &alpha,
                T.data_ptr<float>(), d,      // B (T) ldb = d
                Ac.data_ptr<float>(), r,      // A (A) lda = r
                &beta_one,
                Y.data_ptr<float>(), d);      // C (Y) ldc = d

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward using three cuBLAS SGEMMs");
}