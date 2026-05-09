#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
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
                "all inputs must be 2D tensors");
    int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W, torch::Tensor X,
                      torch::Tensor A, torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Ensure contiguous (no-op if already contiguous)
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = Wc.size(0);
    const int r = 16;
    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    // Allocate output Y (row-major d x d) and temporary U (row-major d x r)
    auto Y = torch::empty({d, d}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto U = torch::empty({d, r}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Get cuBLAS handle and bind to the current PyTorch CUDA stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    // SGEMM1: compute Y = W @ X (row-major) using column-major view
    // In column-major:  C = A * B  => C = X_cm * W_cm = X^T * W^T = (W*X)^T
    // Writing into row-major Y automatically transposes to yield W*X.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,   // no transpose (column-major view)
                d, d, d,                    // m, n, k
                &alpha,
                Xc.data_ptr<float>(), d,     // A = X_cm (X as column-major)
                Wc.data_ptr<float>(), d,     // B = W_cm (W as column-major)
                &beta0,
                Y.data_ptr<float>(), d);    // C = Y (output as column-major)

    // SGEMM2: compute U = X_cm * B_cm = X^T * B = (B^T X)^T  (shape d x r)
    // U is row-major {d, r}, but we write column-major output with ldc=d.
    // The memory layout of U after this write is exactly the column-major
    // representation of a (d x r) matrix, which matches the row-major
    // representation of (r x d) with leading dimension d.  Later we will
    // read U as column-major with leading dimension r (since U_rm (d,r)
    // has column-major view shape (r,d) with leading dimension r).
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, r, d,                    // m=d, n=16, k=d
                &alpha,
                Xc.data_ptr<float>(), d,     // A = X_cm
                Bc.data_ptr<float>(), d,     // B = B_cm (column-major view of B)
                &beta0,
                U.data_ptr<float>(), d);    // C = U (ldc = d)

    // SGEMM3: accumulate low-rank term into Y: Y += A @ (B^T X)
    // This is accomplished by computing Y += U_cm * A_cm, where
    // U_cm is column-major view of U_rm: shape (r, d), leading dimension r.
    // A_cm is column-major view of A_rm: shape (r, d)? Wait: A_rm is (d, r),
    // so A_cm is (r, d) with leading dimension r.
    // The cuBLAS call: C = alpha * op(A) * op(B) + beta * C
    // We set op(A) = N for U_cm (k x n = r x d), op(B) = N for A_cm (k x m? actually op(B) is second matrix).
    // Let's set: m=d, n=d, k=r.
    // First matrix (A) is U, size k x n = r x d, leading dimension r.
    // Second matrix (B) is A, size k x m = r x d, leading dimension r.
    // C = Y (d x d), leading dimension d.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, d, r,                    // m=d, n=d, k=16
                &alpha,
                U.data_ptr<float>(), r,      // A = U_cm (lda = r)
                Ac.data_ptr<float>(), r,     // B = A_cm (ldb = r)
                &beta1,
                Y.data_ptr<float>(), d);    // C = Y (beta=1 accumulates)

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (W@X + A@B^T@X) via pure cuBLAS");
}