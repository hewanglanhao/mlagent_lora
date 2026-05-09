#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

namespace {

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors.");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "All inputs must be float32.");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "All inputs must be rank-2.");
    int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d.");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16.");
    TORCH_CHECK(d >= 3584 && d <= 4608,
                "d is outside [3584, 4608].");
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

    // Output tensor Y = d x d, row-major
    auto Y = torch::empty({d, d}, torch::TensorOptions()
                              .device(torch::kCUDA)
                              .dtype(torch::kFloat32));

    // Get cuBLAS handle and bind to current stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    // SGEMM 1: Y = W @ X
    // Row-major multiplication: C = A * B, A is W (d x d), B is X (d x d)
    // cuBLAS call: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, alpha, B, N, A, K, beta, C, M)
    // Here m = d, n = d, k = d
    // A = W, B = X
    // Result is row-major Y
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                d, d, d,
                &alpha,
                Xc.data_ptr<float>(), d,      // B = X, ldb = d
                Wc.data_ptr<float>(), d,      // A = W, lda = d
                &beta0,
                Y.data_ptr<float>(), d);      // C = Y, ldc = d

    // SGEMM 2: compute U = X * B^T, shape (d,16) row-major
    auto U = torch::empty({d, 16}, torch::TensorOptions()
                                .device(torch::kCUDA)
                                .dtype(torch::kFloat32));
    // Row-major: C = A * B, A is X (d x d), B is B^T (d x 16)
    // cuBLAS: m=d, k=d, n=16
    // call with A = X, B = Bc
    // Result U row-major (d x 16)
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                16, d, d,
                &alpha,
                Bc.data_ptr<float>(), 16,   // B = Bc, ldb = 16
                Xc.data_ptr<float>(), d,    // A = X, lda = d
                &beta0,
                U.data_ptr<float>(), d);   // C = U, ldc = d

    // SGEMM 3: accumulate Y += U * A^T, i.e., Y = Y + U * A^T
    // Row-major: C = A * B, A is U (d x 16), B is A^T (16 x d)
    // cuBLAS: m=d, k=16, n=d
    // Result added to Y with beta=1
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                d, d, 16,
                &alpha,
                Ac.data_ptr<float>(), 16,   // B = A (d x 16 row-major), ldb = 16
                U.data_ptr<float>(), d,     // A = U (d x 16 row-major), lda = d
                &beta1,
                Y.data_ptr<float>(), d);    // C = Y, ldc = d

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward using three SGEMMs");
}