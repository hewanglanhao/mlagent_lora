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
                "All inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "All inputs must be float32 tensors");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "All inputs must be rank-2 tensors");
    int64_t d = W.size(0);
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

    // Ensure contiguous and get dimensions
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = Wc.size(0);

    // Allocate output Y with shape (d, d), initially uninitialized (we set beta=0 in SGEMM1)
    auto Y = torch::empty({d, d}, torch::TensorOptions().dtype(torch::kFloat32).device(Wc.device()));

    // Get cuBLAS handle and bind to current PyTorch stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    // Pointer to raw float data
    const float* pW = Wc.data_ptr<float>();
    const float* pX = Xc.data_ptr<float>();
    const float* pA = Ac.data_ptr<float>();
    const float* pB = Bc.data_ptr<float>();
    float* pY = Y.data_ptr<float>();

    // SGEMM 1: Y = W @ X   (row-major)
    // cuBLAS column-major view: C_op = X * W  => C == Y
    cublasStatus_t stat;
    stat = cublasSgemm(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,  // op(A)=N, op(B)=N
                       d, d, d,                    // m=n=k=d
                       &alpha,
                       pX, d,                      // A = X, lda = d
                       pW, d,                      // B = W, ldb = d
                       &beta0,
                       pY, d);                     // C = Y, ldc = d
    TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cublasSgemm1 failed");

    // SGEMM 2: U = X @ B.T   (row-major intermediate, size [d,16])
    // cuBLAS column-major: U_op = (B) * X  => we set op(A)=N, op(B)=T, A=X, B=B
    auto U = torch::empty({d, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(Wc.device()));
    float* pU = U.data_ptr<float>();
    stat = cublasSgemm(handle,
                       CUBLAS_OP_N, CUBLAS_OP_T,  // op(A)=N, op(B)=T
                       d, 16, d,                    // m=d, n=16, k=d
                       &alpha,
                       pX, d,                      // A = X, lda = d
                       pB, 16,                     // B = B, ldb = 16 (B is d x 16)
                       &beta0,
                       pU, d);                     // C = U, ldc = d
    TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cublasSgemm2 failed");

    // SGEMM 3: Y = Y + A @ U   (accumulate into Y)
    // cuBLAS column-major: Y += (U) * A  => op(A)=N, op(B)=N, A=U, B=A
    stat = cublasSgemm(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,  // op(A)=N, op(B)=N
                       d, d, 16,                    // m=d, n=d, k=16
                       &alpha,
                       pU, d,                      // A = U, lda = d
                       pA, 16,                     // B = A, ldb = 16 (A is d x 16)
                       &beta1,
                       pY, d);                     // C = Y, ldc = d
    TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cublasSgemm3 failed");

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward with three SGEMM calls (cuBLAS)");
}