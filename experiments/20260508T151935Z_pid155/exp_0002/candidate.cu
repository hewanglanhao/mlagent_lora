#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// -----------------------------------------------------------------------------
// Error checking macro for cuBLAS calls
// -----------------------------------------------------------------------------
#define CUBLAS_CHECK(status)                                                  \
  do {                                                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::stringstream ss;                                                   \
      ss << "cuBLAS error at " << __FILE__ << ":" << __LINE__;                \
      TORCH_CHECK(false, ss.str());                                           \
    }                                                                         \
  } while (0)

// -----------------------------------------------------------------------------
// Static cuBLAS handle management (lazy initialization, reuse)
// -----------------------------------------------------------------------------
static cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }
  // Set the current CUDA stream from PyTorch
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return handle;
}

// -----------------------------------------------------------------------------
// Validation helper
// -----------------------------------------------------------------------------
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
  const int64_t d = W.size(0);
  const int64_t r = 16;
  TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
              "W and X must be d x d");
  TORCH_CHECK(A.size(0) == d && A.size(1) == r &&
              B.size(0) == d && B.size(1) == r,
              "A and B must be d x ", r);
}

// -----------------------------------------------------------------------------
// Forward: Y = W @ X + A @ (B^T @ X)
// Uses three cuBLAS SGEMM calls with no explicit transpose of B.
// -----------------------------------------------------------------------------
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  check_inputs(W, X, A, B);

  W = W.contiguous();
  X = X.contiguous();
  A = A.contiguous();
  B = B.contiguous();

  const int64_t d = W.size(0);
  const int64_t r = 16;
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;

  cublasHandle_t handle = get_cublas_handle();

  // --- Step 1: Compute Y_cm = X^T @ W^T  (column-major, size d x d)
  //     This is (W @ X)^T.
  auto Y = torch::empty({d, d},
      torch::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
  // op(A) = X_ptr -> X^T (since X stored row-major, column-major is X^T)
  // op(B) = W_ptr -> W^T
  CUBLAS_CHECK(cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // transa, transb: both N (use as stored column-major)
    d, d, d,                     // m, n, k
    &alpha,
    X.data_ptr<float>(), d,      // A = X, lda = d (row-major -> column-major X^T)
    W.data_ptr<float>(), d,      // B = W, ldb = d
    &beta0,
    Y.data_ptr<float>(), d));    // C = Y (will hold Y^T)

  // --- Step 2: Compute U = X^T @ B  (column-major, size d x r)
  //     We need op(B) = (stored column-major B^T)^T = B.
  //     Use transb = CUBLAS_OP_T.
  auto U = torch::empty({d, r},
      torch::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  CUBLAS_CHECK(cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_T,   // transa = N (X), transb = T (B^T -> B)
    d, r, d,                     // m = d, n = r, k = d
    &alpha,
    X.data_ptr<float>(), d,      // A = X, lda = d
    B.data_ptr<float>(), r,      // B = B, ldb = r (B stored row-major -> col-major B^T, size r x d)
    &beta0,
    U.data_ptr<float>(), d));    // C = U, ldc = d

  // --- Step 3: Y^T += U @ A^T  (column-major)
  //     op(A) = U (as stored), op(B) = A_ptr -> A^T (since A stored row-major -> col-major A^T)
  CUBLAS_CHECK(cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // both N
    d, d, r,                     // m = d, n = d, k = r
    &alpha,
    U.data_ptr<float>(), d,      // A = U, lda = d
    A.data_ptr<float>(), r,      // B = A, ldb = r (A stored row-major -> col-major A^T, size r x d)
    &beta1,
    Y.data_ptr<float>(), d));    // C = Y (now Y^T = (W@X)^T + (A@(B^T@X))^T)

  // --- Step 4: Transpose Y (column-major) back to row-major Y_row
  //     Use cublasSgeam: C = alpha * op(A) + beta * op(B)
  //     op(A) = CUBLAS_OP_T on Y -> gives Y^row (since (Y^T)^T = Y).
  //     We compute directly into a new tensor.
  auto Y_row = torch::empty({d, d},
      torch::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  CUBLAS_CHECK(cublasSgeam(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,   // transa = T, transb = N (dummy)
    d, d,
    &alpha,
    Y.data_ptr<float>(), d,     // A = Y (column-major Y^T), lda = d
    &beta0,                     // beta = 0, op(B) not used
    Y_row.data_ptr<float>(), d, // B pointer (unused), ldb = d
    Y_row.data_ptr<float>(), d));// C = Y_row, ldc = d

  return Y_row;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "LoRA forward (pure cuBLAS three‑SGEMM)");
}