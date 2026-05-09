#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUBLAS_CHECK(call)                                                        \
  do {                                                                            \
    cublasStatus_t err = call;                                                    \
    TORCH_CHECK(err == CUBLAS_STATUS_SUCCESS,                                     \
                "cuBLAS error: ", err, " at ", __FILE__, ":", __LINE__);          \
  } while (0)

namespace {

// Static cuBLAS handle to avoid API issues across runs.
cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }
  // Bind to the current PyTorch CUDA stream.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return handle;
}

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
              "All inputs must be 2D tensors");
  const int64_t d = W.size(0);
  TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
              "W and X must be d x d");
  TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
              B.size(0) == d && B.size(1) == 16,
              "A and B must be d x 16");
  const int64_t rank = 16;
  TORCH_CHECK(A.size(1) == rank && B.size(1) == rank,
              "Inner dimension of A and B must be the rank (16)");
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  check_inputs(W, X, A, B);

  // Ensure contiguous memory for pointer access.
  auto W_c = W.contiguous();
  auto X_c = X.contiguous();
  auto A_c = A.contiguous();
  auto B_c = B.contiguous();

  int64_t d = W_c.size(0);
  const int64_t rank = 16;

  // Allocate output Y and intermediate C (for A @ B.T)
  auto Y = torch::empty({d, d}, torch::TensorOptions().dtype(at::kFloat).device(W_c.device()));
  auto C = torch::empty({d, d}, torch::TensorOptions().dtype(at::kFloat).device(W_c.device()));

  // Get raw pointers (all float*)
  float* W_ptr = W_c.data_ptr<float>();
  float* X_ptr = X_c.data_ptr<float>();
  float* A_ptr = A_c.data_ptr<float>();
  float* B_ptr = B_c.data_ptr<float>();
  float* Y_ptr = Y.data_ptr<float>();
  float* C_ptr = C.data_ptr<float>();

  // cuBLAS handle
  cublasHandle_t handle = get_cublas_handle();

  // Constants for SGEMM
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;

  // -----------------------------------------------------------------
  // Stage 1: Y = W @ X
  // Row-major C = A * B computed as:
  //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
  //               B, ldb, A, lda, &beta, C, ldc)
  // with A = W (d x d, lda = d), B = X (d x d, ldb = d)
  // -----------------------------------------------------------------
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           d, d, d,
                           &alpha,
                           X_ptr, d,   // "B" in the pattern (second matrix)
                           W_ptr, d,   // "A" in the pattern (first matrix)
                           &beta0,
                           Y_ptr, d));

  // -----------------------------------------------------------------
  // Stage 2: C = A @ B.T   (A: d x 16, B: d x 16, B.T: 16 x d)
  // Pattern: A as first matrix (lda=rank), B as second matrix (ldb=rank)
  // -----------------------------------------------------------------
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           d, d, rank,
                           &alpha,
                           B_ptr, rank, // second matrix pointer (B row-major -> column-major is B.T)
                           A_ptr, rank, // first matrix pointer (A)
                           &beta0,
                           C_ptr, d));

  // -----------------------------------------------------------------
  // Stage 3: Y = alpha * (C @ X) + beta * Y  (with beta=1)
  // C: d x d, X: d x d
  // Pattern: C as first matrix (lda=d), X as second matrix (ldb=d)
  // -----------------------------------------------------------------
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           d, d, d,
                           &alpha,
                           X_ptr, d,   // second matrix pointer
                           C_ptr, d,   // first matrix pointer
                           &beta1,
                           Y_ptr, d));

  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "LoRA forward: Y = W @ X + A @ (B.T @ X) using three cuBLAS SGEMM calls");
}