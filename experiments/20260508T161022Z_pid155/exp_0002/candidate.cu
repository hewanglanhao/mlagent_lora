#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
  // Shape checks
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All inputs must be CUDA tensors");
  TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
              A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
              "All inputs must be float32");
  TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
              "All inputs must be 2D");
  const int64_t d = W.size(0);
  TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
              "W and X must be d x d");
  TORCH_CHECK(A.size(0) == d && A.size(1) == 16 && B.size(0) == d && B.size(1) == 16,
              "A and B must be d x 16");

  // Make contiguous
  auto Wc = W.contiguous();
  auto Xc = X.contiguous();
  auto Ac = A.contiguous();
  auto Bc = B.contiguous();

  // Allocate output Y and temporary U
  auto Y = torch::empty({d, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto U = torch::empty({16, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  // cuBLAS handle and stream
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasSetStream(handle, stream);

  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;

  // Compute Y = W @ X   (row‑major, using pattern cublas_row_major_matmul)
  // cublasSgemm with opN, opN, m=d, n=d, k=d, lda=d, ldb=d, ldc=d
  // A_ptr = X, B_ptr = W, C_ptr = Y
  cublasSgemm(handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              d, d, d,
              &alpha,
              Xc.data_ptr<float>(), d,
              Wc.data_ptr<float>(), d,
              &beta0,
              Y.data_ptr<float>(), d);

  // Compute U = B.T @ X   (row‑major result U 16×d)
  // opA = N (X), opB = T (B), m=d, n=16, k=d, lda=d, ldb=d, ldc=16
  // A_ptr = X, B_ptr = B, C_ptr = U
  cublasSgemm(handle,
              CUBLAS_OP_N, CUBLAS_OP_T,
              d, 16, d,
              &alpha,
              Xc.data_ptr<float>(), d,
              Bc.data_ptr<float>(), d,
              &beta0,
              U.data_ptr<float>(), 16);

  // Accumulate Y += A @ U   (row‑major, beta=1)
  // opA = N (U), opB = N (A), m=d, n=d, k=16, lda=d, ldb=16, ldc=d
  // A_ptr = U, B_ptr = A, C_ptr = Y
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
  m.def("forward", &forward, "LoRA forward with pure cuBLAS three-SGEMM");
}