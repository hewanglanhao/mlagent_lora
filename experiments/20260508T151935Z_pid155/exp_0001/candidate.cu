#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

torch::Tensor forward(torch::Tensor W, torch::Tensor X,
                      torch::Tensor A, torch::Tensor B) {
  // --- input validation ---
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
  // Ensure contiguity
  W = W.contiguous();
  X = X.contiguous();
  A = A.contiguous();
  B = B.contiguous();

  // cuBLAS handle bound to current PyTorch stream
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;

  // Allocate column-major output buffer: holds (W@X)^T initially
  auto Y_cm = torch::empty({d, d},
      torch::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  // Step 1: compute Y_cm = (W@X)^T via: X^T @ W^T
  //   Matrices: X (dxd row-major) treated as column-major X^T,
  //             W (dxd row-major) treated as column-major W^T.
  //   cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  //   here: transa = CUBLAS_OP_N (use X as stored), transb = CUBLAS_OP_N (use W as stored)
  //         m = d, n = d, k = d,
  //         A = X, lda = d  => op(A) = X^T
  //         B = W, ldb = d  => op(B) = W^T
  //         C = Y_cm, ldc = d
  //   result: D = X^T @ W^T = (W@X)^T  (column-major)
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              d, d, d,
              &alpha,
              X.data_ptr<float>(), d,
              W.data_ptr<float>(), d,
              &beta0,
              Y_cm.data_ptr<float>(), d);

  // Step 2: compute V = X^T @ B  (d x r column-major)
  //   This is (B^T @ X)^T, the transpose of the low‑rank intermediate.
  auto V = torch::empty({d, r},
      torch::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  //   Matrices: X (dxd) gives X^T, B (dxr row-major) treated as column-major gives B^T
  //   cublasSgemm: transa=OP_N, transb=OP_N, m=d, n=r, k=d,
  //               A=X, lda=d, B=B, ldb=r,
  //               C=V, ldc=d
  //   result: X^T @ B^T = (B @ X)^T, but we need (B^T @ X)^T = X^T @ B.
  //   With B stored row-major (d,r) and ldb=r, cuBLAS interprets it as column-major (r,d) = B^T.
  //   So op(B) = stored_B = B^T, product = X^T @ B^T = (B @ X)^T.
  //   This is NOT what we want. We need X^T @ B.
  //   Correction: to get X^T @ B, we need op(B) = B (row-major B as column-major?).
  //   Actually, B row-major stored as (d,r), if we view it as column-major (r,d) with lda = r,
  //   then stored_B = B^T. So op(B)=stored_B = B^T.
  //   To get op(B)=B, we need stored_B to be the column-major of B (r x d) with lda = r? That's B^T.
  //   So the only way to get op(B)=B is to transpose B explicitly or use CUBLAS_OP_T.
  //   Since we want to avoid explicit transpose, we compute (B^T @ X)^T = X^T @ B.
  //   That is exactly X^T @ B, not X^T @ B^T.
  //   So we need to compute X^T @ B, not X^T @ B^T.
  //   Minimal change: Use transb = CUBLAS_OP_T on B, so that op(B) = (stored_B)^T.
  //   If we keep ldb=r and stored_B = B (row-major as column-major gives B^T), then op(B)=B.
  //   So set transb = CUBLAS_OP_T.
  //   Re-evaluate: stored_B with transb = CUBLAS_OP_T: the stored matrix is considered column-major of size (n, k)???
  //   Confusing. Let's instead use a different order.
  //   We'll compute V = X^T @ B by setting:
  //     transa = CUBLAS_OP_N (A=X -> op(A)=X^T)
  //     transb = CUBLAS_OP_T (B as stored: with ldb = d? no, ldb must equal columns of stored.)
  //   Actually, it's simpler to recall the row-major method for B (d x r):
  //   To compute C = A * B (A row-major m*k, B row-major k*n) using cuBLAS row-major trick:
  //     cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);
  //   For our subproblem: we want V = X^T @ B, where X^T is d x d, B is d x r.
  //   Here A = X^T (row-major d x d), B = B (row-major d x r). So use formula:
  //     m = r (n of B), n = d (m of A), k = d (inner), 
  //     A_ptr = B, lda = r, B_ptr = X^T, ldb = d, C_ptr = V, ldc = r.
  //   This gives C = (X^T) @ B? Let's test: op(A) = ??? very error-prone.
  //   Given the risk, I revert to simple explicit transpose for B to guarantee correctness.
  //   Since B (d x r) is small, the overhead is negligible.
  auto B_T = B.transpose(0, 1).contiguous();  // 16 x d row-major

  // Now compute V = B_T @ X   (16 x d row-major) using the row-major cuBLAS pattern.
  // We need V of shape (r, d) = (16, d). We'll allocate V as row-major.
  auto V_row = torch::empty({r, d},
      torch::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  // Use the row-major pattern: C = A * B (row-major) => cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
  //    B.cols, A.rows, A.cols, alpha, A, A.cols, B, B.cols, beta, C, B.cols)
  // Here A = B_T (r x d), B = X (d x d), C = V_row (r x d).
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
              d, r, d,
              &alpha,
              B_T.data_ptr<float>(), d,   // lda = d (cols of B_T, row-major)
              X.data_ptr<float>(), d,     // ldb = d (cols of X)
              &beta0,
              V_row.data_ptr<float>(), d); // ldc = d (cols of V_row)
  // V_row now holds B^T @ X (row-major, shape 16 x d).

  // Step 3: compute (A @ (B^T @ X))^T and add to Y_cm with beta=1.
  // We need to compute A @ V (A: d x r row-major, V: r x d row-major) in column-major.
  // Compute (A @ V)^T = V^T @ A^T.
  // V_row is r x d row-major, its column-major representation is V^T (d x r).
  // A is d x r row-major, its column-major representation is A^T (r x d).
  // So we want L_cm = V^T @ A^T, then add to Y_cm.
  // Use cuBLAS: C = op(A) * op(B). Set transa=CUBLAS_OP_T (on V_row), transb=CUBLAS_OP_T (on A)
  //   m = d (rows of op(A) = V^T rows = d), n = d (cols of op(B)=A^T cols = d), k = r.
  //   A = V_row, lda = d (since V_row is r x d row-major, lda=d is columns of V_row)
  //   B = A, ldb = r
  //   C = Y_cm, ldc = d, beta = 1.0
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
              d, d, r,
              &alpha,
              V_row.data_ptr<float>(), d,   // leading dimension = d (cols of V_row)
              A.data_ptr<float>(), r,         // leading dimension = r (cols of A)
              &beta1,
              Y_cm.data_ptr<float>(), d);

  // Step 4: transpose Y_cm to get row-major Y = W@X + A@(B^T@X)
  auto Y = torch::empty({d, d},
      torch::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  // Use cublas<t>geam: C = alpha * op(A) + beta * op(B)
  // op(A) = CUBLAS_OP_T of Y_cm => Y_cm^T
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
              d, d,
              &alpha,
              Y_cm.data_ptr<float>(), d,
              &beta0,
              Y.data_ptr<float>(), d,   // dummy, not used
              Y.data_ptr<float>(), d);

  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "LoRA forward (pure cuBLAS three‑SGEMM)");
}