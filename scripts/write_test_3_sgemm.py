from __future__ import annotations

import argparse
from pathlib import Path


SOURCE = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <limits>

namespace {

constexpr int kRank = 16;

const char* cublas_status_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed: ", cublas_status_name(status));
}

void check_inputs(
    const torch::Tensor& W,
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
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() && A.is_contiguous() && B.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(W.device() == X.device() && W.device() == A.device() && W.device() == B.device(),
              "all inputs must be on the same CUDA device");

  const int64_t d = W.size(0);
  TORCH_CHECK(d > 0, "hidden dimension must be positive");
  TORCH_CHECK(W.size(1) == d, "W must have shape [d, d]");
  TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must have shape [d, d]");
  TORCH_CHECK(A.size(0) == d && A.size(1) == kRank, "A must have shape [d, 16]");
  TORCH_CHECK(B.size(0) == d && B.size(1) == kRank, "B must have shape [d, 16]");
  TORCH_CHECK(d <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              "hidden dimension is too large for cuBLAS int dimensions");
}

}  // namespace

torch::Tensor forward(
    torch::Tensor W,
    torch::Tensor X,
    torch::Tensor A,
    torch::Tensor B) {
  check_inputs(W, X, A, B);
  c10::cuda::CUDAGuard device_guard(W.device());

  const int d = static_cast<int>(W.size(0));
  auto Y = torch::empty_like(W);
  auto U = torch::empty({d, kRank}, W.options());

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const float zero = 0.0f;
  const float one = 1.0f;

  // Row-major W @ X is column-major X^T @ W^T in the same memory.
  check_cublas(
      cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          d,
          d,
          d,
          &one,
          X.data_ptr<float>(),
          d,
          W.data_ptr<float>(),
          d,
          &zero,
          Y.data_ptr<float>(),
          d),
      "cublasSgemm(WX)");

  // U column-major [d, 16] = X^T @ B. This is (B^T @ X)^T without materializing B^T.
  check_cublas(
      cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          d,
          kRank,
          d,
          &one,
          X.data_ptr<float>(),
          d,
          B.data_ptr<float>(),
          kRank,
          &zero,
          U.data_ptr<float>(),
          d),
      "cublasSgemm(XTB)");

  // Y column-major += U @ A^T, which is row-major A @ (B^T @ X).
  check_cublas(
      cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          d,
          d,
          kRank,
          &one,
          U.data_ptr<float>(),
          d,
          A.data_ptr<float>(),
          kRank,
          &one,
          Y.data_ptr<float>(),
          d),
      "cublasSgemm(low_rank)");

  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "LoRA forward optimized");
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the embedded 3-SGEMM test CUDA source.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(SOURCE, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
