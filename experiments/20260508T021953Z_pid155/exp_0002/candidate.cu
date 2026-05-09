#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

// Kernel 1: Compute T = B^T @ X, T is 16 x d
__global__ void bt_x_kernel(const float* B, const float* X, float* T, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 16 * d;
    if (idx >= total) return;
    int r = idx / d;         // row in T (0..15)
    int c = idx % d;         // column in T (0..d-1)
    float sum = 0.0f;
    // Loop over common dimension k (0..d-1)
    for (int k = 0; k < d; ++k) {
        // B[k][r] = B[k * 16 + r]
        // X[k][c] = X[k * d + c]
        sum += B[k * 16 + r] * X[k * d + c];
    }
    T[r * d + c] = sum;
}

// Kernel 2: Compute Y += A @ T, with fully unrolled rank-16 loop
__global__ void a_t_add_kernel(const float* A, const float* T, float* Y, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * d;
    if (idx >= total) return;
    int i = idx / d;         // row in Y
    int j = idx % d;         // column in Y
    float update = 0.0f;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        // A[i][r] = A[i * 16 + r]
        // T[r][j] = T[r * d + j]
        update += A[i * 16 + r] * T[r * d + j];
    }
    Y[idx] += update;
}

} // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(W));

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = W.size(0);

    // Step 1: Y = W @ X  (cuBLAS via ATen)
    auto Y = at::matmul(Wc, Xc);  // d x d, contiguous

    // Step 2: Compute T = B^T @ X, shape (16, d)
    auto T = torch::empty({16, d}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const int block_size = 256;
    int total_btx = 16 * d;
    dim3 grid_btx((total_btx + block_size - 1) / block_size);
    bt_x_kernel<<<grid_btx, block_size>>>(Bc.data_ptr<float>(),
                                          Xc.data_ptr<float>(),
                                          T.data_ptr<float>(),
                                          (int)d);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Step 3: Y += A @ T
    int total_y = d * d;
    dim3 grid_y((total_y + block_size - 1) / block_size);
    a_t_add_kernel<<<grid_y, block_size>>>(Ac.data_ptr<float>(),
                                           T.data_ptr<float>(),
                                           Y.data_ptr<float>(),
                                           (int)d);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (cuBLAS W@X + custom rank-16 update)");
}