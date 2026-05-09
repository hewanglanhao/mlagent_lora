#include <torch/extension.h>

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
    TORCH_CHECK(d >= 3584 && d <= 4608,
                "d must be in [3584, 4608]");
}

// Kernel: T = B^T @ X, where T is 16 x d
// Each thread computes one element T[k][j]
__global__ void compute_BT_X_kernel(const float* __restrict__ B,
                                    const float* __restrict__ X,
                                    float* __restrict__ T,
                                    int d) {
    // total threads = 16 * d
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 16 * d) return;
    int k = idx / d;   // 0..15
    int j = idx % d;   // 0..d-1
    float sum = 0.0f;
    // B is d x 16, row-major: B[l][k] = l*16 + k
    // X is d x d, row-major: X[l][j] = l*d + j
    for (int l = 0; l < d; ++l) {
        sum += B[l * 16 + k] * X[l * d + j];
    }
    T[k * d + j] = sum;
}

// Kernel: Y += A @ T, where A is d x 16, T is 16 x d, Y is d x d
// Each thread computes one element Y[i][j] += sum_k A[i][k] * T[k][j]
__global__ void add_A_T_kernel(const float* __restrict__ A,
                               const float* __restrict__ T,
                               float* __restrict__ Y,
                               int d) {
    // total threads = d * d
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d * d) return;
    int i = idx / d;   // 0..d-1
    int j = idx % d;   // 0..d-1
    // A is d x 16, row-major: A[i][k] = i*16 + k
    // T is 16 x d, row-major: T[k][j] = k*d + j
    float sum = 0.0f;
    for (int k = 0; k < 16; ++k) {
        sum += A[i * 16 + k] * T[k * d + j];
    }
    Y[i * d + j] += sum;
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

    // Use cuBLAS for W @ X
    auto Y = at::matmul(Wc, Xc);  // d x d

    const int d = Wc.size(0);

    // Allocate temporary for T = B^T @ X (16 x d)
    auto T = torch::empty({16, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const int threads_per_block = 128;
    int threads;

    // Launch kernel for B^T @ X
    threads = 16 * d;
    int blocks1 = (threads + threads_per_block - 1) / threads_per_block;
    compute_BT_X_kernel<<<blocks1, threads_per_block>>>(
        Bc.data_ptr<float>(), Xc.data_ptr<float>(),
        T.data_ptr<float>(), d);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "compute_BT_X_kernel launch failed");

    // Launch kernel for Y += A @ T
    threads = d * d;
    int blocks2 = (threads + threads_per_block - 1) / threads_per_block;
    add_A_T_kernel<<<blocks2, threads_per_block>>>(
        Ac.data_ptr<float>(), T.data_ptr<float>(),
        Y.data_ptr<float>(), d);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_A_T_kernel launch failed");

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W@X + A@(B^T@X)");
}