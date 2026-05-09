#include <torch/extension.h>

namespace {

// Validate inputs: CUDA, float32, rank-2, correct shapes.
void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat &&
                X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat &&
                B.scalar_type() == at::kFloat,
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

// Custom kernel: Y = WX + A @ T, where T is 16 x d (B.T @ X).
// Each thread computes one output element (row, col).
// Fully unrolled over the rank dimension (k = 0..15).
__global__ void rank16_add_kernel(
    const float* __restrict__ WX,
    const float* __restrict__ A,
    const float* __restrict__ T,
    float* __restrict__ Y,
    int d) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * d;
    if (idx >= total) return;

    int row = idx / d;
    int col = idx % d;

    const float* A_row = A + row * 16;
    float sum = 0.0f;

    // Unrolled k = 0..15
    sum += A_row[0]  * T[0  * d + col];
    sum += A_row[1]  * T[1  * d + col];
    sum += A_row[2]  * T[2  * d + col];
    sum += A_row[3]  * T[3  * d + col];
    sum += A_row[4]  * T[4  * d + col];
    sum += A_row[5]  * T[5  * d + col];
    sum += A_row[6]  * T[6  * d + col];
    sum += A_row[7]  * T[7  * d + col];
    sum += A_row[8]  * T[8  * d + col];
    sum += A_row[9]  * T[9  * d + col];
    sum += A_row[10] * T[10 * d + col];
    sum += A_row[11] * T[11 * d + col];
    sum += A_row[12] * T[12 * d + col];
    sum += A_row[13] * T[13 * d + col];
    sum += A_row[14] * T[14 * d + col];
    sum += A_row[15] * T[15 * d + col];

    Y[idx] = WX[idx] + sum;
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Ensure contiguous memory for pointer-based access.
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = Wc.size(0);

    // 1. W @ X using cuBLAS (ATen).
    auto WX = at::matmul(Wc, Xc);

    // 2. T = B.T @ X using cuBLAS. B.T is 16 x d, X is d x d -> 16 x d.
    auto Bt = Bc.transpose(0, 1).contiguous();
    auto T = at::matmul(Bt, Xc);

    // 3. Fused rank-16 update: Y = WX + A @ T.
    auto Y = torch::empty_like(WX);
    const int total_elements = static_cast<int>(d * d);
    const int block_size = 128;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    rank16_add_kernel<<<grid_size, block_size, 0, 0>>>(
        WX.data_ptr<float>(),
        Ac.data_ptr<float>(),
        T.data_ptr<float>(),
        Y.data_ptr<float>(),
        static_cast<int>(d)
    );

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (rank16 scalar b128)");
}