#include <torch/extension.h>
#include <cuda_runtime.h>

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
}

// Kernel to compute Weff = W + A @ B.T, where rank=16.
// Uses 16x16 tiles, each block computes one tile.
// Block dimensions: (16,16), total 256 threads per block.
// Each thread computes one element of the tile.
// Vectorized loads (float4) are used for A and B rows/columns.
__global__ void rank16_update_kernel(float* __restrict__ Weff,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     int d) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int local_row = ty;
    const int local_col = tx;
    const int row = by * 16 + local_row;
    const int col = bx * 16 + local_col;

    if (row >= d || col >= d) return;

    // Load A values for this row (16 floats) using float4
    float4 a0, a1, a2, a3;
    const float* a_ptr = A + row * 16;
    a0 = *reinterpret_cast<const float4*>(a_ptr);
    a1 = *reinterpret_cast<const float4*>(a_ptr + 4);
    a2 = *reinterpret_cast<const float4*>(a_ptr + 8);
    a3 = *reinterpret_cast<const float4*>(a_ptr + 12);

    // Load B values for this column (16 floats) using float4
    float4 b0, b1, b2, b3;
    const float* b_ptr = B + col * 16;
    b0 = *reinterpret_cast<const float4*>(b_ptr);
    b1 = *reinterpret_cast<const float4*>(b_ptr + 4);
    b2 = *reinterpret_cast<const float4*>(b_ptr + 8);
    b3 = *reinterpret_cast<const float4*>(b_ptr + 12);

    // Compute dot product fully unrolled
    float sum = 0.0f;
    sum += a0.x * b0.x + a0.y * b0.y + a0.z * b0.z + a0.w * b0.w;
    sum += a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w;
    sum += a2.x * b2.x + a2.y * b2.y + a2.z * b2.z + a2.w * b2.w;
    sum += a3.x * b3.x + a3.y * b3.y + a3.z * b3.z + a3.w * b3.w;

    // Load original W value and add
    Weff[row * d + col] += sum;
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

    const int d = Wc.size(0);

    // Allocate Weff as a copy of W (not modifying original)
    auto Weff = Wc.clone();

    // Launch kernel to add A @ B.T into Weff
    dim3 block(16, 16); // 256 threads
    dim3 grid((d + 15) / 16, (d + 15) / 16);
    rank16_update_kernel<<<grid, block>>>(
        Weff.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Bc.data_ptr<float>(),
        d);

    // Compute final result = Weff @ X
    auto result = at::matmul(Weff, Xc);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward with rank-16 precomputed update");
}