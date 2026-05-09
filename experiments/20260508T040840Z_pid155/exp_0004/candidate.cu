#include <torch/extension.h>

// Block size for the fused LoRA update kernel
constexpr int BLOCK_SIZE = 128;

// Kernel that computes A @ (B.T @ X) and adds it in-place to Y (which initially holds W @ X)
__global__ void lora_add_kernel(
    float* __restrict__ Y,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ X,
    const int d) {

    const int row = blockIdx.y;
    const int col_base = blockIdx.x * BLOCK_SIZE;
    const int tx = threadIdx.x;
    const int col = col_base + tx;
    if (col >= d) return;

    // Shared memory for one row of A and one row of B (each rank=16)
    __shared__ float s_A[16];
    __shared__ float s_B[16];

    // Load A[row] into shared memory using float4 (16 floats = 4 float4)
    if (tx < 4) {
        const float4* src = reinterpret_cast<const float4*>(A + row * 16 + tx * 4);
        float4* dst = reinterpret_cast<float4*>(s_A + tx * 4);
        *dst = *src;
    }
    __syncthreads();  // ensure A is visible to all

    float lora_val = 0.0f;

    for (int l = 0; l < d; ++l) {
        // Load B[l] into shared memory (4 consecutive float4 loads by first 4 threads)
        if (tx < 4) {
            const float4* src = reinterpret_cast<const float4*>(B + l * 16 + tx * 4);
            float4* dst = reinterpret_cast<float4*>(s_B + tx * 4);
            *dst = *src;
        }
        __syncthreads();  // ensure B is visible before reading

        // Load one element from X row l (coalesced across threads)
        float x = X[l * d + col];

        // Fully unrolled dot product of s_A and s_B (rank 16)
        float ab = 0.0f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            ab += s_A[r] * s_B[r];
        }
        lora_val += ab * x;

        __syncthreads();  // ensure all threads are done before next B load
    }

    // Atomically add the LoRA contribution (each output element is written by exactly one thread)
    Y[row * d + col] += lora_val;
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
    const int64_t d = W.size(0);
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

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    const int64_t d = Wc.size(0);

    // Compute W @ X using cuBLAS (via PyTorch)
    auto Y = at::matmul(Wc, Xc);

    // Launch fused LoRA kernel: each thread computes one output element
    dim3 block(BLOCK_SIZE);
    dim3 grid((d + BLOCK_SIZE - 1) / BLOCK_SIZE, d);

    lora_add_kernel<<<grid, block>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Bc.data_ptr<float>(),
        Xc.data_ptr<float>(),
        d
    );

    // Optional: synchronize and catch errors (omitted for performance, but safe for correctness)
    // cudaDeviceSynchronize();

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (W@X + A@(B.T@X)) with fused rank-16 update");
}