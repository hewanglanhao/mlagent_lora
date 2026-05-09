#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

// Runtime shape and type checks
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

// Fused kernel: Y += A @ C, where C = B.T @ X (16 x d)
// Each block computes a 16x16 tile of the output.
// Vectorized loads (float4) are used for both A and C tiles.
__global__ void fused_ac_add_kernel(
    float* __restrict__ Y,
    const float* __restrict__ A,
    const float* __restrict__ C,
    int d) {

    // Shared memory for A tile (16x16) and C tile (16x16)
    __shared__ float As[16][16];
    __shared__ float Cs[16][16];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Load A tile using float4 (64 float4 loads needed)
    if (tid < 64) {
        int row = tid / 4;          // 0..15
        int col_group = tid % 4;    // 0..3
        int global_row = blockIdx.y * 16 + row;
        if (global_row < d) {
            const float4* src = reinterpret_cast<const float4*>(A + global_row * 16 + col_group * 4);
            float4* dst = reinterpret_cast<float4*>(&As[row][col_group * 4]);
            *dst = *src;
        } else {
            // Out-of-bounds rows: fill with zero
            float4* dst = reinterpret_cast<float4*>(&As[row][col_group * 4]);
            *dst = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    // Load C tile using float4 (64 float4 loads needed)
    if (tid < 64) {
        int row = tid / 4;          // 0..15
        int col_group = tid % 4;    // 0..3
        int global_col = blockIdx.x * 16 + col_group * 4;
        if (global_col < d) {
            const float4* src = reinterpret_cast<const float4*>(C + row * d + global_col);
            float4* dst = reinterpret_cast<float4*>(&Cs[row][col_group * 4]);
            *dst = *src;
        } else {
            // Out-of-bounds columns: fill with zero
            float4* dst = reinterpret_cast<float4*>(&Cs[row][col_group * 4]);
            *dst = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    __syncthreads();

    // Compute one output element
    int row = threadIdx.y;
    int col = threadIdx.x;
    int global_row = blockIdx.y * 16 + row;
    int global_col = blockIdx.x * 16 + col;

    if (global_row < d && global_col < d) {
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += As[row][k] * Cs[k][col];
        }
        Y[global_row * d + global_col] += sum;
    }
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Ensure contiguous memory
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    // Step 1: W @ X using cuBLAS (via ATen)
    auto Y = at::matmul(Wc, Xc);
    Y = Y.contiguous();  // ensure contiguous for kernel access

    // Step 2: C = B.T @ X using cuBLAS
    auto C = at::matmul(Bc.transpose(0, 1).contiguous(), Xc);
    C = C.contiguous();

    int64_t d = W.size(0);

    // Step 3: Fused A @ C + add to Y using custom vectorized kernel
    dim3 block(16, 16);
    dim3 grid((d + 15) / 16, (d + 15) / 16);

    fused_ac_add_kernel<<<grid, block>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        C.data_ptr<float>(),
        static_cast<int>(d)
    );

    // No need to sync here; PyTorch handles stream synchronization on return
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (cuBLAS + fused rank-16 update with float4)");
}