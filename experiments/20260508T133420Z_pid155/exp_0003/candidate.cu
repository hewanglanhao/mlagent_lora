#include <torch/extension.h>

// Block size for tile-based outer product kernel
constexpr int BLOCK_SIZE = 128;
constexpr int RANK = 16;

// CUDA kernel: add A @ B.T to Weff (d x d), tile-based with shared memory
__global__ void outer_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Weff,
    int d) {
    // Shared memory for B tile: 128 columns x 16 rank
    __shared__ float sB[BLOCK_SIZE][RANK];

    int tile_row = blockIdx.y * BLOCK_SIZE;
    int tile_col = blockIdx.x * BLOCK_SIZE;
    int local_idx = threadIdx.x;  // 0..127

    // Load B tile: each thread loads one column (local_idx) for all ranks
    int global_col = tile_col + local_idx;
    if (global_col < d) {
        for (int k = 0; k < RANK; ++k) {
            sB[local_idx][k] = B[global_col * RANK + k];
        }
    } else {
        // Out-of-range columns: fill with zeros (safe)
        for (int k = 0; k < RANK; ++k) {
            sB[local_idx][k] = 0.0f;
        }
    }
    __syncthreads();

    // Each thread handles one row of the tile
    int global_row = tile_row + local_idx;
    if (global_row >= d) return;

    // Load the entire row of A (16 floats) into registers
    float a_row[RANK];
    for (int k = 0; k < RANK; ++k) {
        a_row[k] = A[global_row * RANK + k];
    }

    // Iterate over all columns in the tile and compute the outer product contribution
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        if (tile_col + j >= d) break;
        float sum = 0.0f;
        // Unrolled loop over rank
        #pragma unroll
        for (int k = 0; k < RANK; ++k) {
            sum += a_row[k] * sB[j][k];
        }
        Weff[global_row * d + (tile_col + j)] += sum;
    }
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
    int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == RANK &&
                B.size(0) == d && B.size(1) == RANK,
                "A and B must be d x 16");
}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Make sure all tensors are contiguous
    auto W_cont = W.contiguous();
    auto X_cont = X.contiguous();
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();

    int64_t d = W_cont.size(0);

    // Precompute Weff = W + A @ B.T using a custom outer product kernel
    auto Weff = W_cont.clone();  // d x d, initialized with W

    // Launch kernel
    dim3 grid((d + BLOCK_SIZE - 1) / BLOCK_SIZE, (d + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    outer_add_kernel<<<grid, block>>>(
        A_cont.data_ptr<float>(),
        B_cont.data_ptr<float>(),
        Weff.data_ptr<float>(),
        d);

    // Synchronize to catch kernel errors
    cudaDeviceSynchronize();

    // Compute final result: Weff @ X
    return at::matmul(Weff, X_cont);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward using precomputed Weff = W + A @ B.T");
}