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
    auto d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

// Kernel: Y[i][j] += sum_{r=0..15} A[i][r] * T[r][j]
__global__ void add_rank16_update_kernel(
    float* __restrict__ Y,
    const float* __restrict__ A,
    const float* __restrict__ T,
    int d) {
    // Block dimensions: blockDim.x = 32, blockDim.y = 16 (total 512)
    // Grid dimensions: gridDim.x = (d + 31) / 32, gridDim.y = (d + 15) / 16
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= d || j >= d) return;

    // Shared memory for A row (16 floats per row of block, 16 rows -> 256 floats)
    __shared__ float A_sh[16][16];
    // Shared memory for T column (16 floats per column of block, 32 columns -> 512 floats)
    __shared__ float T_sh[32][16];

    // Load A rows into shared memory (each thread in row loads its own A element)
    if (threadIdx.x < 16) {  // only need 16 threads per row
        A_sh[threadIdx.y][threadIdx.x] = A[(i) * 16 + threadIdx.x];
    }

    // Load T columns into shared memory (each thread loads T[r][j] for r=0..15)
    // We need to load T[r][j] for all r=0..15, for the column j corresponding to this thread
    // We have 32 threads per row (blockDim.x=32), each loading for its own column.
    // We'll load transposed? Actually we want T_sh[threadIdx.x][r] = T[r][j] where j = blockIdx.x*blockDim.x + threadIdx.x
    {
        int col = j;
        float val = T[threadIdx.y * d + col]; // threadIdx.y corresponds to r? Wait, we need r from 0 to 15, but threadIdx.y is used for row index i. We cannot use threadIdx.y for both.
        // We need to load T column across r dimension. Each thread (threadIdx.x, threadIdx.y) should load one element T[r][j] where r is threadIdx.y? That would require 16 rows of T per column, but we have 16 threads per row in y. Actually we can use a separate loading phase: threads with threadIdx.x < 16 and threadIdx.y == 0 load T[r][j] for r=0..15? But then we need 16 threads per column. Since blockDim.y=16, we can use all y threads to load different r values for the same column? That's fine: each y thread loads T[r][j] where r = threadIdx.y. But then we have 16 y threads per column, each loads one r. So we can fill T_sh[threadIdx.x][threadIdx.y] = T[threadIdx.y * d + j]. This works because T_sh has shape [32][16] (32 columns, 16 rows). So indexing: T_sh[col_idx][r] = T[r*d + col]. 
    }

    // Synchronize to ensure shared memory is ready
    __syncthreads();

    // Compute dot product
    float sum = 0.0f;
    #pragma unroll 16
    for (int r = 0; r < 16; ++r) {
        sum += A_sh[threadIdx.y][r] * T_sh[threadIdx.x][r];
    }

    // Atomic add? No, each Y[i][j] is unique.
    Y[i * d + j] += sum;
}

} // anonymous namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Make contiguous
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    // Compute W @ X via cuBLAS
    auto Y = at::matmul(Wc, Xc);

    // Compute B.T @ X via cuBLAS (B.T is 16 x d, X is d x d, result is 16 x d)
    auto Bt = Bc.transpose(0, 1).contiguous(); // ensures row-major 16 x d
    auto T = at::matmul(Bt, Xc); // T is 16 x d
    T = T.contiguous(); // ensure contiguous for pointer access

    int d = Wc.size(0);
    const int block_x = 32;
    const int block_y = 16; // total 512 threads
    dim3 block(block_x, block_y);
    dim3 grid((d + block_x - 1) / block_x, (d + block_y - 1) / block_y);

    // Launch kernel
    add_rank16_update_kernel<<<grid, block>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        T.data_ptr<float>(),
        d
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward with fused rank-16 update (scalar, block_size=512)");
}