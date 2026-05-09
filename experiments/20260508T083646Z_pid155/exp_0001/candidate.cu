#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

// Input validation
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

// Fused kernel: computes A @ (B.T @ X) and adds to Y (which already holds W @ X)
// Each block handles a 16x16 tile of the output.
// Shared memory holds tiles of A, B, and X for the current k-tile.
template <int TILE_SIZE>
__global__ void fused_rank16_update_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int d) {

    // Block indices
    int block_i = blockIdx.x * TILE_SIZE;
    int block_j = blockIdx.y * TILE_SIZE;

    // Thread indices within the tile
    int ti = threadIdx.x;  // row in tile (0..15)
    int tj = threadIdx.y;  // col in tile (0..15)

    // Global row and column for this thread
    int row = block_i + ti;
    int col = block_j + tj;

    // Shared memory for tiles: A_tile[16][16], B_tile[16][16], X_tile[16][16]
    __shared__ float As[TILE_SIZE][16];
    __shared__ float Bs[TILE_SIZE][16];
    __shared__ float Xs[TILE_SIZE][TILE_SIZE];

    // Accumulator for the rank-16 update for this output element
    float acc = 0.0f;

    // Preload the A tile for this block's rows (constant across k-tiles)
    // Each thread loads one element of A (if within bounds)
    if (row < d) {
        for (int r = 0; r < 16; r += TILE_SIZE) {
            int rr = r + ti;
            if (rr < 16) {
                As[ti][rr] = A[row * 16 + rr];
            }
        }
    } else {
        // Out-of-bounds rows: set A tile to zero
        for (int r = 0; r < 16; r += TILE_SIZE) {
            int rr = r + ti;
            if (rr < 16) {
                As[ti][rr] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Loop over k-tiles of size TILE_SIZE along the reduction dimension d
    for (int k_start = 0; k_start < d; k_start += TILE_SIZE) {
        // Load B tile: B[k_start : k_start+TILE_SIZE, :] -> Bs[16][16]
        // B is d x 16, so B[k, r] = B[k * 16 + r]
        int k = k_start + ti;  // row in B tile
        if (k < d) {
            for (int r = 0; r < 16; r += TILE_SIZE) {
                int rr = r + tj;
                if (rr < 16) {
                    Bs[ti][rr] = B[k * 16 + rr];
                }
            }
        } else {
            for (int r = 0; r < 16; r += TILE_SIZE) {
                int rr = r + tj;
                if (rr < 16) {
                    Bs[ti][rr] = 0.0f;
                }
            }
        }

        // Load X tile: X[k_start : k_start+TILE_SIZE, block_j : block_j+TILE_SIZE] -> Xs[16][16]
        // X is d x d, so X[k, col] = X[k * d + col]
        int x_col = block_j + tj;
        if (k < d && x_col < d) {
            Xs[ti][tj] = X[k * d + x_col];
        } else {
            Xs[ti][tj] = 0.0f;
        }

        __syncthreads();

        // Compute partial update for this k-tile
        // For each rank r = 0..15, compute inner product over kk (0..TILE_SIZE-1)
        // and multiply by A[row][r], then accumulate.
        // Unroll the rank loop completely.
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            float a_val = As[ti][r];
            float inner = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < TILE_SIZE; ++kk) {
                inner += Bs[kk][r] * Xs[kk][tj];
            }
            acc += a_val * inner;
        }

        __syncthreads();
    }

    // Add the accumulated update to the output Y (which already contains W @ X)
    if (row < d && col < d) {
        Y[row * d + col] += acc;
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

    // Step 2: Fused rank-16 update: Y += A @ (B.T @ X)
    const int64_t d = Wc.size(0);
    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((d + TILE_SIZE - 1) / TILE_SIZE,
              (d + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the custom kernel
    fused_rank16_update_kernel<TILE_SIZE><<<grid, block>>>(
        Ac.data_ptr<float>(),
        Bc.data_ptr<float>(),
        Xc.data_ptr<float>(),
        Y.data_ptr<float>(),
        static_cast<int>(d)
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (cuBLAS + fused rank-16 update)");
}