#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

// ---------------------------------------------------------------------------
// Input validation (same as baseline)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Kernel 1: T = B^T @ X   (16 x d) = (16 x d) * (d x d)
// ---------------------------------------------------------------------------
template <int TILE_N, int TILE_K>
__global__ void compute_T_kernel(const float* __restrict__ B,
                                 const float* __restrict__ X,
                                 float* __restrict__ C,
                                 int d) {
    // blockIdx.x  : column tile index
    // blockDim.x  : TILE_N
    // blockDim.y  : 16
    int col = blockIdx.x * TILE_N + threadIdx.x;   // output column
    int row = threadIdx.y;                         // output row (0..15)

    if (col >= d) return;

    float sum = 0.0f;

    // Loop over reduction dimension i in tiles of TILE_K
    for (int i = 0; i < d; i += TILE_K) {
        __shared__ float Bs[TILE_K][16];
        __shared__ float Xs[TILE_K][TILE_N];

        // Coalesced load of B tile (d x 16) -> Bs[TILE_K][16]
        if (threadIdx.y < TILE_K && threadIdx.x < 16) {
            int i_idx = i + threadIdx.y;
            Bs[threadIdx.y][threadIdx.x] = (i_idx < d) ? B[i_idx * 16 + threadIdx.x] : 0.0f;
        }

        // Coalesced load of X tile (d x d) -> Xs[TILE_K][TILE_N]
        int load_col = blockIdx.x * TILE_N + threadIdx.x;
        int i_idx = i + threadIdx.y;
        if (i_idx < d && load_col < d) {
            Xs[threadIdx.y][threadIdx.x] = X[i_idx * d + load_col];
        } else {
            Xs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate partial dot products (unrolled over TILE_K)
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += Bs[k][row] * Xs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * d + col] = sum;
}

// ---------------------------------------------------------------------------
// Kernel 2: Y += A @ T   (d x d) += (d x 16) * (16 x d)
// ---------------------------------------------------------------------------
__global__ void add_rank16_update_kernel(float* __restrict__ Y,
                                         const float* __restrict__ A,
                                         const float* __restrict__ T,
                                         int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d * d) return;

    int row = idx / d;
    int col = idx % d;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        sum += A[row * 16 + k] * T[k * d + col];
    }
    Y[idx] += sum;
}

}  // namespace

// ---------------------------------------------------------------------------
// Exported forward function
// ---------------------------------------------------------------------------
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

    const int64_t d = Wc.size(0);
    const int r = 16;

    // Step 1: Y = W @ X  (cuBLAS)
    auto Y = at::matmul(Wc, Xc);

    // Step 2: T = B^T @ X  (custom kernel, 16 x d)
    auto T = torch::empty({r, d}, torch::TensorOptions().dtype(torch::kFloat32).device(Wc.device()));

    constexpr int TILE_N = 128;   // block size in columns
    constexpr int TILE_K = 16;    // tile size in reduction dimension

    dim3 block_T(TILE_N, r);      // (128, 16) threads per block
    dim3 grid_T((d + TILE_N - 1) / TILE_N, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Wc.device().index());
    compute_T_kernel<TILE_N, TILE_K><<<grid_T, block_T, 0, stream>>>(
        Bc.data_ptr<float>(),
        Xc.data_ptr<float>(),
        T.data_ptr<float>(),
        d);

    // Step 3: Y += A @ T  (custom kernel, d x d)
    const int threads_update = 128;
    const int blocks_update = (d * d + threads_update - 1) / threads_update;

    add_rank16_update_kernel<<<blocks_update, threads_update, 0, stream>>>(
        Y.data_ptr<float>(),
        Ac.data_ptr<float>(),
        T.data_ptr<float>(),
        d);

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (cuBLAS + custom rank-16 update)");
}