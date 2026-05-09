#include <torch/extension.h>

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 8;
constexpr int BLOCK_K = 32;
constexpr int RANK = 16;

__global__ void fused_rank16_kernel(
    float* __restrict__ Y,
    const float* __restrict__ X,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const int d)
{
    __shared__ float As[BLOCK_M][RANK];
    __shared__ float Bs[BLOCK_K][RANK];
    __shared__ float Xs[BLOCK_K][BLOCK_N];

    int localRow = threadIdx.x / BLOCK_N;
    int localCol = threadIdx.x % BLOCK_N;
    int tid = threadIdx.x;

    // global row and column for this thread's output element
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int globalRow = blockRow * BLOCK_M + localRow;
    int globalCol = blockCol * BLOCK_N + localCol;

    // ----- Load A into shared memory -----
    // Only BLOCK_M threads needed; use first BLOCK_M threads
    if (tid < BLOCK_M) {
        int row = blockRow * BLOCK_M + tid;
        if (row < d) {
            for (int k = 0; k < RANK; ++k) {
                As[tid][k] = A[row * RANK + k];
            }
        } else {
            for (int k = 0; k < RANK; ++k) {
                As[tid][k] = 0.0f;
            }
        }
    }
    __syncthreads();

    // If output element is out of bounds, nothing to compute
    if (globalRow >= d || globalCol >= d) return;

    // Load A row into registers (fully unrolled for performance)
    float regA[RANK];
    #pragma unroll
    for (int k = 0; k < RANK; ++k) {
        regA[k] = As[localRow][k];
    }

    // Loop over chunks of the inner dimension l
    for (int l_start = 0; l_start < d; l_start += BLOCK_K) {
        // ----- Load B and X tiles into shared memory -----
        // Load Bs: BLOCK_K rows, RANK columns
        for (int idx = tid; idx < BLOCK_K * RANK; idx += blockDim.x) {
            int l = idx / RANK;
            int k = idx % RANK;
            int row = l_start + l;
            float val = (row < d) ? B[row * RANK + k] : 0.0f;
            Bs[l][k] = val;
        }
        // Load Xs: BLOCK_K rows, BLOCK_N columns
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += blockDim.x) {
            int l = idx / BLOCK_N;
            int c = idx % BLOCK_N;
            int row = l_start + l;
            int col = blockCol * BLOCK_N + c;
            float val = (row < d && col < d) ? X[row * d + col] : 0.0f;
            Xs[l][c] = val;
        }
        __syncthreads();

        // ----- Compute partial contribution for this chunk -----
        float sum = 0.0f;
        #pragma unroll
        for (int l = 0; l < BLOCK_K; ++l) {
            float xval = Xs[l][localCol];
            #pragma unroll
            for (int k = 0; k < RANK; ++k) {
                sum += xval * regA[k] * Bs[l][k];
            }
        }

        // Write the contribution (accumulate into Y)
        Y[globalRow * d + globalCol] += sum;
        __syncthreads();
    }
}

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors.");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "All inputs must be float32 tensors.");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "All inputs must be rank-2 tensors.");
    const int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d.");
    TORCH_CHECK(A.size(0) == d && A.size(1) == RANK,
                "A must be d x 16.");
    TORCH_CHECK(B.size(0) == d && B.size(1) == RANK,
                "B must be d x 16.");
}

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

    // W @ X via cuBLAS
    auto Y = at::matmul(Wc, Xc);
    if (!Y.is_contiguous()) Y = Y.contiguous();

    // Fused low‑rank update kernel
    int64_t d = Wc.size(0);
    dim3 block(256);
    dim3 grid((d + BLOCK_N - 1) / BLOCK_N, (d + BLOCK_M - 1) / BLOCK_M);
    fused_rank16_kernel<<<grid, block>>>(
        Y.data_ptr<float>(),
        Xc.data_ptr<float>(),
        Ac.data_ptr<float>(),
        Bc.data_ptr<float>(),
        (int)d);

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward with fused rank‑16 update");
}