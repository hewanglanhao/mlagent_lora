#include <torch/extension.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Helper: check inputs and ensure contiguous
// -----------------------------------------------------------------------------
static void check_inputs(const torch::Tensor& W,
                         const torch::Tensor& X,
                         const torch::Tensor& A,
                         const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "all inputs must be float32");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be 2D");
    const int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be square d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16,
                "A must be d x 16");
    TORCH_CHECK(B.size(0) == d && B.size(1) == 16,
                "B must be d x 16");
}

// -----------------------------------------------------------------------------
// Kernel: compute C = B^T @ X, stored as d x 16 (row‑major)
//   C[j][r] = sum_{k=0}^{d-1} B[k][r] * X[k][j]
//   Each thread handles one element (j,r). GridDim = (16*d).
// -----------------------------------------------------------------------------
__global__ void compute_C_kernel(const float* __restrict__ B,
                                 const float* __restrict__ X,
                                 float* __restrict__ C,
                                 int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 16 * d) return;
    int j = idx / 16;          // output row (column of X)
    int r = idx % 16;          // rank index

    float sum = 0.0f;
    for (int k = 0; k < d; ++k) {
        sum += B[k * 16 + r] * X[k * d + j];
    }
    C[j * 16 + r] = sum;
}

// -----------------------------------------------------------------------------
// Kernel: add A @ C to WX (output)
//   output[i*d + j] += dot(A[i,:], C[j,:])
//   Each thread handles one output element (i,j). GridDim = (d, d).
//   Fully unrolled over rank=16, using float4 loads.
// -----------------------------------------------------------------------------
__global__ void add_rank16_kernel(const float* __restrict__ A,
                                  const float* __restrict__ C,
                                  float* __restrict__ output,
                                  int d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d || j >= d) return;

    // Load A[i,0:15] as 4 float4
    const float4* A4 = reinterpret_cast<const float4*>(&A[i * 16]);
    float4 a0 = A4[0];
    float4 a1 = A4[1];
    float4 a2 = A4[2];
    float4 a3 = A4[3];

    // Load C[j,0:15] as 4 float4
    const float4* C4 = reinterpret_cast<const float4*>(&C[j * 16]);
    float4 c0 = C4[0];
    float4 c1 = C4[1];
    float4 c2 = C4[2];
    float4 c3 = C4[3];

    // Dot product fully unrolled
    float sum = a0.x * c0.x + a0.y * c0.y + a0.z * c0.z + a0.w * c0.w +
                a1.x * c1.x + a1.y * c1.y + a1.z * c1.z + a1.w * c1.w +
                a2.x * c2.x + a2.y * c2.y + a2.z * c2.z + a2.w * c2.w +
                a3.x * c3.x + a3.y * c3.y + a3.z * c3.z + a3.w * c3.w;

    output[i * d + j] += sum;
}

// -----------------------------------------------------------------------------
// Forward function
// -----------------------------------------------------------------------------
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

    // 1) Compute W @ X via ATen / cuBLAS
    auto output = at::matmul(Wc, Xc);   // d x d

    // 2) Compute C = B^T @ X, stored as d x 16 (row‑major)
    auto C = torch::empty({d, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    const int block_size_C = 256;
    const int grid_size_C = (16 * d + block_size_C - 1) / block_size_C;
    compute_C_kernel<<<grid_size_C, block_size_C>>>(
        Bc.data_ptr<float>(),
        Xc.data_ptr<float>(),
        C.data_ptr<float>(),
        (int)d
    );

    // 3) Add A @ C to output
    const int block_size_add = 16;  // small blocks, each thread handles one element
    const dim3 block(16, 16);
    const dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y);
    add_rank16_kernel<<<grid, block>>>(
        Ac.data_ptr<float>(),
        C.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)d
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: Y = W@X + A@(B.T@X)");
}