#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------
static void check_inputs(const torch::Tensor& W,
                         const torch::Tensor& X,
                         const torch::Tensor& A,
                         const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "All inputs must be float32");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "All inputs must be rank-2 tensors");
    int64_t d = W.size(0);
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 &&
                B.size(0) == d && B.size(1) == 16,
                "A and B must be d x 16");
}

// ---------------------------------------------------------------------------
// Kernel 1: C = B^T @ X   (C is 16 x d)
// ---------------------------------------------------------------------------
__global__ void compute_C_kernel(const float* __restrict__ B,
                                 const float* __restrict__ X,
                                 float* __restrict__ C,
                                 int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 16 * d;
    if (idx >= total) return;

    int k = idx / d;          // row of C (0..15)
    int j = idx % d;          // column of C

    float sum = 0.0f;
    for (int l = 0; l < d; ++l) {
        sum += B[l * 16 + k] * X[l * d + j];
    }
    C[k * d + j] = sum;
}

// ---------------------------------------------------------------------------
// Kernel 2: Y = WX + A @ C   (vectorized, 4 columns per thread)
// ---------------------------------------------------------------------------
__global__ void compute_Y_kernel(const float* __restrict__ A,
                                 const float* __restrict__ C,
                                 const float* __restrict__ WX,
                                 float* __restrict__ Y,
                                 int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = d / 4;               // number of 4‑column groups
    int total_threads = d * num_groups;
    if (tid >= total_threads) return;

    int i = tid / num_groups;
    int j_group = tid % num_groups;
    int j = j_group * 4;

    const float* A_row = A + i * 16;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        float aik = A_row[k];
        const float* C_row = C + k * d + j;
        // Load 4 consecutive floats from C as float4
        float4 c_vec = *reinterpret_cast<const float4*>(C_row);
        sum0 += aik * c_vec.x;
        sum1 += aik * c_vec.y;
        sum2 += aik * c_vec.z;
        sum3 += aik * c_vec.w;
    }

    // Load corresponding 4 elements from WX
    const float4 wx_vec = *reinterpret_cast<const float4*>(WX + i * d + j);

    float4 y_vec;
    y_vec.x = wx_vec.x + sum0;
    y_vec.y = wx_vec.y + sum1;
    y_vec.z = wx_vec.z + sum2;
    y_vec.w = wx_vec.w + sum3;

    *reinterpret_cast<float4*>(Y + i * d + j) = y_vec;
}

// ---------------------------------------------------------------------------
// Exported forward function
// ---------------------------------------------------------------------------
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);

    // Ensure contiguous memory layouts
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    int64_t d = Wc.size(0);
    TORCH_CHECK(d % 4 == 0, "d must be divisible by 4 for vectorized kernel");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Wc.device());

    // 1. Large GEMM: W @ X  (cuBLAS)
    auto WX = at::matmul(Wc, Xc);   // d x d, contiguous

    // 2. Intermediate: C = B^T @ X   (16 x d)
    auto C = torch::empty({16, d}, options);

    const int block1 = 256;
    int grid1 = (16 * d + block1 - 1) / block1;
    compute_C_kernel<<<grid1, block1>>>(
        Bc.data_ptr<float>(), Xc.data_ptr<float>(), C.data_ptr<float>(), d);

    // 3. Final result: Y = WX + A @ C   (d x d)
    auto Y = torch::empty({d, d}, options);

    const int block2 = 256;
    int num_groups = d / 4;
    int total_threads2 = d * num_groups;
    int grid2 = (total_threads2 + block2 - 1) / block2;
    compute_Y_kernel<<<grid2, block2>>>(
        Ac.data_ptr<float>(), C.data_ptr<float>(), WX.data_ptr<float>(),
        Y.data_ptr<float>(), d);

    return Y;
}

// ---------------------------------------------------------------------------
// PyTorch module binding
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward: W@X + A@(B.T@X) with fused rank-16 update");
}