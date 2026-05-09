#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kRank = 16;
constexpr int kBlockSize = 512;

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
    TORCH_CHECK(W.device() == X.device() && W.device() == A.device() && W.device() == B.device(),
                "all inputs must be on the same CUDA device");

    const int64_t d = W.size(0);
    TORCH_CHECK(d >= 3584 && d <= 4608, "d must be in [3584, 4608]");
    TORCH_CHECK(W.size(1) == d, "W must be d x d");
    TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must be d x d");
    TORCH_CHECK(A.size(0) == d && A.size(1) == kRank, "A must be d x 16");
    TORCH_CHECK(B.size(0) == d && B.size(1) == kRank, "B must be d x 16");
}

__global__ void rank16_add_scalar_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ T,
    int d) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(d) * d;
    if (idx >= total) return;

    int row = static_cast<int>(idx / d);
    int col = static_cast<int>(idx - static_cast<int64_t>(row) * d);

    const float* arow = A + static_cast<int64_t>(row) * kRank;
    const float* tcol = T + col;

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < kRank; ++k) {
        acc += arow[k] * tcol[static_cast<int64_t>(k) * d];
    }
    y[idx] += acc;
}

__global__ void rank16_add_vec4_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ T,
    int d_vec,
    int d) {
    int64_t vec_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_vec = static_cast<int64_t>(d) * d_vec;
    if (vec_idx >= total_vec) return;

    int row = static_cast<int>(vec_idx / d_vec);
    int col_vec = static_cast<int>(vec_idx - static_cast<int64_t>(row) * d_vec);
    int col = col_vec * 4;

    const float* arow = A + static_cast<int64_t>(row) * kRank;

    float4 out = reinterpret_cast<float4*>(y + static_cast<int64_t>(row) * d + col_vec * 4)[0];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #pragma unroll
    for (int k = 0; k < kRank; ++k) {
        float a = arow[k];
        const float4 tv = reinterpret_cast<const float4*>(T + static_cast<int64_t>(k) * d + col_vec * 4)[0];
        acc0 += a * tv.x;
        acc1 += a * tv.y;
        acc2 += a * tv.z;
        acc3 += a * tv.w;
    }

    out.x += acc0;
    out.y += acc1;
    out.z += acc2;
    out.w += acc3;

    reinterpret_cast<float4*>(y + static_cast<int64_t>(row) * d + col_vec * 4)[0] = out;
}

bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

} // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    const int64_t d64 = Wc.size(0);
    const int d = static_cast<int>(d64);

    auto y = at::matmul(Wc, Xc);
    auto t = at::matmul(Bc.transpose(0, 1).contiguous(), Xc);

    TORCH_CHECK(y.is_cuda() && t.is_cuda(), "internal tensors must be CUDA");
    TORCH_CHECK(y.is_contiguous() && t.is_contiguous() && Ac.is_contiguous(),
                "internal tensors must be contiguous");
    TORCH_CHECK(y.scalar_type() == at::kFloat && t.scalar_type() == at::kFloat,
                "internal tensors must be float32");
    TORCH_CHECK(t.size(0) == kRank && t.size(1) == d64,
                "internal tensor t must have shape 16 x d");

    float* y_ptr = y.data_ptr<float>();
    const float* a_ptr = Ac.data_ptr<float>();
    const float* t_ptr = t.data_ptr<float>();

    auto stream = at::cuda::getDefaultCUDAStream(W.device().index()).stream();
    stream = at::cuda::getDefaultCUDAStream(W.device().index()).stream();
    stream = at::cuda::getCurrentCUDAStream(W.device().index()).stream();

    const int64_t total = d64 * d64;

    if ((d % 4) == 0 && is_aligned_16(y_ptr) && is_aligned_16(t_ptr)) {
        const int d_vec = d / 4;
        const int64_t total_vec = d64 * d_vec;
        const int blocks = static_cast<int>((total_vec + kBlockSize - 1) / kBlockSize);
        rank16_add_vec4_kernel<<<blocks, kBlockSize, 0, stream>>>(y_ptr, a_ptr, t_ptr, d_vec, d);
    } else {
        const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
        rank16_add_scalar_kernel<<<blocks, kBlockSize, 0, stream>>>(y_ptr, a_ptr, t_ptr, d);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}