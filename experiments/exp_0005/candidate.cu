#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define LORA_BLOCK_SIZE 256
#define LORA_VECTOR_WIDTH 4

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

__global__ void rank16_add_scalar_kernel(float* __restrict__ y,
                                         const float* __restrict__ A,
                                         const float* __restrict__ T,
                                         int64_t d) {
    const int64_t total = d * d;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const int64_t row = idx / d;
        const int64_t col = idx - row * d;
        const float* a = A + row * 16;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc = fmaf(a[k], T[static_cast<int64_t>(k) * d + col], acc);
        }
        y[idx] += acc;
    }
}

__global__ void rank16_add_vec4_kernel(float* __restrict__ y,
                                       const float* __restrict__ A,
                                       const float* __restrict__ T,
                                       int64_t d) {
    const int64_t vec_cols = d / 4;
    const int64_t total = d * vec_cols;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const int64_t row = idx / vec_cols;
        const int64_t col = (idx - row * vec_cols) * 4;
        const float* a = A + row * 16;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float aval = a[k];
            const float4 tv = *reinterpret_cast<const float4*>(T + static_cast<int64_t>(k) * d + col);
            acc.x = fmaf(aval, tv.x, acc.x);
            acc.y = fmaf(aval, tv.y, acc.y);
            acc.z = fmaf(aval, tv.z, acc.z);
            acc.w = fmaf(aval, tv.w, acc.w);
        }
        float4 yv = *reinterpret_cast<float4*>(y + row * d + col);
        yv.x += acc.x;
        yv.y += acc.y;
        yv.z += acc.z;
        yv.w += acc.w;
        *reinterpret_cast<float4*>(y + row * d + col) = yv;
    }
}

void launch_rank16_add(torch::Tensor& y, const torch::Tensor& A, const torch::Tensor& T, int64_t d) {
    const int threads = LORA_BLOCK_SIZE;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (LORA_VECTOR_WIDTH == 4 && (d % 4 == 0)) {
        const int64_t total = d * (d / 4);
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        rank16_add_vec4_kernel<<<blocks, threads, 0, stream>>>(
            y.data_ptr<float>(), A.data_ptr<float>(), T.data_ptr<float>(), d);
    } else {
        const int64_t total = d * d;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        rank16_add_scalar_kernel<<<blocks, threads, 0, stream>>>(
            y.data_ptr<float>(), A.data_ptr<float>(), T.data_ptr<float>(), d);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);
    const int64_t d = W.size(0);
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    auto y = at::matmul(Wc, Xc);
    auto t = at::matmul(Bc.transpose(0, 1).contiguous(), Xc);
    launch_rank16_add(y, Ac, t, d);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}
