#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    c10::cuda::CUDAGuard device_guard(W.device());

    const int d = static_cast<int>(W.size(0));
    constexpr int r = 16;

    auto Y = torch::empty_like(W);
    auto U = torch::empty({static_cast<int64_t>(d), static_cast<int64_t>(r)}, W.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const float one = 1.0f;
    const float zero = 0.0f;

    // Y_col = X_col * W_col
    (void)cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d, d, d,
        &one,
        X.data_ptr<float>(), d,
        W.data_ptr<float>(), d,
        &zero,
        Y.data_ptr<float>(), d
    );

    // U_col[d,16] = X_col * B_col^T
    (void)cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        d, r, d,
        &one,
        X.data_ptr<float>(), d,
        B.data_ptr<float>(), r,
        &zero,
        U.data_ptr<float>(), d
    );

    // Y_col += U_col * A_col
    (void)cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d, d, r,
        &one,
        U.data_ptr<float>(), d,
        A.data_ptr<float>(), r,
        &one,
        Y.data_ptr<float>(), d
    );

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}