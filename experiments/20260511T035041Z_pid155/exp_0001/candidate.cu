#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    const c10::cuda::CUDAGuard device_guard(W.device());

    const int d = static_cast<int>(W.size(0));
    auto Y = torch::empty_like(W);
    auto U = torch::empty({static_cast<int64_t>(d), 16}, W.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    // Y_col = X_col * W_col
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, d, d,
                &alpha,
                X.data_ptr<float>(), d,
                W.data_ptr<float>(), d,
                &beta0,
                Y.data_ptr<float>(), d);

    // U_col[d,16] = X_col * B_col^T
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                d, 16, d,
                &alpha,
                X.data_ptr<float>(), d,
                B.data_ptr<float>(), 16,
                &beta0,
                U.data_ptr<float>(), d);

    // Y_col += U_col * A_col
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, d, 16,
                &alpha,
                U.data_ptr<float>(), d,
                A.data_ptr<float>(), 16,
                &beta1,
                Y.data_ptr<float>(), d);

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}