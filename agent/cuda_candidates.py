from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .memory import atomic_write_text


@dataclass(frozen=True)
class CandidateSpec:
    experiment_id: int
    name: str
    family: str
    block_size: int = 256
    vector_width: int = 1
    use_fast_math: bool = False
    shape_dispatch: bool = False
    llm_generated: bool = False
    strategy: str = ""
    parent: int | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "id": self.experiment_id,
            "name": self.name,
            "family": self.family,
            "block_size": self.block_size,
            "vector_width": self.vector_width,
            "use_fast_math": self.use_fast_math,
            "shape_dispatch": self.shape_dispatch,
            "llm_generated": self.llm_generated,
            "strategy": self.strategy,
            "parent": self.parent,
        }


class CUDACandidateGenerator:
    def baseline(self, experiment_id: int = 0) -> CandidateSpec:
        return CandidateSpec(
            experiment_id=experiment_id,
            name="baseline_aten",
            family="aten_reference",
        )

    def default_search_space(self, start_id: int = 1) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        configs = [
            ("rank16_scalar_b128", 128, 1, False),
            ("rank16_scalar_b256", 256, 1, False),
            ("rank16_scalar_b512", 512, 1, False),
            ("rank16_vec4_b128", 128, 4, False),
            ("rank16_vec4_b256", 256, 4, False),
            ("rank16_vec4_b512", 512, 4, False),
            ("rank16_shape_scalar", 256, 1, True),
            ("rank16_shape_vec4", 256, 4, True),
        ]
        for offset, (name, block_size, vector_width, shape_dispatch) in enumerate(configs):
            specs.append(
                CandidateSpec(
                    experiment_id=start_id + offset,
                    name=name,
                    family="cublas_plus_custom_rank16_update",
                    block_size=block_size,
                    vector_width=vector_width,
                    shape_dispatch=shape_dispatch,
                    strategy="deterministic_rank16_update",
                    parent=0,
                )
            )
        return specs

    def candidate_key(self, spec: CandidateSpec) -> tuple[object, ...]:
        return (
            spec.family,
            spec.block_size,
            spec.vector_width,
            spec.use_fast_math,
            spec.shape_dispatch,
        )

    def from_mutation(self, experiment_id: int, mutation: dict[str, object], parent: int | None) -> CandidateSpec:
        params = mutation.get("parameters") if isinstance(mutation.get("parameters"), dict) else {}
        assert isinstance(params, dict)
        family = str(mutation.get("candidate_family") or "cublas_plus_custom_rank16_update")
        block_size = int(params.get("block_size") or mutation.get("block_size") or 256)
        vector_width = int(params.get("vector_width") or mutation.get("vector_width") or 1)
        shape_dispatch = bool(params.get("shape_dispatch") or mutation.get("shape_dispatch") or False)
        use_fast_math = bool(params.get("use_fast_math") or mutation.get("use_fast_math") or False)
        name = str(mutation.get("next_mutation_name") or f"{family}_{experiment_id}")
        return CandidateSpec(
            experiment_id=experiment_id,
            name=name,
            family=family,
            block_size=block_size,
            vector_width=vector_width,
            use_fast_math=use_fast_math,
            shape_dispatch=shape_dispatch,
            strategy=str(mutation.get("expected_benefit") or mutation.get("strategy") or "llm_mutation"),
            parent=parent,
        )

    def next_untried(
        self,
        experiment_id: int,
        history: list[dict[str, object]],
        parent: int | None,
    ) -> CandidateSpec | None:
        tried = {
            (
                item.get("family"),
                item.get("block_size"),
                item.get("vector_width"),
                item.get("use_fast_math"),
                item.get("shape_dispatch"),
            )
            for item in history
        }
        for candidate in self.default_search_space(start_id=experiment_id):
            adjusted = CandidateSpec(
                experiment_id=experiment_id,
                name=candidate.name,
                family=candidate.family,
                block_size=candidate.block_size,
                vector_width=candidate.vector_width,
                use_fast_math=candidate.use_fast_math,
                shape_dispatch=candidate.shape_dispatch,
                strategy=candidate.strategy,
                parent=parent,
            )
            if self.candidate_key(adjusted) not in tried:
                return adjusted
        return None

    def write_candidate(self, spec: CandidateSpec, path: Path) -> None:
        if spec.family == "aten_reference":
            code = self._baseline_code()
        elif spec.family == "cublas_plus_custom_rank16_update":
            code = self._custom_rank16_code(spec)
        else:
            raise ValueError(f"Unknown candidate family: {spec.family}")
        atomic_write_text(path, code)

    def _baseline_code(self) -> str:
        return r'''#include <torch/extension.h>

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

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    check_inputs(W, X, A, B);
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    auto y = at::matmul(Wc, Xc);
    auto t = at::matmul(Bc.transpose(0, 1).contiguous(), Xc);
    y.add_(at::matmul(Ac, t));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward");
}
'''

    def _custom_rank16_code(self, spec: CandidateSpec) -> str:
        template = r'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define LORA_BLOCK_SIZE __BLOCK_SIZE__
#define LORA_VECTOR_WIDTH __VECTOR_WIDTH__
#define LORA_SHAPE_DISPATCH __SHAPE_DISPATCH__

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
    int threads = LORA_BLOCK_SIZE;
    if (LORA_SHAPE_DISPATCH) {
        if (d <= 3840) {
            threads = 128;
        } else if (d <= 4352) {
            threads = 256;
        } else {
            threads = 512;
        }
    }
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
'''
        return (
            template.replace("__BLOCK_SIZE__", str(spec.block_size))
            .replace("__VECTOR_WIDTH__", str(spec.vector_width))
            .replace("__SHAPE_DISPATCH__", "1" if spec.shape_dispatch else "0")
        )
