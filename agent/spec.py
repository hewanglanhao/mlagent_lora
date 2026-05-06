from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoraSpec:
    rank: int = 16
    min_d: int = 3584
    max_d: int = 4608
    dtype: str = "float32"
    entrypoint: str = "forward"
    final_filename: str = "optimized_lora.cu"

    @property
    def correctness_sizes(self) -> tuple[int, ...]:
        return (3584, 4096, 4608)

    @property
    def benchmark_sizes(self) -> tuple[int, ...]:
        return (3584, 4096, 4608)

    def as_prompt_context(self) -> str:
        return (
            "Optimize Y = W @ X + A @ (B.T @ X). "
            f"W and X are d x d, A and B are d x {self.rank}, "
            f"d is runtime-selected in [{self.min_d}, {self.max_d}], "
            "all tensors are CUDA float32, and the exported function is "
            "torch::Tensor forward(torch::Tensor W, torch::Tensor X, "
            "torch::Tensor A, torch::Tensor B)."
        )


DEFAULT_SPEC = LoraSpec()

