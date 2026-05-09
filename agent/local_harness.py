from __future__ import annotations

import importlib.util
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompileResult:
    compiled: bool
    skipped: bool = False
    error_type: str | None = None
    error_summary: str | None = None
    compile_time_sec: float = 0.0
    module_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiled": self.compiled,
            "skipped": self.skipped,
            "error_type": self.error_type,
            "error_summary": self.error_summary,
            "compile_time_sec": self.compile_time_sec,
            "module_name": self.module_name,
        }


@dataclass
class CorrectnessResult:
    correct: bool
    max_abs_err: float | None = None
    rel_l2_err: float | None = None
    failed_shapes: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "max_abs_err": self.max_abs_err,
            "rel_l2_err": self.rel_l2_err,
            "failed_shapes": self.failed_shapes,
            "details": self.details,
        }


@dataclass
class BenchmarkResult:
    latency_ms: dict[str, float] = field(default_factory=dict)
    torch_ms: dict[str, float] = field(default_factory=dict)
    speedup: dict[str, float] = field(default_factory=dict)
    aggregate_speedup: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "torch_ms": self.torch_ms,
            "speedup": self.speedup,
            "aggregate_speedup": self.aggregate_speedup,
            "details": self.details,
        }


class SyntheticDataGenerator:
    """Generates public-range synthetic tensors for local search."""

    def __init__(self, rank: int = 16, seed: int = 2026) -> None:
        self.rank = rank
        self.seed = seed

    def make_inputs(self, torch, d: int, case_index: int = 0, scale: float = 1.0):
        torch.manual_seed(self.seed + case_index)
        torch.cuda.manual_seed_all(self.seed + case_index)
        W = torch.randn(d, d, device="cuda", dtype=torch.float32) * scale
        X = torch.randn(d, d, device="cuda", dtype=torch.float32) * scale
        A = torch.randn(d, self.rank, device="cuda", dtype=torch.float32) * scale
        B = torch.randn(d, self.rank, device="cuda", dtype=torch.float32) * scale
        return W, X, A, B

    def reference(self, W, X, A, B):
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


class EnvironmentInspectorAgent:
    def inspect(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "nvcc": shutil.which("nvcc"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "torch_importable": importlib.util.find_spec("torch") is not None,
        }
        if not info["torch_importable"]:
            info["cuda_available"] = False
            info["reason"] = "torch is not importable"
            return info

        try:
            import torch

            info.update(
                {
                    "torch_version": torch.__version__,
                    "torch_cuda": getattr(torch.version, "cuda", None),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            )
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info.update(
                    {
                        "gpu_name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "sm_count": props.multi_processor_count,
                        "total_memory_bytes": props.total_memory,
                        "max_threads_per_block": getattr(props, "max_threads_per_block", None),
                        "shared_memory_per_block": getattr(props, "shared_memory_per_block", None),
                        "shared_memory_per_multiprocessor": getattr(
                            props, "shared_memory_per_multiprocessor", None
                        ),
                        "max_threads_per_multiprocessor": getattr(
                            props, "max_threads_per_multi_processor", None
                        ),
                        "candidate_block_sizes": [128, 256, 512],
                        "candidate_vector_widths": [1, 4],
                        "candidate_tile_m": [16, 32, 64],
                        "candidate_tile_n": [16, 32, 64],
                        "candidate_tile_k": [8, 16, 32],
                        "rank16_update_supported": True,
                    }
                )
        except Exception as exc:
            info["cuda_available"] = False
            info["reason"] = f"{type(exc).__name__}: {exc}"
        return info

    def can_run_cuda_tests(self, info: dict[str, Any]) -> tuple[bool, str]:
        if not info.get("torch_importable"):
            return False, "torch is not importable"
        if not info.get("cuda_available"):
            return False, "torch.cuda.is_available() is false"
        if not info.get("nvcc"):
            return False, "nvcc was not found in PATH"
        return True, "CUDA test environment is available"


class BuildCompileAgent:
    def compile(self, source: Path, build_dir: Path, experiment_id: int, use_fast_math: bool = False):
        start = time.perf_counter()
        module_name = f"lora_candidate_ext_{experiment_id}_{int(time.time() * 1000)}"
        try:
            import torch
            from torch.utils.cpp_extension import load

            build_dir.mkdir(parents=True, exist_ok=True)
            extra_cuda_cflags = ["-O3"]
            if use_fast_math:
                extra_cuda_cflags.append("--use_fast_math")
            module = load(
                name=module_name,
                sources=[str(source)],
                verbose=False,
                extra_cuda_cflags=extra_cuda_cflags,
                with_cuda=True,
                build_directory=str(build_dir),
            )
            elapsed = time.perf_counter() - start
            return CompileResult(True, compile_time_sec=elapsed, module_name=module_name), module
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return (
                CompileResult(
                    False,
                    error_type=self._classify_error(str(exc)),
                    error_summary=self._trim_error(str(exc)),
                    compile_time_sec=elapsed,
                    module_name=module_name,
                ),
                None,
            )

    def _classify_error(self, text: str) -> str:
        lowered = text.lower()
        if "timeout" in lowered:
            return "compile_timeout"
        if "pybind" in lowered or "forward" in lowered:
            return "pybind_or_signature_error"
        if "cuda" in lowered or "nvcc" in lowered:
            return "cuda_compile_error"
        if "syntax" in lowered or "expected" in lowered:
            return "syntax_error"
        return "compile_error"

    def _trim_error(self, text: str, limit: int = 5000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...<trimmed>..."


class CorrectnessTestAgent:
    def __init__(
        self,
        sizes: tuple[int, ...],
        seed: int = 2026,
        rtol: float = 1e-4,
        atol: float = 2e-3,
    ) -> None:
        self.sizes = sizes
        self.rtol = rtol
        self.atol = atol
        self.data = SyntheticDataGenerator(seed=seed)

    def run(self, module) -> CorrectnessResult:
        import torch

        failed: list[int] = []
        max_abs = 0.0
        max_rel_l2 = 0.0
        details: dict[str, Any] = {}
        for idx, d in enumerate(self.sizes):
            scale = (1.0, 0.1, 2.0)[idx % 3]
            W, X, A, B = self.data.make_inputs(torch, d, case_index=idx, scale=scale)
            with torch.no_grad():
                y_student = module.forward(W, X, A, B)
                y_ref = self.data.reference(W, X, A, B)
                torch.cuda.synchronize()
                diff = (y_student - y_ref).float()
                shape_max_abs = diff.abs().max().item()
                shape_rel_l2 = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
                passed = bool(torch.allclose(y_student, y_ref, rtol=self.rtol, atol=self.atol))
                finite = bool(torch.isfinite(y_student).all().item())
            max_abs = max(max_abs, shape_max_abs)
            max_rel_l2 = max(max_rel_l2, shape_rel_l2)
            details[str(d)] = {
                "passed": passed,
                "finite": finite,
                "max_abs_err": shape_max_abs,
                "rel_l2_err": shape_rel_l2,
                "rtol": self.rtol,
                "atol": self.atol,
            }
            if not passed or not finite:
                failed.append(d)
            del W, X, A, B, y_student, y_ref, diff
            torch.cuda.empty_cache()
        return CorrectnessResult(
            correct=not failed,
            max_abs_err=max_abs,
            rel_l2_err=max_rel_l2,
            failed_shapes=failed,
            details=details,
        )


class BenchmarkAgent:
    def __init__(
        self,
        sizes: tuple[int, ...],
        warmup: int = 3,
        iters: int = 10,
        seed: int = 2027,
    ) -> None:
        self.sizes = sizes
        self.warmup = warmup
        self.iters = iters
        self.data = SyntheticDataGenerator(seed=seed)

    def run(self, module) -> BenchmarkResult:
        import torch

        latency: dict[str, float] = {}
        torch_latency: dict[str, float] = {}
        speedup: dict[str, float] = {}
        for idx, d in enumerate(self.sizes):
            W, X, A, B = self.data.make_inputs(torch, d, case_index=idx)
            with torch.no_grad():
                candidate_ms = self._benchmark(torch, lambda: module.forward(W, X, A, B))
                reference_ms = self._benchmark(torch, lambda: self.data.reference(W, X, A, B))
            latency[str(d)] = candidate_ms
            torch_latency[str(d)] = reference_ms
            speedup[str(d)] = reference_ms / candidate_ms if candidate_ms > 0 else 0.0
            del W, X, A, B
            torch.cuda.empty_cache()

        aggregate = self._weighted_aggregate(speedup)
        return BenchmarkResult(
            latency_ms=latency,
            torch_ms=torch_latency,
            speedup=speedup,
            aggregate_speedup=aggregate,
            details={"warmup": self.warmup, "iters": self.iters},
        )

    def _benchmark(self, torch, fn) -> float:
        for _ in range(self.warmup):
            _ = fn()
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(self.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        return float(statistics.median(times))

    def _weighted_aggregate(self, speedup: dict[str, float]) -> float:
        if not speedup:
            return 0.0
        weights = {"3584": 0.3, "4096": 0.4, "4608": 0.3}
        total_weight = 0.0
        score = 0.0
        for shape, value in speedup.items():
            weight = weights.get(shape, 1.0)
            score += weight * value
            total_weight += weight
        return score / total_weight if total_weight else 0.0


class ProfilingAgent:
    def __init__(self, enable_torch_profile: bool = False, sample_d: int = 3584) -> None:
        self.enable_torch_profile = enable_torch_profile
        self.sample_d = sample_d
        self.data = SyntheticDataGenerator(seed=2028)

    def profile(self, candidate_family: str, benchmark: BenchmarkResult | None, module=None) -> dict[str, Any]:
        result = self._lightweight_profile(candidate_family, benchmark)
        if self.enable_torch_profile and module is not None:
            result["torch_profiler"] = self._torch_profile(module)
        return result

    def _lightweight_profile(self, candidate_family: str, benchmark: BenchmarkResult | None) -> dict[str, Any]:
        if benchmark is None or benchmark.aggregate_speedup is None:
            return {
                "bottleneck": "unknown",
                "evidence": "benchmark unavailable",
                "suggested_actions": ["keep the last known-good candidate"],
            }
        if candidate_family == "aten_reference":
            return {
                "bottleneck": "multiple_aten_gemm_and_add_launches",
                "evidence": benchmark.to_dict(),
                "suggested_actions": [
                    "keep cuBLAS for W @ X",
                    "fuse rank-16 update into the output tensor",
                    "test scalar and vectorized rank-16 kernels",
                ],
            }
        return {
            "bottleneck": "rank16_update_or_memory_bandwidth",
            "evidence": benchmark.to_dict(),
            "suggested_actions": [
                "vary block size",
                "try float4 stores when d is divisible by 4",
                "fall back to ATen if numerical error appears",
            ],
        }

    def _torch_profile(self, module) -> dict[str, Any]:
        try:
            import torch

            W, X, A, B = self.data.make_inputs(torch, self.sample_d)
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.no_grad():
                for _ in range(2):
                    _ = module.forward(W, X, A, B)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
                    _ = module.forward(W, X, A, B)
                    torch.cuda.synchronize()
            events = []
            for item in prof.key_averages()[:20]:
                events.append(
                    {
                        "key": item.key,
                        "cpu_time_total_us": item.cpu_time_total,
                        "cuda_time_total_us": getattr(item, "cuda_time_total", 0.0),
                        "count": item.count,
                    }
                )
            del W, X, A, B
            torch.cuda.empty_cache()
            return {"enabled": True, "sample_d": self.sample_d, "events": events}
        except Exception as exc:
            return {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}
