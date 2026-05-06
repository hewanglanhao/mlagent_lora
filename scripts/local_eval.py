from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]


def parse_sizes(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_inputs(input_dir: Path):
    W = torch.load(input_dir / "W.pt", map_location="cpu").contiguous().cuda()
    X = torch.load(input_dir / "X.pt", map_location="cpu").contiguous().cuda()
    A = torch.load(input_dir / "A.pt", map_location="cpu").contiguous().cuda()
    B = torch.load(input_dir / "B.pt", map_location="cpu").contiguous().cuda()
    return W, X, A, B


def make_inputs(d: int, seed: int, scale: float = 1.0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    W = torch.randn(d, d, device="cuda", dtype=torch.float32) * scale
    X = torch.randn(d, d, device="cuda", dtype=torch.float32) * scale
    A = torch.randn(d, 16, device="cuda", dtype=torch.float32) * scale
    B = torch.randn(d, 16, device="cuda", dtype=torch.float32) * scale
    return W, X, A, B


def reference_impl(W, X, A, B):
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def build_module(cu_path: Path, build_dir: Path):
    build_dir.mkdir(parents=True, exist_ok=True)
    name = f"local_eval_lora_ext_{int(time.time() * 1000)}"
    return load(
        name=name,
        sources=[str(cu_path)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        build_directory=str(build_dir),
    )


def check_correctness(y, y_ref) -> dict[str, Any]:
    diff = (y - y_ref).float()
    max_abs_err = diff.abs().max().item()
    rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = bool(torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4))
    finite = bool(torch.isfinite(y).all().item())
    return {
        "passed": passed and finite,
        "finite": finite,
        "max_abs_err": max_abs_err,
        "rel_l2_err": rel_l2_err,
    }


def benchmark(fn, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    times.sort()
    return {
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "p25_ms": float(times[len(times) // 4]),
        "p75_ms": float(times[(len(times) * 3) // 4]),
        "iters": iters,
        "warmup": warmup,
    }


def evaluate_case(module, W, X, A, B, warmup: int, iters: int, run_benchmark: bool) -> dict[str, Any]:
    with torch.no_grad():
        y_student = module.forward(W, X, A, B)
        y_ref = reference_impl(W, X, A, B)
        torch.cuda.synchronize()
    correctness = check_correctness(y_student, y_ref)
    result: dict[str, Any] = {"correctness": correctness}

    if correctness["passed"] and run_benchmark:
        student = benchmark(lambda: module.forward(W, X, A, B), warmup=warmup, iters=iters)
        torch_ref = benchmark(lambda: reference_impl(W, X, A, B), warmup=warmup, iters=iters)
        result["student"] = student
        result["torch_reference"] = torch_ref
        result["speedup"] = torch_ref["median_ms"] / student["median_ms"] if student["median_ms"] > 0 else 0.0
    else:
        result["speedup"] = 0.0
    return result


def evaluate_synthetic(args, module) -> dict[str, Any]:
    results: dict[str, Any] = {}
    scales = [1.0, 0.1, 2.0]
    for idx, d in enumerate(args.sizes):
        W, X, A, B = make_inputs(d, seed=args.seed + idx, scale=scales[idx % len(scales)])
        case_result = evaluate_case(
            module,
            W,
            X,
            A,
            B,
            warmup=args.warmup,
            iters=args.iters,
            run_benchmark=not args.correctness_only,
        )
        results[str(d)] = case_result
        del W, X, A, B
        torch.cuda.empty_cache()
    return results


def evaluate_input_dir(args, module) -> dict[str, Any]:
    W, X, A, B = load_inputs(args.input_dir)
    d = int(W.size(0))
    result = evaluate_case(
        module,
        W,
        X,
        A,
        B,
        warmup=args.warmup,
        iters=args.iters,
        run_benchmark=not args.correctness_only,
    )
    del W, X, A, B
    torch.cuda.empty_cache()
    return {str(d): result}


def summarize(results: dict[str, Any]) -> dict[str, Any]:
    correct = all(item["correctness"]["passed"] for item in results.values())
    speedups = [item.get("speedup", 0.0) for item in results.values() if item["correctness"]["passed"]]
    return {
        "correct": correct,
        "aggregate_speedup": float(statistics.mean(speedups)) if speedups else 0.0,
        "min_speedup": float(min(speedups)) if speedups else 0.0,
        "max_speedup": float(max(speedups)) if speedups else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local evaluator for optimized_lora.cu")
    parser.add_argument("--cu-path", type=Path, default=ROOT / "optimized_lora.cu")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("3584,4096,4608"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "eval_results" / "local_eval.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; local evaluation requires a CUDA GPU.")
    if not args.cu_path.exists():
        raise FileNotFoundError(args.cu_path)

    module = build_module(args.cu_path, build_dir=ROOT / "build" / "local_eval")
    if args.input_dir is not None:
        results = evaluate_input_dir(args, module)
    else:
        results = evaluate_synthetic(args, module)

    report = {
        "cu_path": str(args.cu_path),
        "input_dir": str(args.input_dir) if args.input_dir else None,
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

