from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .memory import atomic_write_text


def write_output_report(
    *,
    output_path: Path,
    run_id: str,
    constraints: dict[str, Any],
    llm_status: str,
    pipeline: dict[str, Any],
    stop_reason: str,
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
    final_path: Path,
    experiments_dir: Path,
) -> None:
    """Write the single evaluator-facing markdown report for the run."""
    sections = [
        "# LoRA Operator Optimization Report",
        _run_summary(
            run_id=run_id,
            stop_reason=stop_reason,
            llm_status=llm_status,
            pipeline=pipeline,
            final_path=final_path,
            best=best,
        ),
        _constraints_section(constraints),
        _architecture_section(pipeline),
        _candidate_lifecycle_section(history, best),
        _search_process_section(history),
        _candidate_table(history),
        _best_section(best),
        _llm_section(experiments_dir),
        _environment_section(experiments_dir),
        _artifacts_section(experiments_dir, final_path),
    ]
    atomic_write_text(output_path, "\n\n".join(section for section in sections if section).rstrip() + "\n")


def _run_summary(
    *,
    run_id: str,
    stop_reason: str,
    llm_status: str,
    pipeline: dict[str, Any],
    final_path: Path,
    best: dict[str, Any] | None,
) -> str:
    lines = [
        "## Run Summary",
        f"- Run ID: `{run_id}`",
        f"- Stop reason: `{stop_reason}`",
        f"- LLM: {_escape_inline(llm_status)}",
        f"- Pipeline: `{json.dumps(pipeline, sort_keys=True)}`",
        f"- Final file: `{final_path}`",
    ]
    if final_path.exists():
        lines.append(f"- Final SHA256: `{_sha256_file(final_path)}`")
    if best:
        lines.extend(
            [
                f"- Best candidate: `{best.get('name')}` (`id={best.get('id')}`)",
                f"- Best decision: `{best.get('decision')}`",
                f"- Best aggregate speedup: `{_fmt(best.get('aggregate_speedup'))}`",
                f"- Best source: `{best.get('source_path')}`",
            ]
        )
    else:
        lines.append("- Best candidate: not available in this run record")
    lines.append(f"- Report generated UTC: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`")
    return "\n".join(lines)


def _constraints_section(constraints: dict[str, Any]) -> str:
    if not constraints:
        return ""
    return "\n".join(
        [
            "## Optimization Target",
            f"- Operator: `{constraints.get('operator', 'unknown')}`",
            f"- Rank: `{constraints.get('rank', 'unknown')}`",
            f"- d range: `{constraints.get('d_range', 'unknown')}`",
            f"- dtype: `{constraints.get('dtype', 'unknown')}`",
            f"- Required entrypoint: `{constraints.get('entrypoint', 'forward')}`",
            f"- Final filename: `{constraints.get('final_file', 'optimized_lora.cu')}`",
        ]
    )


def _architecture_section(pipeline: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## System Architecture",
            "This optimizer is organized as a supervisor-driven multi-agent system. The Supervisor owns the run budget, experiment memory, best-candidate state, and final artifact synchronization. Specialized agents then handle the optimization loop around the LoRA CUDA operator.",
            "",
            "- SearchSpacePlannerAgent proposes the next optimization family or strategy from prior evidence.",
            "- OptimizationMutationAgent turns benchmark, correctness, and diagnosis feedback into a concrete mutation request.",
            "- LLMCUDACandidateGeneratorAgent or the deterministic CUDACandidateGenerator writes a self-contained `candidate.cu` implementation for `forward(W, X, A, B)`.",
            "- StaticCodeReviewAgent and LLMStaticCodeReviewAgent inspect the generated CUDA/C++ before expensive execution; LLM warnings are recorded in the promotion evidence.",
            "- BuildCompileAgent compiles each candidate as a PyTorch CUDA extension with `torch.utils.cpp_extension.load` and `-O3`, matching the expected official harness style.",
            "- CorrectnessTestAgent executes `module.forward` and compares against the PyTorch reference `W @ X + A @ (B.T.contiguous() @ X)` before any benchmark-based promotion.",
            "- BenchmarkAgent times the candidate and the PyTorch reference with CUDA events, then computes per-shape speedup as `torch_ms / candidate_ms`.",
            "- ProfilingAgent and PerformanceDiagnosisAgent summarize bottlenecks and feed the next LLM mutation, closing the generate -> compile -> test -> benchmark -> diagnose -> regenerate loop.",
            "- BestCandidateManager promotes only validated candidates and writes the final `optimized_lora.cu` used by external evaluation.",
            "",
            f"Pipeline settings for this run: `{json.dumps(pipeline, sort_keys=True)}`",
        ]
    )


def _candidate_lifecycle_section(history: list[dict[str, Any]], best: dict[str, Any] | None) -> str:
    candidate = _representative_candidate(history, best)
    lines = [
        "## Candidate Lifecycle",
        "A CUDA candidate flows through the system as follows:",
        "",
        "1. Strategy selection: the planner or mutation agent chooses an optimization direction using the current best record, failed candidates, benchmark evidence, and diagnosis feedback.",
        "2. CUDA generation: the generator writes a complete `candidate.cu` file under the current experiment directory.",
        "3. Static review: local rule checks and optional LLM review flag unsafe APIs, signature mismatches, layout risks, and likely correctness problems.",
        "4. Compilation: the candidate is compiled into a PyTorch CUDA extension. Compilation failures are classified and can trigger LLM repair attempts.",
        "5. Correctness execution: the compiled `forward(W, X, A, B)` is run on CUDA tensors and compared to the PyTorch reference before any timing result is trusted.",
        "6. Benchmarking: only correct candidates are benchmarked against the same PyTorch reference implementation. Speedup is computed as reference median latency divided by candidate median latency.",
        "7. Profiling and diagnosis: timing data, correctness errors, and optional profiler evidence are converted into bottleneck explanations and next-step recommendations.",
        "8. Promotion: a candidate replaces the current best only when it compiles, passes correctness, and improves the promotion metric. The chosen source is copied to `optimized_lora.cu`.",
    ]
    if candidate:
        lines.extend(["", "Representative candidate from this run:"])
        lines.extend(_candidate_flow_lines(candidate))
    return "\n".join(lines)


def _search_process_section(history: list[dict[str, Any]]) -> str:
    total = len(history)
    compiled = sum(1 for item in history if item.get("compiled"))
    correct = sum(1 for item in history if item.get("correct"))
    promoted = sum(1 for item in history if item.get("decision") == "promoted_to_best")
    origins = Counter(str(item.get("generation", {}).get("origin", "unknown")) for item in history)
    decisions = Counter(str(item.get("decision", "unknown")) for item in history)
    lines = [
        "## Search Process",
        f"- Candidates generated: `{total}`",
        f"- Candidates compiled: `{compiled}`",
        f"- Candidates correct: `{correct}`",
        f"- Candidates promoted: `{promoted}`",
        f"- Generation origins: `{dict(origins)}`",
        f"- Decisions: `{dict(decisions)}`",
    ]
    return "\n".join(lines)


def _candidate_table(history: list[dict[str, Any]]) -> str:
    if not history:
        return "## Candidate Comparison\n\nNo candidate records were written."
    lines = [
        "## Candidate Comparison",
        "| id | name | origin | compiled | correct | decision | aggregate speedup | per-shape speedup |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(history, key=lambda item: item.get("id", item.get("experiment_id", 0)))
    for item in ordered:
        benchmark = item.get("benchmark") or {}
        speedup = benchmark.get("speedup") or {}
        origin = (item.get("generation") or {}).get("origin", "unknown")
        lines.append(
            "| {id} | {name} | {origin} | {compiled} | {correct} | {decision} | {agg} | {speedup} |".format(
                id=_cell(item.get("id")),
                name=_cell(item.get("name")),
                origin=_cell(origin),
                compiled=_cell(item.get("compiled")),
                correct=_cell(item.get("correct")),
                decision=_cell(item.get("decision")),
                agg=_cell(_fmt(item.get("aggregate_speedup"))),
                speedup=_cell(_shape_map(speedup)),
            )
        )
    return "\n".join(lines)


def _best_section(best: dict[str, Any] | None) -> str:
    if not best:
        return "## Best Candidate Details\n\nNo best candidate was selected in this run."
    benchmark = best.get("benchmark") or {}
    profile = best.get("profile") or {}
    diagnosis = best.get("diagnosis") or {}
    llm_review = best.get("llm_static_review")
    lines = [
        "## Best Candidate Details",
        f"- Name: `{best.get('name')}`",
        f"- Family: `{best.get('family')}`",
        f"- Block size: `{best.get('block_size')}`",
        f"- Vector width: `{best.get('vector_width')}`",
        f"- Shape dispatch: `{best.get('shape_dispatch')}`",
        f"- LLM generated source: `{best.get('llm_generated')}`",
        f"- Generation origin: `{(best.get('generation') or {}).get('origin', 'unknown')}`",
        f"- Correctness: `compiled={best.get('compiled')}, correct={best.get('correct')}`",
        f"- Latency ms: `{_shape_map(benchmark.get('latency_ms') or best.get('latency_by_d') or {})}`",
        f"- Torch reference ms: `{_shape_map(benchmark.get('torch_ms') or {})}`",
        f"- Speedup: `{_shape_map(benchmark.get('speedup') or {})}`",
        f"- Profile bottleneck: `{profile.get('bottleneck', 'unknown')}`",
        f"- Diagnosis bottleneck: `{diagnosis.get('bottleneck', 'unknown')}`",
        f"- Diagnosis promotion advice: `{diagnosis.get('promotion_advice', 'unknown')}`",
    ]
    recommended = diagnosis.get("recommended_actions") or profile.get("suggested_actions") or []
    if recommended:
        lines.append("- Recommended next actions:")
        lines.extend(f"  - {_escape_text(str(item))}" for item in recommended[:8])
    if isinstance(llm_review, dict) and llm_review.get("pass") is False:
        lines.append("- Strong warning: LLM static review returned `pass=false`.")
        for err in (llm_review.get("errors") or [])[:5]:
            lines.append(f"  - {_escape_text(str(err))}")
    return "\n".join(lines)


def _llm_section(experiments_dir: Path) -> str:
    manifest = experiments_dir / "llm_calls" / "manifest.jsonl"
    if not manifest.exists():
        return "## LLM Call Summary\n\nNo LLM call manifest was found."
    status_counts: Counter[str] = Counter()
    by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    durations: list[float] = []
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("event") == "parse":
            continue
        status = str(item.get("status", "started"))
        agent = str(item.get("agent", "unknown"))
        status_counts[status] += 1
        by_agent[agent][status] += 1
        if item.get("error"):
            errors[_short_error(str(item["error"]))] += 1
        if isinstance(item.get("duration_sec"), (int, float)):
            durations.append(float(item["duration_sec"]))
    lines = [
        "## LLM Call Summary",
        f"- Status counts: `{dict(status_counts)}`",
        f"- Calls by agent: `{ {agent: dict(counts) for agent, counts in sorted(by_agent.items())} }`",
    ]
    if durations:
        lines.append(f"- Total LLM wall time recorded: `{sum(durations):.2f}s`")
        lines.append(f"- Max single LLM call duration: `{max(durations):.2f}s`")
    if errors:
        lines.append(f"- Top errors: `{dict(errors.most_common(5))}`")
    return "\n".join(lines)


def _environment_section(experiments_dir: Path) -> str:
    path = experiments_dir / "environment.json"
    if not path.exists():
        return ""
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    keys = ["gpu_name", "compute_capability", "torch_version", "torch_cuda", "cuda_available", "nvcc"]
    lines = ["## Environment"]
    for key in keys:
        if key in env:
            lines.append(f"- {key}: `{env[key]}`")
    return "\n".join(lines)


def _artifacts_section(experiments_dir: Path, final_path: Path) -> str:
    return "\n".join(
        [
            "## Artifacts",
            f"- Experiments directory: `{experiments_dir}`",
            f"- Leaderboard: `{experiments_dir / 'leaderboard.json'}`",
            f"- Agent trace: `{experiments_dir / 'agent_trace.jsonl'}`",
            f"- LLM calls: `{experiments_dir / 'llm_calls'}`",
            f"- Final CUDA source: `{final_path}`",
        ]
    )


def _representative_candidate(
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if best:
        best_id = best.get("id")
        for item in history:
            if item.get("id") == best_id:
                return item
        return best
    promoted = [item for item in history if item.get("decision") == "promoted_to_best"]
    if promoted:
        return promoted[-1]
    compiled = [item for item in history if item.get("compiled")]
    if compiled:
        return compiled[-1]
    return history[-1] if history else None


def _candidate_flow_lines(candidate: dict[str, Any]) -> list[str]:
    generation = candidate.get("generation") or {}
    compile_result = candidate.get("compile") or {}
    correctness = candidate.get("correctness") or {}
    benchmark = candidate.get("benchmark") or {}
    profile = candidate.get("profile") or {}
    diagnosis = candidate.get("diagnosis") or {}
    static_review = candidate.get("static_review") or {}
    llm_review = candidate.get("llm_static_review")
    lines = [
        f"- Candidate: `{candidate.get('name')}` (`id={candidate.get('id')}`, family=`{candidate.get('family')}`)",
        f"- Generation: origin=`{generation.get('origin', 'unknown')}`, llm_generated=`{candidate.get('llm_generated')}`, parent=`{candidate.get('parent')}`",
        f"- Source path: `{candidate.get('source_path', 'unknown')}`",
        f"- Static review: passed=`{static_review.get('passed', 'unknown')}`, risk=`{static_review.get('risk_level', 'unknown')}`",
    ]
    if isinstance(llm_review, dict):
        lines.append(
            f"- LLM static review: pass=`{llm_review.get('pass', 'unknown')}`, risk=`{llm_review.get('risk_level', 'unknown')}`"
        )
    else:
        lines.append("- LLM static review: not recorded for this candidate")
    lines.extend(
        [
            f"- Compile: compiled=`{compile_result.get('compiled', candidate.get('compiled'))}`, time=`{_fmt(compile_result.get('compile_time_sec'))}s`, error=`{compile_result.get('error_type')}`",
            f"- Correctness: correct=`{correctness.get('correct', candidate.get('correct'))}`, max_abs=`{_fmt(correctness.get('max_abs_err'))}`, rel_l2=`{_fmt(correctness.get('rel_l2_err'))}`, failed_shapes=`{correctness.get('failed_shapes', [])}`",
            f"- Benchmark latency: `{_shape_map(benchmark.get('latency_ms') or candidate.get('latency_by_d') or {})}`",
            f"- Benchmark speedup: `{_shape_map(benchmark.get('speedup') or {})}`, aggregate=`{_fmt(candidate.get('aggregate_speedup') or benchmark.get('aggregate_speedup'))}`",
            f"- Profile bottleneck: `{profile.get('bottleneck', 'unknown')}`",
            f"- Diagnosis: bottleneck=`{diagnosis.get('bottleneck', 'unknown')}`, promotion_advice=`{diagnosis.get('promotion_advice', 'unknown')}`",
            f"- Final decision: `{candidate.get('decision', 'unknown')}`",
        ]
    )
    return lines


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "n/a"
    return str(value)


def _shape_map(value: dict[str, Any]) -> str:
    if not value:
        return "n/a"
    return ", ".join(f"{key}:{_fmt(val)}" for key, val in sorted(value.items()))


def _cell(value: Any) -> str:
    return _escape_text(str(value)).replace("\n", " ")


def _escape_text(text: str) -> str:
    return text.replace("|", "\\|")


def _escape_inline(text: str) -> str:
    return text.replace("`", "'")


def _short_error(text: str) -> str:
    first = text.splitlines()[0] if text else "unknown"
    if len(first) > 180:
        return first[:177] + "..."
    return first
