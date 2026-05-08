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
