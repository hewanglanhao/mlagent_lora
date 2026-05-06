from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cuda_candidates import CUDACandidateGenerator, CandidateSpec
from .llm_adapter import LLMClient
from .llm_call_logger import LLMCallLogger
from .llm_agents import (
    LLMCUDACandidateGeneratorAgent,
    LLMStaticCodeReviewAgent,
    OptimizationMutationAgent,
    PerformanceDiagnosisAgent,
    parse_json_object,
    summarize_code,
)
from .local_harness import (
    BenchmarkAgent,
    BuildCompileAgent,
    CorrectnessTestAgent,
    EnvironmentInspectorAgent,
    ProfilingAgent,
)
from .memory import ExperimentMemory, atomic_copy
from .prompt_library import PromptLibrary
from .runtime_logging import setup_runtime_logger
from .run_context import make_run_id
from .spec import DEFAULT_SPEC, LoraSpec
from .static_review import StaticCodeReviewAgent


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CandidateRecord:
    spec: CandidateSpec
    source_path: Path
    compiled: bool = False
    correct: bool = False
    aggregate_speedup: float | None = None
    latency_by_d: dict[str, float] = field(default_factory=dict)
    decision: str = "not_evaluated"
    generation_result: dict[str, Any] = field(default_factory=dict)
    static_review_result: dict[str, Any] = field(default_factory=dict)
    llm_static_review_result: dict[str, Any] | None = None
    compile_result: dict[str, Any] = field(default_factory=dict)
    correctness_result: dict[str, Any] = field(default_factory=dict)
    benchmark_result: dict[str, Any] = field(default_factory=dict)
    profile_result: dict[str, Any] = field(default_factory=dict)
    diagnosis_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.metadata(),
            "source_path": str(self.source_path),
            "compiled": self.compiled,
            "correct": self.correct,
            "aggregate_speedup": self.aggregate_speedup,
            "latency_by_d": self.latency_by_d,
            "decision": self.decision,
            "generation": self.generation_result,
            "static_review": self.static_review_result,
            "llm_static_review": self.llm_static_review_result,
            "compile": self.compile_result,
            "correctness": self.correctness_result,
            "benchmark": self.benchmark_result,
            "profile": self.profile_result,
            "diagnosis": self.diagnosis_result,
        }


class ConstraintSpecManager:
    def __init__(self, spec: LoraSpec) -> None:
        self.spec = spec

    def checklist(self) -> dict[str, Any]:
        return {
            "operator": "Y = W @ X + A @ (B.T @ X)",
            "rank": self.spec.rank,
            "d_range": [self.spec.min_d, self.spec.max_d],
            "dtype": self.spec.dtype,
            "entrypoint": self.spec.entrypoint,
            "final_file": self.spec.final_filename,
            "single_file": True,
            "dynamic_shape_required": True,
        }


class SearchSpacePlannerAgent:
    def __init__(
        self,
        spec: LoraSpec,
        llm: LLMClient,
        prompts: PromptLibrary,
        llm_calls: LLMCallLogger | None = None,
    ) -> None:
        self.spec = spec
        self.llm = llm
        self.prompts = prompts
        self.llm_calls = llm_calls

    def plan(self, history: list[CandidateRecord]) -> dict[str, Any]:
        base_plan = {
            "strategy": "deterministic_rank16_search",
            "notes": [
                "Use ATen/cuBLAS for the d x d GEMM.",
                "Explore kernels that fuse Y += A @ T for rank 16.",
                "Validate every candidate before promotion.",
            ],
        }
        llm_response = self._ask_llm(history)
        if llm_response:
            parsed = parse_json_object(llm_response.content)
            if self.llm_calls:
                self.llm_calls.record_parse(
                    llm_response.call_id,
                    "json_object",
                    parsed is not None,
                    "" if parsed is not None else "search plan JSON parse failed",
                )
            base_plan["llm_advice"] = parsed if parsed is not None else llm_response.content
            if parsed and parsed.get("strategy"):
                base_plan["strategy"] = str(parsed["strategy"])
        return base_plan

    def _ask_llm(self, history: list[CandidateRecord]):
        if not self.llm.enabled:
            return None
        compact_history = [record.to_dict() for record in history[-5:]]
        prompt = self.prompts.pair("search_space_planner")
        user = self.prompts.render(
            prompt.user,
            {
                "SPEC_CONTEXT": self.spec.as_prompt_context(),
                "HISTORY_JSON": json.dumps(compact_history, indent=2, sort_keys=True),
            },
        )
        if self.llm_calls:
            return self.llm_calls.complete(
                self.llm,
                "search_space_planner",
                prompt.system,
                user,
                {"history_count": len(history)},
            )
        return self.llm.complete(system=prompt.system, user=user)


class BestCandidateManager:
    def __init__(self, root: Path, spec: LoraSpec) -> None:
        self.root = root
        self.final_path = root / spec.final_filename
        self.backup_path = root / "optimized_lora.best.cu"
        self.last_good_path = root / "optimized_lora.last_good.cu"
        self.best_record: CandidateRecord | None = None

    def bootstrap(self, source: Path, record: CandidateRecord) -> None:
        atomic_copy(source, self.final_path)
        atomic_copy(source, self.backup_path)
        atomic_copy(source, self.last_good_path)
        record.decision = "bootstrapped_as_initial_best"
        self.best_record = record

    def consider(self, source: Path, record: CandidateRecord) -> bool:
        if not record.compiled or not record.correct:
            if record.decision in {"not_evaluated", "evaluated_valid"}:
                record.decision = "rejected_invalid"
            return False
        if self.best_record is None:
            should_promote = True
        elif self.best_record.aggregate_speedup is None:
            should_promote = True
        elif record.aggregate_speedup is None:
            should_promote = False
        else:
            should_promote = record.aggregate_speedup > self.best_record.aggregate_speedup * 1.005

        if not should_promote:
            record.decision = "kept_existing_best"
            return False

        atomic_copy(source, self.final_path)
        atomic_copy(source, self.backup_path)
        atomic_copy(source, self.last_good_path)
        record.decision = "promoted_to_best"
        self.best_record = record
        return True

    def ensure_final_exists(self) -> None:
        if self.final_path.exists() and self.final_path.stat().st_size > 0:
            return
        if self.backup_path.exists() and self.backup_path.stat().st_size > 0:
            atomic_copy(self.backup_path, self.final_path)


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = ROOT
        self.spec = DEFAULT_SPEC
        self.args = args
        self.run_id = args.run_id or make_run_id()
        self.deadline = time.monotonic() + args.max_time
        self.logger = setup_runtime_logger(self.root, self.run_id)
        self.memory = ExperimentMemory(self.root, self.run_id)
        self.constraints = ConstraintSpecManager(self.spec)
        self.llm = LLMClient()
        self.llm_calls = LLMCallLogger(self.root, self.run_id, self.logger)
        self.prompts = PromptLibrary()
        self.planner = SearchSpacePlannerAgent(self.spec, self.llm, self.prompts, self.llm_calls)
        self.generator = CUDACandidateGenerator()
        self.candidate_generator_agent = LLMCUDACandidateGeneratorAgent(
            self.spec,
            self.llm,
            self.prompts,
            self.generator,
            self.llm_calls,
        )
        self.static_review = StaticCodeReviewAgent()
        self.llm_static_review = LLMStaticCodeReviewAgent(self.spec, self.llm, self.prompts, self.llm_calls)
        self.env_agent = EnvironmentInspectorAgent()
        self.compile_agent = BuildCompileAgent()
        self.correctness_agent = CorrectnessTestAgent(sizes=args.correctness_sizes)
        self.benchmark_agent = BenchmarkAgent(
            sizes=args.benchmark_sizes,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
        )
        self.profile_agent = ProfilingAgent(
            enable_torch_profile=args.enable_torch_profile,
            sample_d=args.profile_size,
        )
        self.diagnosis_agent = PerformanceDiagnosisAgent(self.spec, self.llm, self.prompts, self.llm_calls)
        self.mutation_agent = OptimizationMutationAgent(
            self.spec,
            self.llm,
            self.prompts,
            self.generator,
            self.llm_calls,
        )
        self.best_manager = BestCandidateManager(self.root, self.spec)
        self.history: list[CandidateRecord] = []

    def run(self) -> int:
        self.logger.info(
            "supervisor start: run_id=%s max_time=%ss max_iters=%s",
            self.run_id,
            self.args.max_time,
            self.args.max_iters,
        )
        self.logger.info("llm status: %s", self.llm.status)
        self.memory.append_trace(
            {
                "event": "supervisor_start",
                "run_id": self.run_id,
                "constraints": self.constraints.checklist(),
                "llm": self.llm.status,
                "max_time_sec": self.args.max_time,
            }
        )

        baseline_record = self._generate_baseline()
        if self.args.bootstrap_only:
            self.best_manager.ensure_final_exists()
            self._flush_leaderboard()
            self.logger.info("bootstrap-only run complete; final file is %s", self.best_manager.final_path)
            return 0

        env = self.env_agent.inspect()
        self.memory.write_json(self.memory.experiments_dir / "environment.json", env)
        can_test, reason = self.env_agent.can_run_cuda_tests(env)
        self.memory.append_trace({"event": "environment_checked", "can_test": can_test, "reason": reason})
        self.logger.info("environment checked: can_test=%s reason=%s", can_test, reason)
        if not can_test:
            baseline_record.decision = "kept_without_cuda_validation"
            self.best_manager.ensure_final_exists()
            self._flush_leaderboard()
            self.logger.warning("CUDA validation skipped; keeping baseline optimized_lora.cu")
            return 0

        self._evaluate_candidate(baseline_record)
        if baseline_record.compiled and baseline_record.correct:
            promoted = self.best_manager.consider(baseline_record.source_path, baseline_record)
            self.logger.info("baseline validation complete; promoted=%s", promoted)

        if self.args.max_iters == 0:
            self.best_manager.ensure_final_exists()
            self._flush_leaderboard()
            self.logger.info("max_iters=0; stopping after baseline validation")
            return 0

        plan = self.planner.plan(self.history)
        self.memory.write_json(self.memory.experiments_dir / "search_plan.json", plan)
        self.logger.info("search plan selected: %s", plan.get("strategy"))

        latest_diagnosis: dict[str, Any] | None = baseline_record.diagnosis_result or None
        next_experiment_id = 1
        while len([r for r in self.history if r.spec.experiment_id > 0]) < self.args.max_iters:
            if len([r for r in self.history if r.spec.experiment_id > 0]) >= self.args.max_iters:
                break
            if self._time_remaining() < self.args.stop_margin:
                self.memory.append_trace({"event": "stop_before_timeout", "remaining_sec": self._time_remaining()})
                self.logger.warning("stopping before timeout; remaining_sec=%.2f", self._time_remaining())
                break
            spec = self.mutation_agent.propose(
                experiment_id=next_experiment_id,
                best_record=self.best_manager.best_record.to_dict() if self.best_manager.best_record else None,
                history=[item.to_dict() for item in self.history],
                diagnosis=latest_diagnosis,
                time_remaining=self._time_remaining(),
            )
            if spec is None:
                self.logger.info("mutation agent found no untried candidates; stopping")
                break
            record = self._generate_candidate(spec)
            self._evaluate_candidate(record)
            promoted = self.best_manager.consider(record.source_path, record)
            latest_diagnosis = record.diagnosis_result or latest_diagnosis
            self.logger.info(
                "candidate %s decision=%s promoted=%s aggregate_speedup=%s",
                record.spec.name,
                record.decision,
                promoted,
                record.aggregate_speedup,
            )
            self._flush_leaderboard()
            next_experiment_id += 1

        self.best_manager.ensure_final_exists()
        self._flush_leaderboard()
        self.memory.append_trace(
            {
                "event": "supervisor_finish",
                "best": self.best_manager.best_record.to_dict() if self.best_manager.best_record else None,
            }
        )
        self.logger.info(
            "supervisor finish; best=%s final=%s",
            self.best_manager.best_record.to_dict() if self.best_manager.best_record else None,
            self.best_manager.final_path,
        )
        return 0

    def _generate_baseline(self) -> CandidateRecord:
        spec = self.generator.baseline(experiment_id=0)
        exp_dir = self.memory.experiment_dir(spec.experiment_id)
        source = exp_dir / "candidate.cu"
        self.generator.write_candidate(spec, source)
        record = CandidateRecord(spec=spec, source_path=source)
        record.generation_result = {"origin": "deterministic_baseline", "candidate": spec.metadata()}
        self.best_manager.bootstrap(source, record)
        self.history.append(record)
        self.memory.write_json(exp_dir / "candidate_metadata.json", spec.metadata())
        self.memory.append_trace({"event": "baseline_bootstrapped", "source": str(source)})
        self.logger.info("baseline bootstrapped from %s", source)
        return record

    def _generate_candidate(self, spec: CandidateSpec) -> CandidateRecord:
        exp_dir = self.memory.experiment_dir(spec.experiment_id)
        source = exp_dir / "candidate.cu"
        generation = self.candidate_generator_agent.write_candidate(
            spec,
            source,
            history=[item.to_dict() for item in self.history],
            best_code_summary=summarize_code(self.best_manager.final_path),
        )
        record = CandidateRecord(spec=spec, source_path=source)
        record.generation_result = generation
        self.history.append(record)
        self.memory.write_json(exp_dir / "candidate_metadata.json", spec.metadata())
        self.memory.write_json(exp_dir / "generation.json", generation)
        self.memory.append_trace({"event": "candidate_generated", "candidate": spec.metadata()})
        self.logger.info(
            "candidate generated: id=%s name=%s family=%s origin=%s",
            spec.experiment_id,
            spec.name,
            spec.family,
            generation.get("origin"),
        )
        return record

    def _evaluate_candidate(self, record: CandidateRecord) -> None:
        exp_dir = self.memory.experiment_dir(record.spec.experiment_id)
        self.logger.info("evaluating candidate id=%s name=%s", record.spec.experiment_id, record.spec.name)

        review = self.static_review.review(record.source_path)
        record.static_review_result = review.to_dict()
        self.memory.write_json(exp_dir / "static_review.json", record.static_review_result)
        llm_review = self.llm_static_review.review(record.spec, record.source_path)
        record.llm_static_review_result = llm_review
        if llm_review is not None:
            self.memory.write_json(exp_dir / "llm_static_review.json", llm_review)
        if self.llm_static_review.should_reject(llm_review):
            record.decision = "rejected_llm_static_review"
            record.diagnosis_result = self._diagnose(record)
            self.memory.write_json(exp_dir / "diagnosis.json", record.diagnosis_result)
            self.logger.warning("candidate %s rejected by LLM static review", record.spec.name)
            return
        if not review.passed:
            record.decision = "rejected_static_review"
            record.diagnosis_result = self._diagnose(record)
            self.memory.write_json(exp_dir / "diagnosis.json", record.diagnosis_result)
            self.memory.append_trace(
                {"event": "candidate_rejected_static", "candidate": record.spec.metadata(), "review": review.to_dict()}
            )
            self.logger.warning("candidate %s rejected by static review: %s", record.spec.name, review.errors)
            return
        self.logger.info("candidate %s passed static review with risk=%s", record.spec.name, review.risk_level)

        self.logger.info(
            "candidate %s compile started; PyTorch extension builds can be quiet for 1-2 minutes",
            record.spec.name,
        )
        compile_result, module = self.compile_agent.compile(
            record.source_path,
            build_dir=exp_dir / "torch_build",
            experiment_id=record.spec.experiment_id,
            use_fast_math=record.spec.use_fast_math,
        )
        record.compiled = compile_result.compiled
        record.compile_result = compile_result.to_dict()
        self.memory.write_json(exp_dir / "compile.json", record.compile_result)
        if not compile_result.compiled or module is None:
            record.decision = "rejected_compile"
            record.diagnosis_result = self._diagnose(record)
            self.memory.write_json(exp_dir / "diagnosis.json", record.diagnosis_result)
            self.memory.append_trace(
                {
                    "event": "candidate_rejected_compile",
                    "candidate": record.spec.metadata(),
                    "compile": compile_result.to_dict(),
                }
            )
            self.logger.warning(
                "candidate %s compile failed: type=%s summary=%s",
                record.spec.name,
                compile_result.error_type,
                compile_result.error_summary,
            )
            return
        self.logger.info("candidate %s compiled in %.2fs", record.spec.name, compile_result.compile_time_sec)

        correctness = self.correctness_agent.run(module)
        record.correct = correctness.correct
        record.correctness_result = correctness.to_dict()
        self.memory.write_json(exp_dir / "correctness.json", record.correctness_result)
        if not correctness.correct:
            record.decision = "rejected_correctness"
            profile = self.profile_agent.profile(record.spec.family, None)
            record.profile_result = profile
            record.diagnosis_result = self._diagnose(record)
            self.memory.write_json(exp_dir / "profile.json", record.profile_result)
            self.memory.write_json(exp_dir / "diagnosis.json", record.diagnosis_result)
            self.memory.append_trace(
                {
                    "event": "candidate_rejected_correctness",
                    "candidate": record.spec.metadata(),
                    "correctness": correctness.to_dict(),
                }
            )
            self.logger.warning(
                "candidate %s correctness failed: failed_shapes=%s max_abs=%s rel_l2=%s",
                record.spec.name,
                correctness.failed_shapes,
                correctness.max_abs_err,
                correctness.rel_l2_err,
            )
            return
        self.logger.info(
            "candidate %s correctness passed: max_abs=%s rel_l2=%s",
            record.spec.name,
            correctness.max_abs_err,
            correctness.rel_l2_err,
        )

        benchmark = self.benchmark_agent.run(module)
        record.aggregate_speedup = benchmark.aggregate_speedup
        record.latency_by_d = benchmark.latency_ms
        record.benchmark_result = benchmark.to_dict()
        self.memory.write_json(exp_dir / "benchmark.json", record.benchmark_result)
        profile = self.profile_agent.profile(record.spec.family, benchmark, module=module)
        record.profile_result = profile
        self.memory.write_json(exp_dir / "profile.json", record.profile_result)
        record.diagnosis_result = self._diagnose(record)
        self.memory.write_json(exp_dir / "diagnosis.json", record.diagnosis_result)
        record.decision = "evaluated_valid"
        self.memory.append_trace(
            {
                "event": "candidate_evaluated",
                "candidate": record.spec.metadata(),
                "correctness": correctness.to_dict(),
                "benchmark": benchmark.to_dict(),
            }
        )
        self.logger.info(
            "candidate %s benchmark complete: aggregate_speedup=%s latency_ms=%s",
            record.spec.name,
            benchmark.aggregate_speedup,
            benchmark.latency_ms,
        )

    def _diagnose(self, record: CandidateRecord) -> dict[str, Any]:
        return self.diagnosis_agent.diagnose(
            candidate=record.spec,
            static_review=record.static_review_result,
            compile_result=record.compile_result,
            correctness_result=record.correctness_result,
            benchmark_result=record.benchmark_result,
            profile_result=record.profile_result,
        )

    def _flush_leaderboard(self) -> None:
        self.memory.update_leaderboard([record.to_dict() for record in self.history])

    def _time_remaining(self) -> float:
        return self.deadline - time.monotonic()


def _parse_sizes(text: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not text:
        return default
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return values or default


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic LoRA CUDA optimizer")
    parser.add_argument("--max-time", type=float, default=float(os.getenv("MAX_OPT_TIME", "1700")))
    parser.add_argument("--stop-margin", type=float, default=float(os.getenv("STOP_MARGIN_SEC", "60")))
    parser.add_argument("--max-iters", type=int, default=int(os.getenv("MAX_CANDIDATES", "6")))
    parser.add_argument("--bench-warmup", type=int, default=int(os.getenv("BENCH_WARMUP", "3")))
    parser.add_argument("--bench-iters", type=int, default=int(os.getenv("BENCH_ITERS", "10")))
    parser.add_argument("--run-id", default=os.getenv("RUN_ID", ""))
    parser.add_argument(
        "--enable-torch-profile",
        action="store_true",
        default=os.getenv("ENABLE_TORCH_PROFILE", "0") == "1",
    )
    parser.add_argument("--profile-size", type=int, default=int(os.getenv("PROFILE_SIZE", "3584")))
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument(
        "--correctness-sizes",
        type=lambda value: _parse_sizes(value, DEFAULT_SPEC.correctness_sizes),
        default=_parse_sizes(os.getenv("CORRECTNESS_SIZES"), DEFAULT_SPEC.correctness_sizes),
    )
    parser.add_argument(
        "--benchmark-sizes",
        type=lambda value: _parse_sizes(value, DEFAULT_SPEC.benchmark_sizes),
        default=_parse_sizes(os.getenv("BENCHMARK_SIZES"), DEFAULT_SPEC.benchmark_sizes),
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return Supervisor(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
