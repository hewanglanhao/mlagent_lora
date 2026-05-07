from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .background_tasks import BackgroundTask, start_background_task
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
from .memory import ExperimentMemory, atomic_copy, atomic_write_text
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

    def bootstrap(self, source: Path, record: CandidateRecord, overwrite_existing: bool = True) -> bool:
        if not overwrite_existing and self._has_existing_artifact():
            self.ensure_final_exists()
            record.decision = "bootstrap_preserved_existing_best"
            return False
        atomic_copy(source, self.final_path)
        atomic_copy(source, self.backup_path)
        atomic_copy(source, self.last_good_path)
        record.decision = "bootstrapped_as_initial_best"
        self.best_record = record
        return True

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

    def _has_existing_artifact(self) -> bool:
        return any(
            path.exists() and path.stat().st_size > 0
            for path in (self.final_path, self.backup_path, self.last_good_path)
        )


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
        self.async_llm = bool(args.async_llm and self.llm.enabled)
        self.async_llm_advisory = bool(self.async_llm and args.async_llm_advisory)
        self.pending_plan: BackgroundTask[dict[str, Any]] | None = None
        self.pending_mutation: BackgroundTask[dict[str, Any] | None] | None = None
        self.pending_codegen: BackgroundTask[dict[str, Any] | None] | None = None
        self.ready_codegen: list[dict[str, Any]] = []
        self.pending_static_reviews: list[tuple[CandidateRecord, BackgroundTask[dict[str, Any] | None]]] = []
        self.pending_diagnoses: list[tuple[CandidateRecord, BackgroundTask[dict[str, Any]]]] = []
        self._last_async_mutation_status: str | None = None

    def run(self) -> int:
        self.logger.info(
            "supervisor start: run_id=%s max_time=%ss max_iters=%s",
            self.run_id,
            self.args.max_time,
            self.args.max_iters,
        )
        self.logger.info("llm status: %s", self.llm.status)
        self.logger.info(
            "async llm pipeline: enabled=%s advisory=%s codegen=%s repair_attempts=%s idle_wait_sec=%s stale_retry_wait_sec=%s",
            self.async_llm,
            self.async_llm_advisory,
            self.args.async_llm_codegen,
            self.args.llm_codegen_repair_attempts,
            self.args.async_llm_idle_wait,
            self.args.async_llm_stale_retry_wait,
        )
        self.memory.append_trace(
            {
                "event": "supervisor_start",
                "run_id": self.run_id,
                "constraints": self.constraints.checklist(),
                "llm": self.llm.status,
                "async_llm": self.async_llm,
                "async_llm_advisory": self.async_llm_advisory,
                "max_time_sec": self.args.max_time,
            }
        )

        baseline_record = self._generate_baseline(install_initial_best=not self.args.bootstrap_only)
        if self.args.bootstrap_only:
            self.best_manager.ensure_final_exists()
            self._flush_leaderboard()
            self.logger.info(
                "bootstrap-only run complete; baseline_decision=%s final file is %s",
                baseline_record.decision,
                self.best_manager.final_path,
            )
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
            if promoted:
                self._warn_if_promoted_with_failed_llm_review(baseline_record)
            self.logger.info("baseline validation complete; promoted=%s", promoted)

        if self.args.max_iters == 0:
            self.best_manager.ensure_final_exists()
            self._flush_leaderboard()
            self.logger.info("max_iters=0; stopping after baseline validation")
            return 0

        if self.async_llm:
            plan = self._base_search_plan()
            self._start_async_plan()
        else:
            plan = self.planner.plan(self.history)
        self.memory.write_json(self.memory.experiments_dir / "search_plan.json", plan)
        self.logger.info("search plan selected: %s", plan.get("strategy"))

        latest_diagnosis: dict[str, Any] | None = baseline_record.diagnosis_result or None
        next_experiment_id = 1
        if self.async_llm:
            self._start_async_mutation(latest_diagnosis)
        while len([r for r in self.history if r.spec.experiment_id > 0]) < self.args.max_iters:
            self._poll_async_llm_results()
            latest_diagnosis = self._latest_diagnosis(latest_diagnosis)
            if len([r for r in self.history if r.spec.experiment_id > 0]) >= self.args.max_iters:
                break
            if self._time_remaining() < self.args.stop_margin:
                self.memory.append_trace({"event": "stop_before_timeout", "remaining_sec": self._time_remaining()})
                self.logger.warning("stopping before timeout; remaining_sec=%.2f", self._time_remaining())
                break
            if self.async_llm:
                record = self._consume_ready_codegen_candidate(next_experiment_id)
                spec = None if record is not None else self._next_async_pipeline_spec(next_experiment_id, latest_diagnosis)
            else:
                record = None
                spec = self.mutation_agent.propose(
                    experiment_id=next_experiment_id,
                    best_record=self.best_manager.best_record.to_dict() if self.best_manager.best_record else None,
                    history=[item.to_dict() for item in self.history],
                    diagnosis=latest_diagnosis,
                    time_remaining=self._time_remaining(),
                )
            if record is None and spec is None:
                self.logger.info("mutation agent found no untried candidates; stopping")
                break
            if record is None:
                assert spec is not None
                record = self._generate_candidate(spec)
                self._start_async_codegen(spec)
            self._evaluate_candidate(record)
            promoted = self.best_manager.consider(record.source_path, record)
            if promoted:
                self._warn_if_promoted_with_failed_llm_review(record)
            latest_diagnosis = record.diagnosis_result or latest_diagnosis
            self.logger.info(
                "candidate %s decision=%s promoted=%s aggregate_speedup=%s",
                record.spec.name,
                record.decision,
                promoted,
                record.aggregate_speedup,
            )
            self._flush_leaderboard()
            next_experiment_id = self._next_available_experiment_id(next_experiment_id + 1)
            if self.async_llm:
                self._poll_async_llm_results()
                latest_diagnosis = self._latest_diagnosis(latest_diagnosis)
                self._start_async_mutation(latest_diagnosis)

        self._drain_async_llm_results(self.args.async_llm_final_wait)
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

    def _base_search_plan(self) -> dict[str, Any]:
        return {
            "strategy": "deterministic_rank16_search",
            "notes": [
                "Use ATen/cuBLAS for the d x d GEMM.",
                "Explore kernels that fuse Y += A @ T for rank 16.",
                "Validate every candidate before promotion.",
            ],
            "llm_advice_status": "pending_async" if self.async_llm else "disabled",
        }

    def _start_async_plan(self) -> None:
        if not self.async_llm or self.pending_plan is not None:
            return
        history_snapshot = list(self.history)
        self.pending_plan = start_background_task(
            "llm_search_plan",
            lambda: self.planner.plan(history_snapshot),
            {"history_count": len(history_snapshot)},
        )
        self.logger.info("async_llm_start task=search_space_planner history_count=%s", len(history_snapshot))

    def _start_async_mutation(self, diagnosis: dict[str, Any] | None) -> None:
        if not self.async_llm or self.pending_mutation is not None:
            return
        if self._time_remaining() < self.args.stop_margin:
            return
        best_snapshot = self._best_dict()
        history_snapshot = self._history_dicts()
        diagnosis_snapshot = dict(diagnosis) if isinstance(diagnosis, dict) else diagnosis
        time_remaining = self._time_remaining()
        self.pending_mutation = start_background_task(
            "llm_mutation",
            lambda: self.mutation_agent.ask_llm_mutation(
                best_record=best_snapshot,
                history=history_snapshot,
                diagnosis=diagnosis_snapshot,
                time_remaining=time_remaining,
            ),
            {"history_count": len(history_snapshot), "time_remaining": time_remaining},
        )
        self.logger.info(
            "async_llm_start task=optimization_mutation history_count=%s time_remaining=%.2f",
            len(history_snapshot),
            time_remaining,
        )

    def _start_async_codegen(self, spec: CandidateSpec) -> None:
        if not self.async_llm or not self.args.async_llm_codegen or self.pending_codegen is not None:
            return
        if spec.llm_generated:
            return
        history_snapshot = self._history_dicts()
        best_code_summary = summarize_code(self.best_manager.final_path)
        base_spec = spec

        def generate() -> dict[str, Any] | None:
            code = self.candidate_generator_agent.generate_code(
                base_spec,
                history=history_snapshot,
                best_code_summary=best_code_summary,
            )
            if not code:
                return None
            return {
                "base_spec": base_spec,
                "base_metadata": base_spec.metadata(),
                "code": code,
                "code_bytes": len(code.encode("utf-8")),
            }

        self.pending_codegen = start_background_task(
            "llm_cuda_codegen",
            generate,
            {"base_candidate": base_spec.metadata(), "history_count": len(history_snapshot)},
        )
        self.logger.info(
            "async_llm_start task=cuda_candidate_generator base_candidate=%s",
            base_spec.name,
        )

    def _consume_ready_codegen_candidate(self, experiment_id: int) -> CandidateRecord | None:
        if not self.ready_codegen:
            return None
        payload = self.ready_codegen.pop(0)
        base_spec = payload.get("base_spec")
        if not isinstance(base_spec, CandidateSpec):
            self.logger.warning("async_llm_codegen_drop reason=missing_base_spec")
            return None
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            self.logger.warning("async_llm_codegen_drop base_candidate=%s reason=empty_code", base_spec.name)
            return None
        spec = replace(
            base_spec,
            experiment_id=experiment_id,
            name=f"{base_spec.name}_llm_codegen",
            llm_generated=True,
            strategy=f"llm_cuda_codegen_from:{base_spec.name}",
            parent=base_spec.experiment_id,
        )
        tried = {self.generator.candidate_key(item.spec) for item in self.history if item.spec.llm_generated}
        if self.generator.candidate_key(spec) in tried:
            self.logger.info(
                "async_llm_codegen_drop base_candidate=%s reason=duplicate_llm_codegen_key",
                base_spec.name,
            )
            return None
        return self._generate_candidate_from_code(
            spec,
            code,
            generation={
                "origin": "llm_async_codegen",
                "candidate": spec.metadata(),
                "base_candidate": payload.get("base_metadata") or base_spec.metadata(),
                "bytes": payload.get("code_bytes"),
            },
        )

    def _next_async_pipeline_spec(
        self,
        experiment_id: int,
        latest_diagnosis: dict[str, Any] | None,
    ) -> CandidateSpec | None:
        spec = self._consume_async_mutation(experiment_id)
        if spec is not None:
            return spec
        if self._last_async_mutation_status in {"stale", "empty", "error"}:
            status = self._last_async_mutation_status
            self._start_async_mutation(latest_diagnosis)
            if self.pending_mutation is not None:
                self.logger.info(
                    "async_llm_retry task=optimization_mutation reason=%s wait_sec=%.2f",
                    status,
                    self.args.async_llm_stale_retry_wait,
                )
                spec = self._wait_for_async_mutation(
                    experiment_id,
                    wait_override=self.args.async_llm_stale_retry_wait,
                )
                if spec is not None:
                    return spec
        if self.pending_mutation is None:
            self._start_async_mutation(latest_diagnosis)

        parent = self.best_manager.best_record.spec.experiment_id if self.best_manager.best_record else 0
        spec = self.generator.next_untried(experiment_id, self._history_dicts(), parent=parent)
        if spec is not None:
            if self.pending_mutation is not None and not self.pending_mutation.done():
                self.logger.info(
                    "async_llm_pending task=optimization_mutation; evaluating local candidate id=%s name=%s",
                    spec.experiment_id,
                    spec.name,
                )
            return spec

        self._start_async_mutation(latest_diagnosis)
        return self._wait_for_async_mutation(experiment_id)

    def _consume_async_mutation(self, experiment_id: int) -> CandidateSpec | None:
        self._last_async_mutation_status = None
        task = self.pending_mutation
        if task is None or not task.done():
            return None
        self.pending_mutation = None
        if task.error:
            self._last_async_mutation_status = "error"
            self.logger.warning(
                "async_llm_error task=optimization_mutation duration=%.2fs error=%s",
                task.duration_sec(),
                task.error,
            )
            return None
        mutation = task.result
        if not mutation:
            self._last_async_mutation_status = "empty"
            self.logger.info(
                "async_llm_empty task=optimization_mutation duration=%.2fs",
                task.duration_sec(),
            )
            return None
        candidate = self.mutation_agent.candidate_from_mutation(
            experiment_id=experiment_id,
            mutation=mutation,
            best_record=self._best_dict(),
            history=self._history_dicts(),
        )
        if candidate is None:
            self._last_async_mutation_status = "stale"
            self.logger.info(
                "async_llm_stale task=optimization_mutation duration=%.2fs reason=duplicate_or_invalid",
                task.duration_sec(),
            )
            return None
        self.memory.append_trace(
            {
                "event": "async_llm_mutation_selected",
                "candidate": candidate.metadata(),
                "duration_sec": task.duration_sec(),
            }
        )
        self.logger.info(
            "async_llm_ready task=optimization_mutation duration=%.2fs candidate=%s",
            task.duration_sec(),
            candidate.name,
        )
        self._last_async_mutation_status = "ready"
        return candidate

    def _wait_for_async_mutation(self, experiment_id: int, wait_override: float | None = None) -> CandidateSpec | None:
        if self.pending_mutation is None:
            return None
        wait_sec = min(
            self.args.async_llm_idle_wait if wait_override is None else wait_override,
            max(0.0, self._time_remaining() - self.args.stop_margin),
        )
        if wait_sec <= 0:
            return None
        self.logger.info(
            "no local candidate ready; waiting up to %.2fs for pending LLM mutation advice",
            wait_sec,
        )
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            spec = self._consume_async_mutation(experiment_id)
            if spec is not None:
                return spec
            self._poll_async_llm_results()
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return self._consume_async_mutation(experiment_id)

    def _start_async_static_review(self, record: CandidateRecord) -> None:
        if not self.async_llm_advisory or not self.llm_static_review.enabled:
            return
        task = start_background_task(
            "llm_static_review",
            lambda: self.llm_static_review.review(record.spec, record.source_path),
            {"candidate": record.spec.metadata(), "source_path": str(record.source_path)},
        )
        self.pending_static_reviews.append((record, task))
        self.logger.info("async_llm_start task=static_code_review candidate=%s", record.spec.name)

    def _start_async_diagnosis(self, record: CandidateRecord) -> None:
        if not self.async_llm_advisory or not self.diagnosis_agent.enabled:
            return
        static_review = dict(record.static_review_result)
        compile_result = dict(record.compile_result)
        correctness_result = dict(record.correctness_result)
        benchmark_result = dict(record.benchmark_result)
        profile_result = dict(record.profile_result)
        task = start_background_task(
            "llm_diagnosis",
            lambda: self.diagnosis_agent.diagnose(
                candidate=record.spec,
                static_review=static_review,
                compile_result=compile_result,
                correctness_result=correctness_result,
                benchmark_result=benchmark_result,
                profile_result=profile_result,
            ),
            {"candidate": record.spec.metadata()},
        )
        self.pending_diagnoses.append((record, task))
        self.logger.info("async_llm_start task=performance_diagnosis candidate=%s", record.spec.name)

    def _poll_async_llm_results(self) -> None:
        if self.pending_plan is not None and self.pending_plan.done():
            task = self.pending_plan
            self.pending_plan = None
            if task.error:
                self.logger.warning(
                    "async_llm_error task=search_space_planner duration=%.2fs error=%s",
                    task.duration_sec(),
                    task.error,
                )
            elif task.result:
                self.memory.write_json(self.memory.experiments_dir / "search_plan.json", task.result)
                self.memory.append_trace(
                    {"event": "async_search_plan_ready", "duration_sec": task.duration_sec()}
                )
                self.logger.info(
                    "async_llm_ready task=search_space_planner duration=%.2fs strategy=%s",
                    task.duration_sec(),
                    task.result.get("strategy"),
                )

        if self.pending_codegen is not None and self.pending_codegen.done():
            task = self.pending_codegen
            self.pending_codegen = None
            if task.error:
                self.logger.warning(
                    "async_llm_error task=cuda_candidate_generator duration=%.2fs error=%s",
                    task.duration_sec(),
                    task.error,
                )
            elif task.result:
                self.ready_codegen.append(task.result)
                base = task.result.get("base_metadata", {})
                self.memory.append_trace(
                    {
                        "event": "async_cuda_codegen_ready",
                        "base_candidate": base,
                        "duration_sec": task.duration_sec(),
                        "code_bytes": task.result.get("code_bytes"),
                    }
                )
                self.logger.info(
                    "async_llm_ready task=cuda_candidate_generator duration=%.2fs base_candidate=%s code_bytes=%s",
                    task.duration_sec(),
                    base.get("name"),
                    task.result.get("code_bytes"),
                )
            else:
                self.logger.info(
                    "async_llm_empty task=cuda_candidate_generator duration=%.2fs",
                    task.duration_sec(),
                )

        remaining_reviews: list[tuple[CandidateRecord, BackgroundTask[dict[str, Any] | None]]] = []
        for record, task in self.pending_static_reviews:
            if not task.done():
                remaining_reviews.append((record, task))
                continue
            if task.error:
                self.logger.warning(
                    "async_llm_error task=static_code_review candidate=%s duration=%.2fs error=%s",
                    record.spec.name,
                    task.duration_sec(),
                    task.error,
                )
                continue
            if task.result is not None:
                record.llm_static_review_result = task.result
                exp_dir = self.memory.experiment_dir(record.spec.experiment_id)
                self.memory.write_json(exp_dir / "llm_static_review.json", task.result)
                self.logger.info(
                    "async_llm_ready task=static_code_review candidate=%s duration=%.2fs",
                    record.spec.name,
                    task.duration_sec(),
                )
                self._warn_if_promoted_with_failed_llm_review(record)
        self.pending_static_reviews = remaining_reviews

        remaining_diagnoses: list[tuple[CandidateRecord, BackgroundTask[dict[str, Any]]]] = []
        for record, task in self.pending_diagnoses:
            if not task.done():
                remaining_diagnoses.append((record, task))
                continue
            if task.error:
                self.logger.warning(
                    "async_llm_error task=performance_diagnosis candidate=%s duration=%.2fs error=%s",
                    record.spec.name,
                    task.duration_sec(),
                    task.error,
                )
                continue
            if task.result:
                record.diagnosis_result = task.result
                exp_dir = self.memory.experiment_dir(record.spec.experiment_id)
                self.memory.write_json(exp_dir / "diagnosis.json", task.result)
                self.memory.write_json(exp_dir / "diagnosis_llm.json", task.result)
                self.logger.info(
                    "async_llm_ready task=performance_diagnosis candidate=%s duration=%.2fs",
                    record.spec.name,
                    task.duration_sec(),
                )
        self.pending_diagnoses = remaining_diagnoses

    def _drain_async_llm_results(self, wait_sec: float) -> None:
        if not self.async_llm or wait_sec <= 0:
            return
        deadline = time.monotonic() + min(wait_sec, max(0.0, self._time_remaining()))
        while time.monotonic() < deadline:
            self._poll_async_llm_results()
            if self.pending_mutation is not None and self.pending_mutation.done():
                self.logger.info(
                    "async_llm_ready task=optimization_mutation duration=%.2fs ignored=no_remaining_slot",
                    self.pending_mutation.duration_sec(),
                )
                self.pending_mutation = None
            if (
                not self.pending_plan
                and not self.pending_codegen
                and not self.pending_static_reviews
                and not self.pending_diagnoses
            ):
                return
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        self._poll_async_llm_results()

    def _latest_diagnosis(self, fallback: dict[str, Any] | None) -> dict[str, Any] | None:
        for record in reversed(self.history):
            if record.diagnosis_result:
                return record.diagnosis_result
        return fallback

    def _history_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.history]

    def _best_dict(self) -> dict[str, Any] | None:
        return self.best_manager.best_record.to_dict() if self.best_manager.best_record else None

    def _next_available_experiment_id(self, start: int | None = None) -> int:
        used = {record.spec.experiment_id for record in self.history}
        next_id = max(start or 1, (max(used) + 1) if used else 1)
        while next_id in used:
            next_id += 1
        return next_id

    def _warn_if_promoted_with_failed_llm_review(self, record: CandidateRecord) -> None:
        review = record.llm_static_review_result
        if record.decision != "promoted_to_best" or not isinstance(review, dict):
            return
        if review.get("pass") is not False:
            return
        event = {
            "event": "promoted_with_failed_llm_static_review",
            "candidate": record.spec.metadata(),
            "risk_level": review.get("risk_level"),
            "errors": review.get("errors") or [],
            "warnings": review.get("warnings") or [],
            "suggested_fixes": review.get("suggested_fixes") or [],
        }
        self.memory.append_trace(event)
        self.logger.warning(
            "PROMOTED_WITH_FAILED_LLM_STATIC_REVIEW candidate=%s risk=%s errors=%s warnings=%s suggested_fixes=%s",
            record.spec.name,
            review.get("risk_level"),
            review.get("errors") or [],
            review.get("warnings") or [],
            review.get("suggested_fixes") or [],
        )

    def _generate_baseline(self, install_initial_best: bool = True) -> CandidateRecord:
        spec = self.generator.baseline(experiment_id=0)
        exp_dir = self.memory.experiment_dir(spec.experiment_id)
        source = exp_dir / "candidate.cu"
        self.generator.write_candidate(spec, source)
        record = CandidateRecord(spec=spec, source_path=source)
        record.generation_result = {"origin": "deterministic_baseline", "candidate": spec.metadata()}
        installed = self.best_manager.bootstrap(source, record, overwrite_existing=install_initial_best)
        self.history.append(record)
        self.memory.write_json(exp_dir / "candidate_metadata.json", spec.metadata())
        self.memory.append_trace(
            {
                "event": "baseline_bootstrapped",
                "source": str(source),
                "installed_as_initial_best": installed,
                "decision": record.decision,
            }
        )
        self.logger.info(
            "baseline bootstrapped from %s installed_as_initial_best=%s decision=%s",
            source,
            installed,
            record.decision,
        )
        return record

    def _generate_candidate(self, spec: CandidateSpec) -> CandidateRecord:
        exp_dir = self.memory.experiment_dir(spec.experiment_id)
        source = exp_dir / "candidate.cu"
        if self.async_llm:
            self.generator.write_candidate(spec, source)
            generation = {
                "origin": "deterministic_async_pipeline",
                "candidate": spec.metadata(),
                "note": "LLM code generation skipped on the foreground path so compile/test work can continue.",
            }
        else:
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

    def _generate_candidate_from_code(
        self,
        spec: CandidateSpec,
        code: str,
        generation: dict[str, Any],
    ) -> CandidateRecord:
        exp_dir = self.memory.experiment_dir(spec.experiment_id)
        source = exp_dir / "candidate.cu"
        atomic_write_text(source, code)
        record = CandidateRecord(spec=spec, source_path=source)
        record.generation_result = generation
        self.history.append(record)
        self.memory.write_json(exp_dir / "candidate_metadata.json", spec.metadata())
        self.memory.write_json(exp_dir / "generation.json", generation)
        self.memory.append_trace({"event": "candidate_generated", "candidate": spec.metadata()})
        self.logger.info(
            "candidate generated: id=%s name=%s family=%s origin=%s base_candidate=%s",
            spec.experiment_id,
            spec.name,
            spec.family,
            generation.get("origin"),
            generation.get("base_candidate", {}).get("name"),
        )
        return record

    def _evaluate_candidate(self, record: CandidateRecord) -> None:
        exp_dir = self.memory.experiment_dir(record.spec.experiment_id)
        self.logger.info("evaluating candidate id=%s name=%s", record.spec.experiment_id, record.spec.name)

        review = self.static_review.review(record.source_path)
        record.static_review_result = review.to_dict()
        self.memory.write_json(exp_dir / "static_review.json", record.static_review_result)
        if self.async_llm and not self.llm_static_review.can_reject:
            self._start_async_static_review(record)
        else:
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
            self._maybe_repair_failed_llm_codegen(record)
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

    def _maybe_repair_failed_llm_codegen(self, record: CandidateRecord) -> None:
        if not self.async_llm or not self.args.async_llm_codegen:
            return
        if not record.spec.llm_generated:
            return
        max_attempts = self.args.llm_codegen_repair_attempts
        if max_attempts <= 0:
            return
        repair_attempt = int(record.generation_result.get("repair_attempt", 0) or 0) + 1
        if repair_attempt > max_attempts:
            self.logger.info(
                "llm_codegen_repair_skip candidate=%s reason=max_attempts_reached attempts=%s",
                record.spec.name,
                max_attempts,
            )
            return

        failed_code = record.source_path.read_text(encoding="utf-8", errors="replace")
        self.logger.info(
            "llm_codegen_repair_start candidate=%s attempt=%s error_type=%s",
            record.spec.name,
            repair_attempt,
            record.compile_result.get("error_type"),
        )
        repaired_code = self.candidate_generator_agent.repair_code(
            spec=record.spec,
            failed_code=failed_code,
            compile_result=record.compile_result,
            history=self._history_dicts(),
            best_code_summary=summarize_code(self.best_manager.final_path),
            repair_attempt=repair_attempt,
        )
        if not repaired_code:
            self.logger.warning(
                "llm_codegen_repair_empty candidate=%s attempt=%s",
                record.spec.name,
                repair_attempt,
            )
            return

        repair_spec = replace(
            record.spec,
            experiment_id=self._next_available_experiment_id(),
            name=f"{record.spec.name}_repair{repair_attempt}",
            parent=record.spec.experiment_id,
            strategy=f"llm_cuda_repair_attempt:{repair_attempt}",
        )
        repair_record = self._generate_candidate_from_code(
            repair_spec,
            repaired_code,
            generation={
                "origin": "llm_repair_codegen",
                "candidate": repair_spec.metadata(),
                "failed_candidate": record.spec.metadata(),
                "failed_compile": record.compile_result,
                "repair_attempt": repair_attempt,
                "bytes": len(repaired_code.encode("utf-8")),
            },
        )
        self._evaluate_candidate(repair_record)
        promoted = self.best_manager.consider(repair_record.source_path, repair_record)
        if promoted:
            self._warn_if_promoted_with_failed_llm_review(repair_record)
        self.logger.info(
            "candidate %s repair decision=%s promoted=%s aggregate_speedup=%s",
            repair_record.spec.name,
            repair_record.decision,
            promoted,
            repair_record.aggregate_speedup,
        )
        self._flush_leaderboard()

    def _diagnose(self, record: CandidateRecord) -> dict[str, Any]:
        if self.async_llm:
            fallback = self.diagnosis_agent.fallback_diagnosis(
                candidate=record.spec,
                static_review=record.static_review_result,
                compile_result=record.compile_result,
                correctness_result=record.correctness_result,
                benchmark_result=record.benchmark_result,
                profile_result=record.profile_result,
            )
            self._start_async_diagnosis(record)
            return fallback
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


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


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
        "--async-llm",
        action="store_true",
        default=_env_flag("ENABLE_ASYNC_LLM", True),
        help="Run slow LLM advisory calls in the background while local validation continues.",
    )
    parser.add_argument(
        "--disable-async-llm",
        action="store_true",
        default=_env_flag("DISABLE_ASYNC_LLM", False),
        help="Force the older synchronous LLM pipeline.",
    )
    parser.add_argument(
        "--async-llm-advisory",
        action="store_true",
        default=_env_flag("ASYNC_LLM_ADVISORY", False),
        help="Also run non-gating LLM review/diagnosis in background.",
    )
    parser.add_argument(
        "--async-llm-codegen",
        action="store_true",
        default=_env_flag("ASYNC_LLM_CODEGEN", True),
        help="Generate LLM CUDA code in the background and evaluate returned code locally.",
    )
    parser.add_argument(
        "--llm-codegen-repair-attempts",
        type=int,
        default=int(os.getenv("LLM_CODEGEN_REPAIR_ATTEMPTS", "3")),
    )
    parser.add_argument(
        "--async-llm-idle-wait",
        type=float,
        default=float(os.getenv("ASYNC_LLM_IDLE_WAIT_SEC", "3")),
    )
    parser.add_argument(
        "--async-llm-stale-retry-wait",
        type=float,
        default=float(os.getenv("ASYNC_LLM_STALE_RETRY_WAIT_SEC", "45")),
    )
    parser.add_argument(
        "--async-llm-final-wait",
        type=float,
        default=float(os.getenv("ASYNC_LLM_FINAL_WAIT_SEC", "2")),
    )
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
    if args.disable_async_llm:
        args.async_llm = False
    return Supervisor(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
