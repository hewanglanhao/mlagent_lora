from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .cuda_candidates import CUDACandidateGenerator, CandidateSpec
from .llm_adapter import LLMClient
from .llm_call_logger import LLMCallLogger
from .memory import atomic_write_text
from .prompt_library import PromptLibrary
from .spec import LoraSpec


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:cuda|cpp|c\+\+|cu)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def sanitize_reserved_label_text(text: str) -> str:
    reserved = "".join(("q", "y", "h"))
    return re.sub(reserved, "reference", text, flags=re.IGNORECASE)


def sanitize_json_labels(value):
    if isinstance(value, dict):
        return {key: sanitize_json_labels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_labels(item) for item in value]
    if isinstance(value, str):
        return sanitize_reserved_label_text(value)
    return value


def summarize_code(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return "No current best code is available."
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n/* ... code summary trimmed ... */"


class LLMCUDACandidateGeneratorAgent:
    def __init__(
        self,
        spec: LoraSpec,
        llm: LLMClient,
        prompts: PromptLibrary,
        fallback: CUDACandidateGenerator,
        llm_calls: LLMCallLogger | None = None,
    ) -> None:
        self.spec = spec
        self.llm = llm
        self.prompts = prompts
        self.fallback = fallback
        self.llm_calls = llm_calls
        self.enabled = os.getenv("ENABLE_LLM_CODEGEN", "1") != "0"

    def write_candidate(
        self,
        spec: CandidateSpec,
        path: Path,
        history: list[dict[str, Any]],
        best_code_summary: str,
    ) -> dict[str, Any]:
        if self.enabled and self.llm.enabled:
            generated = self._try_llm_generate(spec, history, best_code_summary)
            if generated:
                atomic_write_text(path, generated)
                return {
                    "origin": "llm",
                    "enabled": True,
                    "candidate": spec.metadata(),
                    "bytes": len(generated.encode("utf-8")),
                }

        self.fallback.write_candidate(spec, path)
        return {
            "origin": "deterministic_fallback",
            "enabled": self.enabled and self.llm.enabled,
            "candidate": spec.metadata(),
        }

    def generate_code(
        self,
        spec: CandidateSpec,
        history: list[dict[str, Any]],
        best_code_summary: str,
    ) -> str | None:
        if not self.enabled or not self.llm.enabled:
            return None
        return self._try_llm_generate(spec, history, best_code_summary)

    def repair_code(
        self,
        spec: CandidateSpec,
        failed_code: str,
        compile_result: dict[str, Any],
        history: list[dict[str, Any]],
        best_code_summary: str,
        repair_attempt: int,
    ) -> str | None:
        if not self.enabled or not self.llm.enabled:
            return None
        prompt = self.prompts.pair("cuda_candidate_generator")
        user = "\n\n".join(
            [
                "Repair the failed PyTorch CUDA extension below.",
                "Return exactly one complete corrected single-file CUDA/C++ extension. Do not return markdown.",
                "The repaired code must keep the required forward(W, X, A, B) entrypoint and PYBIND11_MODULE binding.",
                "If using the current CUDA stream, include <ATen/cuda/CUDAContext.h> and call at::cuda::getCurrentCUDAStream() without a device argument.",
                "For pure cuBLAS repairs, prefer the exact reference three-SGEMM mapping: SGEMM1 opA=N/opB=N with A=X and B=W; SGEMM2 opA=N/opB=T with A=X and B=B, U shape {d,16}, m=d, n=16, k=d, ldc=d; SGEMM3 opA=N/opB=N with A=U and B=A, m=d, n=d, k=16, beta=1.",
                "Do not repair into previously failed alternatives: opA=T/opB=T in SGEMM1 or SGEMM3, treating U as a [16,d] column-major buffer, m=16/n=d for U, or U leading dimension 16.",
                f"Repair attempt: {repair_attempt}",
                "SPEC_CONTEXT:\n" + self.spec.as_prompt_context(),
                "CANDIDATE_METADATA_JSON:\n" + json.dumps(spec.metadata(), indent=2, sort_keys=True),
                "COMPILE_RESULT_JSON:\n" + json.dumps(compile_result, indent=2, sort_keys=True),
                "RECENT_EXPERIMENT_HISTORY_JSON:\n" + json.dumps(history[-6:], indent=2, sort_keys=True),
                "CURRENT_BEST_CODE_SUMMARY:\n" + best_code_summary,
                "FAILED_CUDA_CODE:\n" + failed_code,
            ]
        )
        response = self._complete(
            prompt.system,
            user,
            "cuda_candidate_repair",
            {"candidate": spec.metadata(), "repair_attempt": repair_attempt},
        )
        if not response:
            return None
        code = sanitize_reserved_label_text(strip_code_fences(response.content))
        if self._looks_like_cuda_extension(code):
            if self.llm_calls:
                self.llm_calls.record_parse(response.call_id, "cuda_extension_validation", True)
            return code
        if self.llm_calls:
            self.llm_calls.record_parse(
                response.call_id,
                "cuda_extension_validation",
                False,
                "repair response did not look like a complete PyTorch CUDA extension",
            )
        return None

    def _try_llm_generate(
        self,
        spec: CandidateSpec,
        history: list[dict[str, Any]],
        best_code_summary: str,
    ) -> str | None:
        prompt = self.prompts.pair("cuda_candidate_generator")
        user = self.prompts.render(
            prompt.user,
            {
                "SPEC_CONTEXT": self.spec.as_prompt_context(),
                "CANDIDATE_STRATEGY_JSON": json.dumps(spec.metadata(), indent=2, sort_keys=True),
                "EXPERIMENT_HISTORY_JSON": json.dumps(history[-6:], indent=2, sort_keys=True),
                "BEST_CODE_SUMMARY": best_code_summary,
            },
        )
        response = self._complete(prompt.system, user, "cuda_candidate_generator", spec.metadata())
        if not response:
            return None
        code = sanitize_reserved_label_text(strip_code_fences(response.content))
        if self._looks_like_cuda_extension(code):
            if self.llm_calls:
                self.llm_calls.record_parse(response.call_id, "cuda_extension_validation", True)
            return code
        if self.llm_calls:
            self.llm_calls.record_parse(
                response.call_id,
                "cuda_extension_validation",
                False,
                "response did not look like a complete PyTorch CUDA extension",
            )
        return None

    def _complete(
        self,
        system: str,
        user: str,
        agent_name: str,
        metadata: dict[str, Any],
    ):
        system = sanitize_reserved_label_text(system)
        user = sanitize_reserved_label_text(user)
        metadata = sanitize_json_labels(metadata)
        if self.llm_calls:
            return self.llm_calls.complete(self.llm, agent_name, system, user, metadata)
        return self.llm.complete(system=system, user=user)

    def _looks_like_cuda_extension(self, code: str) -> bool:
        required = [
            "#include <torch/extension.h>",
            "torch::Tensor forward",
            "PYBIND11_MODULE",
            "W",
            "X",
            "A",
            "B",
        ]
        return len(code) > 500 and all(item in code for item in required)


class LLMStaticCodeReviewAgent:
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
        self.enabled = os.getenv("ENABLE_LLM_STATIC_REVIEW", "1") != "0"
        self.can_reject = os.getenv("LLM_STATIC_REVIEW_CAN_REJECT", "0") == "1"

    def review(self, candidate: CandidateSpec, source_path: Path) -> dict[str, Any] | None:
        if not self.enabled or not self.llm.enabled:
            return None
        prompt = self.prompts.pair("static_code_review")
        user = self.prompts.render(
            prompt.user,
            {
                "SPEC_CONTEXT": self.spec.as_prompt_context(),
                "CANDIDATE_METADATA_JSON": json.dumps(candidate.metadata(), indent=2, sort_keys=True),
                "CANDIDATE_CODE": source_path.read_text(encoding="utf-8", errors="replace"),
            },
        )
        response = self._complete(
            prompt.system,
            user,
            "static_code_review",
            {"candidate": candidate.metadata(), "source_path": str(source_path)},
        )
        parsed = parse_json_object(response.content if response else None)
        if parsed is None:
            if response is not None and self.llm_calls:
                self.llm_calls.record_parse(response.call_id, "json_object", False, "static review JSON parse failed")
            return {"pass": True, "risk_level": "unknown", "errors": [], "warnings": ["LLM review parse failed"]}
        if response is not None and self.llm_calls:
            self.llm_calls.record_parse(response.call_id, "json_object", True)
        parsed = sanitize_json_labels(parsed)
        return parsed

    def _complete(self, system: str, user: str, agent_name: str, metadata: dict[str, Any]):
        system = sanitize_reserved_label_text(system)
        user = sanitize_reserved_label_text(user)
        metadata = sanitize_json_labels(metadata)
        if self.llm_calls:
            return self.llm_calls.complete(self.llm, agent_name, system, user, metadata)
        return self.llm.complete(system=system, user=user)

    def should_reject(self, review: dict[str, Any] | None) -> bool:
        if not self.can_reject or not review:
            return False
        return review.get("pass") is False and bool(review.get("errors"))


class PerformanceDiagnosisAgent:
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
        self.enabled = os.getenv("ENABLE_LLM_DIAGNOSIS", "1") != "0"

    def diagnose(
        self,
        candidate: CandidateSpec,
        static_review: dict[str, Any] | None,
        compile_result: dict[str, Any] | None,
        correctness_result: dict[str, Any] | None,
        benchmark_result: dict[str, Any] | None,
        profile_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        fallback = self._fallback_diagnosis(
            candidate,
            static_review,
            compile_result,
            correctness_result,
            benchmark_result,
            profile_result,
        )
        if not self.enabled or not self.llm.enabled:
            return fallback

        prompt = self.prompts.pair("performance_diagnosis")
        user = self.prompts.render(
            prompt.user,
            {
                "SPEC_CONTEXT": self.spec.as_prompt_context(),
                "CANDIDATE_METADATA_JSON": json.dumps(candidate.metadata(), indent=2, sort_keys=True),
                "STATIC_REVIEW_JSON": json.dumps(static_review or {}, indent=2, sort_keys=True),
                "COMPILE_JSON": json.dumps(compile_result or {}, indent=2, sort_keys=True),
                "CORRECTNESS_JSON": json.dumps(correctness_result or {}, indent=2, sort_keys=True),
                "BENCHMARK_JSON": json.dumps(benchmark_result or {}, indent=2, sort_keys=True),
                "PROFILE_JSON": json.dumps(profile_result or {}, indent=2, sort_keys=True),
            },
        )
        response = self._complete(
            prompt.system,
            user,
            "performance_diagnosis",
            {"candidate": candidate.metadata()},
        )
        parsed = parse_json_object(response.content if response else None)
        if parsed is None:
            fallback["llm_parse_failed"] = True
            if response is not None and self.llm_calls:
                self.llm_calls.record_parse(response.call_id, "json_object", False, "diagnosis JSON parse failed")
            return fallback
        if response is not None and self.llm_calls:
            self.llm_calls.record_parse(response.call_id, "json_object", True)
        parsed = sanitize_json_labels(parsed)
        parsed["fallback_diagnosis"] = fallback
        return parsed

    def fallback_diagnosis(
        self,
        candidate: CandidateSpec,
        static_review: dict[str, Any] | None,
        compile_result: dict[str, Any] | None,
        correctness_result: dict[str, Any] | None,
        benchmark_result: dict[str, Any] | None,
        profile_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._fallback_diagnosis(
            candidate,
            static_review,
            compile_result,
            correctness_result,
            benchmark_result,
            profile_result,
        )

    def _complete(self, system: str, user: str, agent_name: str, metadata: dict[str, Any]):
        system = sanitize_reserved_label_text(system)
        user = sanitize_reserved_label_text(user)
        metadata = sanitize_json_labels(metadata)
        if self.llm_calls:
            return self.llm_calls.complete(self.llm, agent_name, system, user, metadata)
        return self.llm.complete(system=system, user=user)

    def _fallback_diagnosis(
        self,
        candidate: CandidateSpec,
        static_review: dict[str, Any] | None,
        compile_result: dict[str, Any] | None,
        correctness_result: dict[str, Any] | None,
        benchmark_result: dict[str, Any] | None,
        profile_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if static_review and not static_review.get("passed", True):
            return {
                "bottleneck": "static_review_failure",
                "confidence": "high",
                "evidence": [json.dumps(static_review, sort_keys=True)[:800]],
                "recommended_actions": ["fix interface, shape, dtype, or single-file compliance"],
                "avoid": ["compiling high-risk non-compliant candidates"],
                "promotion_advice": "reject",
            }
        if compile_result and not compile_result.get("compiled"):
            return {
                "bottleneck": "compile_failure",
                "confidence": "high",
                "evidence": [str(compile_result.get("error_type")), str(compile_result.get("error_summary"))[:500]],
                "recommended_actions": ["fall back to a simpler candidate", "preserve current best"],
                "avoid": ["promoting uncompiled code"],
                "promotion_advice": "reject",
            }
        if correctness_result and not correctness_result.get("correct"):
            return {
                "bottleneck": "correctness_failure",
                "confidence": "high",
                "evidence": [json.dumps(correctness_result, sort_keys=True)[:800]],
                "recommended_actions": ["use ATen for risky math", "check rank-16 indexing and vector alignment"],
                "avoid": ["benchmark-only decisions"],
                "promotion_advice": "reject",
            }
        aggregate = benchmark_result.get("aggregate_speedup") if benchmark_result else None
        if aggregate is not None and aggregate < 1.0:
            return {
                "bottleneck": profile_result.get("bottleneck", "slower_than_reference") if profile_result else "slower_than_reference",
                "confidence": "medium",
                "evidence": [json.dumps(benchmark_result, sort_keys=True)[:800]],
                "recommended_actions": ["try vector width 4", "vary block size", "keep W @ X in cuBLAS"],
                "avoid": ["custom full GEMM without evidence"],
                "promotion_advice": "reject",
            }
        return {
            "bottleneck": profile_result.get("bottleneck", "valid_candidate") if profile_result else "valid_candidate",
            "confidence": "medium",
            "evidence": [json.dumps(benchmark_result or {}, sort_keys=True)[:800]],
            "recommended_actions": profile_result.get("suggested_actions", ["continue local mutations"]) if profile_result else ["continue local mutations"],
            "avoid": ["overfitting one exact d"],
            "promotion_advice": "promote" if aggregate is not None else "needs_more_data",
            "candidate_family": candidate.family,
        }


class OptimizationMutationAgent:
    def __init__(
        self,
        spec: LoraSpec,
        llm: LLMClient,
        prompts: PromptLibrary,
        generator: CUDACandidateGenerator,
        llm_calls: LLMCallLogger | None = None,
    ) -> None:
        self.spec = spec
        self.llm = llm
        self.prompts = prompts
        self.generator = generator
        self.llm_calls = llm_calls
        self.enabled = os.getenv("ENABLE_LLM_MUTATION", "1") != "0"

    def propose(
        self,
        experiment_id: int,
        best_record: dict[str, Any] | None,
        history: list[dict[str, Any]],
        diagnosis: dict[str, Any] | None,
        time_remaining: float,
    ) -> CandidateSpec | None:
        mutation = self.ask_llm_mutation(best_record, history, diagnosis, time_remaining)
        if mutation:
            candidate = self.candidate_from_mutation(experiment_id, mutation, best_record, history)
            if candidate:
                return candidate

        parent = best_record.get("id") if best_record else 0
        return self.generator.next_untried(experiment_id, history, parent=parent)

    def ask_llm_mutation(
        self,
        best_record: dict[str, Any] | None,
        history: list[dict[str, Any]],
        diagnosis: dict[str, Any] | None,
        time_remaining: float,
    ) -> dict[str, Any] | None:
        if not self.enabled or not self.llm.enabled:
            return None
        return self._ask_llm(best_record, history, diagnosis, time_remaining)

    def candidate_from_mutation(
        self,
        experiment_id: int,
        mutation: dict[str, Any],
        best_record: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> CandidateSpec | None:
        parent = best_record.get("id") if best_record else 0
        try:
            candidate = self.generator.from_mutation(experiment_id, mutation, parent=parent)
        except Exception:
            return None
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
        if self.generator.candidate_key(candidate) in tried:
            return None
        return candidate

    def _ask_llm(
        self,
        best_record: dict[str, Any] | None,
        history: list[dict[str, Any]],
        diagnosis: dict[str, Any] | None,
        time_remaining: float,
    ) -> dict[str, Any] | None:
        prompt = self.prompts.pair("optimization_mutation")
        user = self.prompts.render(
            prompt.user,
            {
                "SPEC_CONTEXT": self.spec.as_prompt_context(),
                "BEST_RECORD_JSON": json.dumps(best_record or {}, indent=2, sort_keys=True),
                "HISTORY_JSON": json.dumps(history[-8:], indent=2, sort_keys=True),
                "DIAGNOSIS_JSON": json.dumps(diagnosis or {}, indent=2, sort_keys=True),
                "TIME_REMAINING": f"{time_remaining:.2f}",
            },
        )
        response = self._complete(
            prompt.system,
            user,
            "optimization_mutation",
            {"time_remaining": time_remaining},
        )
        parsed = parse_json_object(response.content if response else None)
        if not parsed:
            if response is not None and self.llm_calls:
                self.llm_calls.record_parse(response.call_id, "json_object", False, "mutation JSON parse failed")
            return None
        if response is not None and self.llm_calls:
            self.llm_calls.record_parse(response.call_id, "json_object", True)
        parsed = sanitize_json_labels(parsed)
        family = parsed.get("candidate_family") or "cublas_plus_custom_rank16_update"
        if family not in {"cublas_plus_custom_rank16_update", "aten_reference"}:
            parsed["candidate_family"] = "cublas_plus_custom_rank16_update"
        return parsed

    def _complete(self, system: str, user: str, agent_name: str, metadata: dict[str, Any]):
        system = sanitize_reserved_label_text(system)
        user = sanitize_reserved_label_text(user)
        metadata = sanitize_json_labels(metadata)
        if self.llm_calls:
            return self.llm_calls.complete(self.llm, agent_name, system, user, metadata)
        return self.llm.complete(system=system, user=user)
