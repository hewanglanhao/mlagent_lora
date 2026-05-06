from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StaticReviewResult:
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    risk_level: str = "low"

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "risk_level": self.risk_level,
        }


class StaticCodeReviewAgent:
    allowed_external_suffixes = (".h", ".hpp", ".cuh", ".cu", ".cpp")

    def review(self, source_path: Path) -> StaticReviewResult:
        text = source_path.read_text(encoding="utf-8")
        errors: list[str] = []
        warnings: list[str] = []

        if "#include <torch/extension.h>" not in text:
            errors.append("missing #include <torch/extension.h>")
        if "PYBIND11_MODULE" not in text:
            errors.append("missing PYBIND11_MODULE")
        if not re.search(r"torch::Tensor\s+forward\s*\(", text):
            errors.append("missing torch::Tensor forward(...) entrypoint")
        if len(re.findall(r"torch::Tensor\s+\w+", self._signature_text(text))) < 4:
            warnings.append("forward signature could not be fully verified")
        if "data_ptr<float>" in text and "at::kFloat" not in text:
            warnings.append("uses data_ptr<float>() without an explicit float32 check")
        if "size(0)" not in text:
            errors.append("does not appear to read d dynamically from tensor shape")
        if re.search(r"\bd\s*==\s*(3584|3712|3840|4096|4352|4480|4608)\b", text):
            errors.append("candidate appears to hardcode an exact hidden dimension")
        if re.search(r"#include\s+\".+\.(?:cu|cuh|h|hpp|cpp)\"", text):
            errors.append("candidate depends on a submission-side source/header file")
        if "TORCH_CHECK" not in text:
            warnings.append("no TORCH_CHECK validation")
        if "contiguous()" not in text:
            warnings.append("does not make tensors contiguous before pointer/matmul use")

        risk_level = "low"
        if warnings:
            risk_level = "medium"
        if errors:
            risk_level = "high"

        return StaticReviewResult(
            passed=not errors,
            warnings=warnings,
            errors=errors,
            risk_level=risk_level,
        )

    def _signature_text(self, text: str) -> str:
        match = re.search(r"torch::Tensor\s+forward\s*\((.*?)\)\s*\{", text, re.S)
        return match.group(1) if match else ""
