Operator and submission constraints:

{{SPEC_CONTEXT}}

Recent experiment history:

{{HISTORY_JSON}}

Current deterministic fallback plan:

- Keep an ATen/cuBLAS baseline as the always-valid best candidate.
- First explore a pure cuBLAS three-SGEMM candidate family.
- The preferred candidate should avoid ATen `mm`, avoid handwritten CUDA kernels, avoid explicit B.T materialization, and use cuBLAS transpose flags plus row-major/column-major equivalence.
- The low-rank update should be accumulated into Y through the third SGEMM with beta = 1.
- Validate every candidate by static review, compile, correctness tests, and benchmark before promotion.

Produce the next search-space plan as JSON.
