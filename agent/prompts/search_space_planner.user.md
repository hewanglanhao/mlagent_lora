Operator and submission constraints:

{{SPEC_CONTEXT}}

Recent experiment history:

{{HISTORY_JSON}}

Current deterministic fallback plan:

- Keep an ATen/cuBLAS baseline as the always-valid best candidate.
- Give high priority to a precompute candidate named like `candidate_002_precompute.cu`: form `Weff = W + A @ B.T`, then compute `Weff @ X`.
- Explore cuBLAS W @ X plus custom rank-16 Y += A @ T kernels.
- Validate every candidate by static review, compile, correctness tests, and benchmark before promotion.

Produce the next search-space plan as JSON.
