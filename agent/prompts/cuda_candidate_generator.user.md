Operator and submission constraints:

{{SPEC_CONTEXT}}

Requested candidate strategy:

{{CANDIDATE_STRATEGY_JSON}}

Recent failures and benchmark evidence:

{{EXPERIMENT_HISTORY_JSON}}

Current best code summary:

{{BEST_CODE_SUMMARY}}

Implementation direction:

- Generate a pure cuBLAS three-SGEMM implementation modeled after `/workspace/lora/3_sgemm.cu` when compatible with the requested strategy.
- Use direct cuBLAS calls for the main term, the low-rank intermediate, and the final beta = 1 accumulation.
- Prefer the exact 3_sgemm layout contract: SGEMM1 opA=N/opB=N with A=X and B=W; SGEMM2 opA=N/opB=T with A=X and B=B, U shape {d,16}, m=d, n=16, k=d, ldc=d; SGEMM3 opA=N/opB=N with A=U and B=A, m=d, n=d, k=16, beta=1.
- Prefer contiguous-input checks rather than input copies, `CUDAGuard`, `getCurrentCUDABlasHandle`, `Y=empty_like(W)`, and `U={d,16}`. Avoid explicit `cublasSetStream` and `getCurrentCUDAStream` in the first attempt.
- Avoid previous failed variants: opA=T/opB=T in SGEMM1 or SGEMM3, treating U as a [16,d] column-major buffer, m=16/n=d for U, or U leading dimension 16.
- Do not include a full code comment copied from examples; explain only the layout reasoning needed to prevent mistakes.
- Avoid ATen matrix multiply calls, handwritten CUDA kernels, and explicit B.T transpose-copy materialization for this strategy.

Generate one complete candidate CUDA source file.
