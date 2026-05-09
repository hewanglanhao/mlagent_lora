Operator and submission constraints:

{{SPEC_CONTEXT}}

Requested candidate strategy:

{{CANDIDATE_STRATEGY_JSON}}

Recent failures and benchmark evidence:

{{EXPERIMENT_HISTORY_JSON}}

Current best code summary:

{{BEST_CODE_SUMMARY}}

Implementation direction:

- Generate a pure cuBLAS three-SGEMM implementation when compatible with the requested strategy.
- Use direct cuBLAS calls for the main term, the low-rank intermediate, and the final beta = 1 accumulation.
- Prefer the qyh-style layout contract: U has shape {d,16}; the low-rank intermediate SGEMM writes U with m=d, n=16, k=d and U leading dimension d; the final SGEMM accumulates using m=d, n=d, k=16 and beta=1.
- Avoid the previous failed variant that treats U as a [16,d] column-major buffer or uses U leading dimension 16.
- Do not include a full code comment copied from examples; explain only the layout reasoning needed to prevent mistakes.
- Avoid ATen matrix multiply calls, handwritten CUDA kernels, and explicit B.T transpose-copy materialization for this strategy.

Generate one complete candidate CUDA source file.
