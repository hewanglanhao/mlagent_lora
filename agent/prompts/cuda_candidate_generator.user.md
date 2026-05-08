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
- Do not include a full code comment copied from examples; explain only the layout reasoning needed to prevent mistakes.
- Avoid ATen matrix multiply calls, handwritten CUDA kernels, and explicit B.T transpose-copy materialization for this strategy.

Generate one complete candidate CUDA source file.
