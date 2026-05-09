Operator and submission constraints:

{{SPEC_CONTEXT}}

Current best record:

{{BEST_RECORD_JSON}}

Recent experiment history:

{{HISTORY_JSON}}

Most recent diagnosis:

{{DIAGNOSIS_JSON}}

Remaining time budget in seconds:

{{TIME_REMAINING}}

Preferred next direction:

- Propose a pure cuBLAS three-SGEMM candidate when it has not already been tried successfully.
- Use a clear mutation name like `pure_cublas_three_sgemm`; keep the candidate family compatible with the framework if needed.
- The mutation should focus on cuBLAS operation flags, leading dimensions, temporary U allocation, beta = 1 accumulation into Y, and avoiding explicit B.T materialization.
- Prefer the qyh-style mapping: U is {d,16}; the second SGEMM uses m=d, n=16, k=d and U leading dimension d; the third SGEMM uses m=d, n=d, k=16 and beta=1.
- Do not ask for a handwritten CUDA rank-16 kernel unless the pure cuBLAS strategy has already failed or regressed.

Propose the next mutation as JSON.
