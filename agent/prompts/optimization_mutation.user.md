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

- Propose a pure cuBLAS three-SGEMM candidate modeled after `/workspace/lora/3_sgemm.cu` when it has not already been tried successfully.
- Use a clear mutation name like `pure_cublas_three_sgemm_3sgemm_style`; keep the candidate family compatible with the framework if needed.
- The mutation should focus on cuBLAS operation flags, leading dimensions, temporary U allocation, beta = 1 accumulation into Y, and avoiding explicit B.T materialization.
- Prefer the exact 3_sgemm-style mapping: SGEMM1 opA=N/opB=N with A=X and B=W; SGEMM2 opA=N/opB=T with A=X and B=B, U shape {d,16}, m=d, n=16, k=d, ldc=d; SGEMM3 opA=N/opB=N with A=U and B=A, m=d, n=d, k=16, beta=1.
- Prefer contiguous-input `TORCH_CHECK`s, `CUDAGuard`, current cuBLAS handle, `empty_like(W)`, and `U={d,16}`. Avoid `.contiguous()` input copies and explicit `cublasSetStream` for the first attempt.
- Do not ask for a handwritten CUDA rank-16 kernel unless the pure cuBLAS strategy has already failed or regressed.

Propose the next mutation as JSON.
