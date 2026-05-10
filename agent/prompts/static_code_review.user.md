Operator and submission constraints:

{{SPEC_CONTEXT}}

Candidate metadata:

{{CANDIDATE_METADATA_JSON}}

Candidate source code:

{{CANDIDATE_CODE}}

Additional review emphasis:

- For a pure cuBLAS three-SGEMM candidate, focus on cuBLAS layout equivalence, operation flags, leading dimensions, beta = 1 accumulation, `CUDAGuard`/cuBLAS handle usage, and absence of ATen matrix multiply or handwritten kernels in the core computation.
- For a 3_sgemm-style candidate, specifically verify SGEMM1 opA=N/opB=N with A=X/B=W; SGEMM2 opA=N/opB=T with U shape {d,16}, m=d/n=16/k=d/ldc=d; and SGEMM3 opA=N/opB=N with m=d/n=d/k=16/beta=1.
- Prefer the `/workspace/lora/3_sgemm.cu` style: contiguous checks, no input `.contiguous()` copies, `Y=empty_like(W)`, `U={d,16}`, current cuBLAS handle, and no explicit stream rebinding unless justified.

Review the candidate and return the JSON result.
