Operator and submission constraints:

{{SPEC_CONTEXT}}

Candidate metadata:

{{CANDIDATE_METADATA_JSON}}

Static review result:

{{STATIC_REVIEW_JSON}}

Compile result:

{{COMPILE_JSON}}

Correctness result:

{{CORRECTNESS_JSON}}

Benchmark result:

{{BENCHMARK_JSON}}

Profile summary:

{{PROFILE_JSON}}

Diagnosis emphasis:

- Prefer recommendations that improve or repair the pure cuBLAS three-SGEMM path: direct cuBLAS for all multiplications, no ATen matrix multiply, no handwritten kernels, no explicit B.T transpose-copy, correct temporary U layout, and beta = 1 accumulation.
- If correctness fails with small relative error or only some d values fail, first recommend the reference U contract: U shape {d,16}, second SGEMM m=d/n=16/k=d/ldc=d, third SGEMM m=d/n=d/k=16/beta=1.
- If the candidate is fast only because of result caching or repeated identical inputs, flag that as benchmark-specific rather than a general LoRA operator improvement.

Diagnose this experiment and return JSON.
