Operator and submission constraints:

{{SPEC_CONTEXT}}

Candidate metadata:

{{CANDIDATE_METADATA_JSON}}

Candidate source code:

{{CANDIDATE_CODE}}

Additional review emphasis:

- For a pure cuBLAS three-SGEMM candidate, focus on cuBLAS layout equivalence, operation flags, leading dimensions, beta = 1 accumulation, stream binding, and absence of ATen matrix multiply or handwritten kernels in the core computation.
- For a qyh-style candidate, specifically verify U shape {d,16}, second SGEMM m=d/n=16/k=d/ldc=d, and third SGEMM m=d/n=d/k=16/beta=1.

Review the candidate and return the JSON result.
