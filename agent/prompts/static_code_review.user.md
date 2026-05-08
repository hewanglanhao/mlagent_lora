Operator and submission constraints:

{{SPEC_CONTEXT}}

Candidate metadata:

{{CANDIDATE_METADATA_JSON}}

Candidate source code:

{{CANDIDATE_CODE}}

Additional review emphasis:

- For a pure cuBLAS three-SGEMM candidate, focus on cuBLAS layout equivalence, operation flags, leading dimensions, beta = 1 accumulation, stream binding, and absence of ATen matrix multiply or handwritten kernels in the core computation.

Review the candidate and return the JSON result.
