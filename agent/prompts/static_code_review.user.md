Operator and submission constraints:

{{SPEC_CONTEXT}}

Candidate metadata:

{{CANDIDATE_METADATA_JSON}}

Candidate source code:

{{CANDIDATE_CODE}}

Cache review note:

If this candidate implements `optimized_lora_result_cache.cu` style final-output caching, check that it keys the cache by W/X/A/B data pointers, W/X/A/B version counters, runtime d, and CUDA device.
Flag any stale-result risk as an error or high-risk warning.

Review the candidate and return the JSON result.
