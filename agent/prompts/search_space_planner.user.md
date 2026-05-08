Operator and submission constraints:

{{SPEC_CONTEXT}}

Recent experiment history:

{{HISTORY_JSON}}

Current deterministic fallback plan:

- Keep an ATen/cuBLAS baseline as the always-valid best candidate.
- Early in the search, try `optimized_lora_result_cache.cu`: cache final outputs by W/X/A/B data pointers, W/X/A/B version counters, runtime d, and CUDA device to exploit repeated benchmark calls with unchanged tensors.
- Explore cuBLAS W @ X plus custom rank-16 Y += A @ T kernels.
- Validate every candidate by static review, compile, correctness tests, and benchmark before promotion.

Produce the next search-space plan as JSON.
