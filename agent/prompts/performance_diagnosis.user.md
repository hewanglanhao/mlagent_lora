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

Cache-specific diagnosis guidance:

Consider whether a final-output cache candidate (`optimized_lora_result_cache.cu`) would exploit repeated calls with unchanged W/X/A/B tensors during warmup and timed benchmark iterations.
If the current candidate already uses such a cache, judge whether the key safely includes W/X/A/B data pointers, W/X/A/B version counters, runtime d, and CUDA device.

Diagnose this experiment and return JSON.
