# LLM Prompt Map

These prompts are written in English and kept out of the Python control logic.
The optimizer can run without an LLM, but when an OpenAI-compatible model is
configured these files define the behavior of each LLM-assisted agent.

Runtime LLM-assisted modules:

- `search_space_planner.*.md`: used by `SearchSpacePlannerAgent`.
- `cuda_candidate_generator.*.md`: for generating candidate single-file CUDA implementations.
- `static_code_review.*.md`: for LLM-assisted pre-compile code review.
- `performance_diagnosis.*.md`: for translating benchmark/profile data into bottleneck diagnoses.
- `optimization_mutation.*.md`: for selecting the next mutation after an experiment.

All LLM-assisted modules have deterministic fallbacks. The local validation
pipeline remains authoritative: static review, compile, correctness, benchmark,
and best-candidate promotion are never bypassed by an LLM response.

Useful environment switches:

- `ENABLE_LLM_CODEGEN=0` disables LLM CUDA source generation.
- `ENABLE_LLM_STATIC_REVIEW=0` disables LLM static review.
- `LLM_STATIC_REVIEW_CAN_REJECT=1` allows LLM static review to reject candidates.
- `ENABLE_LLM_DIAGNOSIS=0` disables LLM performance diagnosis.
- `ENABLE_LLM_MUTATION=0` disables LLM mutation planning.
