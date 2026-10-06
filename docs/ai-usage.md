# Tool-use disclosure

Codex assisted with requirements, domain/rule design, code, tests, deployment
configuration and documentation, including specialized code-review subagents.
No other person collaborated. Ground-truth labels are computed exclusively by the
independent Python reference; task prose and review drafts received assistance.

No local model training, inference or weight downloads were performed. Gemini 3.5 Flash-Lite remote evaluation and real Gemini judge calls ran through
Google APIs; Gemini 3.8 evaluation remains partial after rate limits. Colab/Qwen/RL
and independent human reviews remain pending. The hosted Render/Neon sandbox passed independent Tier-2/3 submissions, access
and leakage checks, and persistence after the owner-confirmed service restart.
