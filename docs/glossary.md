# Plain-English glossary

Definitions describe the implementation, not proof that experiments ran.

- **Gym:** a system issuing tasks, tools and scores to an agent.
- **Synthetic:** fictional data and rules, without real employee records.
- **Clause:** a numbered rule, such as R4 for the contribution ceiling.
- **Ground truth:** the expected answer or blockers computed by Python.
- **Verifier:** code checking content against that ground truth.
- **Schema:** the allowed fields and types in a task or answer.
- **YTD:** pensionable earnings in the pay-date year before this payroll.
- **Marginal tax:** each income slice pays its own rate, rather than applying the
  highest reached rate to all income.
- **Half-up rounding:** round to the nearest cent; an exact half cent rounds up.
- **Clarification/abstention:** request necessary information instead of computing;
  useful only when the evidence genuinely prevents an answer.
- **Reward:** the fixed 0–1 score for an answer. Model weights are different:
  they are learned parameters controlling model behavior.
- **Partial credit:** bounded credit for correct parts of imperfect work.
- **Hard gate:** a condition overriding the weighted sum, such as unsafe computation.
- **Judge:** a remote model assessing only an eligible blocking explanation.
- **Calibration:** answering when supported and requesting specific evidence when not.
- **Baseline:** a simple comparison policy; handwritten attacks are not model runs.
- **Reward gaming:** scoring well without doing the intended useful work.
- **Rollout:** one sampled attempt, including any tool calls.
- **Seed:** a repeatable generation setting; it cannot guarantee remote model determinism.
- **Fingerprint:** a normalized input identifier detecting duplicates or changes.
- **Training:** practice tasks used for optimizer updates.
- **Validation:** separate tasks used to select experiment settings.
- **Held-out evaluation:** untouched final-exam tasks used after selection.
- **Transfer:** changed presentations with unchanged rules.
- **Mean±SD:** average and sample variation of complete rollout means; not a
  confidence interval. **Coverage** shows how many expected scores completed.
- **Inference:** producing model answers without updating parameters.
- **RL:** updating a policy using rewards from sampled answers.
- **GRPO:** learning from relative rewards within groups of answers to one prompt.
- **LoRA:** small trainable adapters attached to a frozen model backbone.
- **Frozen reference:** the unchanged starting policy; unlike the Python reference
  calculator, it does not determine correct payroll.
- **KL / beta:** measured movement from the reference / weight on its penalty.
- **Entropy:** how spread out the model's token choices are.
- **Checkpoint:** cloud-saved model and optimizer state for compatible resumption.
- **Transcript:** actual prompts, responses and tool calls retained for review.
- **API / public DTO:** request-response interface / explicit public field allowlist.
- **Idempotent:** an identical submission retry does not create a different attempt.
- **CI:** automated code checks; green CI does not prove learning occurred.
- **Bundle:** checked JSON results returned from cloud, never model checkpoints.

Continue with [project explanation](project-understanding.md).
