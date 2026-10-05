# Project vocabulary in plain English

Use this when a word in the code or interview guide is unfamiliar. These
definitions explain how the words are used in Aster Payroll Gym. They are not
claims that an experiment has already run.

## Tasks and correctness

- **AI agent:** a model acting on a task, optionally using tools, then submitting
  an answer. Here it reads payroll evidence and returns structured JSON.
- **Gym or environment:** the place that gives the agent tasks and scores its
  work. It includes the task contract, tools and reward, not just a web page.
- **Synthetic data:** invented examples, with no real employee/customer records.
  Aster and AST currency are fictional.
- **Rule set:** the written authority defining what to do. ASTER-1.0 is the version
  used here; changing it requires new versioned evidence.
- **Clause:** one numbered part of the rulebook, such as R4 for the contribution
  ceiling. Clause IDs make explanations and penalties traceable.
- **Ground truth or oracle answer:** the expected result or blockers calculated
  by Python from the rules. "Oracle" does not mean infallible; the implementation
  still needs tests and independent review.
- **Two uses of reference:** the Python reference calculator determines correct
  payroll. The frozen reference policy is the initial language model used to
  regularize training. They are different things.
- **Verifier:** a programmatic checker comparing the submitted content with the
  expected content and relevant clauses.
- **Schema or contract:** the exact allowed fields and types. An answer has six
  keys; a money value must be an integer number of cents.
- **JSON:** a structured text format with objects, lists, strings, numbers and
  `null`. Correct JSON syntax does not imply correct payroll.
- **Authority:** which document is entitled to decide a value. A latest applicable
  signed salary takes precedence over stale, unsigned or future records.
- **YTD:** year to date. Here, pensionable earnings in the payment year **before**
  this payroll, needed to calculate the remaining annual ceiling.
- **Marginal tax:** each slice of income is taxed at its own band rate. Crossing
  into a higher band does not apply that rate to all income.
- **Rounding half up:** round to the nearest cent, with an exact half cent rounded
  upward. The rules specify where to round; do not round every intermediate step.
- **Abstention or clarification:** declining to calculate and asking for specified
  missing/resolved evidence. It is right only when a genuine blocker exists.

## Scoring and experiments

- **Reward:** a score from 0 to 1 describing the submitted work under the rubric.
  It can guide training or measure evaluation performance.
- **Reward weights versus model weights:** reward weights are the fixed rubric
  percentages. Model weights are learned numerical parameters that control model
  behaviour; LoRA updates adapters while leaving the backbone frozen.
- **Partial credit:** limited feedback for correct fields or blockers in an
  imperfect answer. It does not make rejected work a pass.
- **Hard gate:** a rule overriding the weighted sum. Unsafe computation scores
  zero even if the JSON is tidy.
- **LLM judge:** a remote model grading the usefulness of an eligible blocking
  explanation. It does not decide arithmetic or create ground truth.
- **Calibration:** matching action to evidence: answer when solvable; ask for
  the specific required information when genuinely unresolved. This project
  measures decision calibration, not a probability calibration curve.
- **Baseline:** a deliberately simple comparison policy. The current baselines
  are handwritten program outputs, not actual model evaluations.
- **Reward gaming:** finding a way to score well without doing the intended work.
  Always refusing, copying citations or exploiting formatting are possible attacks
  to test, not proof of exploits found during actual training.
- **Rollout:** one attempt by an agent to solve one task, including tool turns.
  Three stochastic rollouts mean three sampled attempts, not three identical copies.
- **Seed:** a number controlling repeatable task generation or sampling settings.
  The same task-generator seed/config recreates the same tasks. It does not make
  a remote provider's outputs perfectly repeatable.
- **Fingerprint or hash:** a compact identifier of normalized inputs or file
  contents. It helps detect duplicates or changes. It does not prove the model
  named in a file actually produced the answers.
- **Train split:** tasks used for optimizer updates—the practice material.
- **Validation split:** separate tasks used to select beta—the tuning material.
- **Held-out evaluation split:** tasks kept out of training and selection—the
  final exam. Do not adjust the experiment after inspecting its results.
- **Transfer test:** unchanged rules with unfamiliar presentations, to probe
  whether behaviour depends on the usual task templates.
- **Mean ± SD:** average reward and sample standard deviation of the complete
  rollout-level means. SD describes variation; it is not itself a confidence
  interval or proof that a ranking is meaningful.
- **Coverage:** how much of the expected evaluation actually has completed scores.
  A high mean with missing judgments must be read alongside coverage.
- **Mean reward versus pass rate:** mean reward includes partial credit. Pass rate
  counts only complete scores at or above .975. They answer different questions.
- **Transcript:** the recorded prompt, response and tool calls. It is the evidence
  needed to inspect a model's actual behaviour.
- **Token:** a model's unit of text processing. Token limits bound prompt/answer
  size; provider token counts are used for cost accounting.
- **Latency:** how long a request or rollout takes. Local baseline timings do not
  describe hosted model or sandbox performance.

## Training and engineering

- **Inference:** using a model to produce an answer without changing its weights.
  Qwen inference still needs Colab here; it is not a permitted laptop task.
- **Reinforcement learning (RL):** updating the policy using scores of sampled
  answers. Implemented training is not evidence of successful learning until run.
- **Policy:** the model's current distribution over possible answers.
- **GRPO:** the chosen training method. Four answers to the same prompt receive
  relative advantages based on their group rewards, then a clipped objective
  updates the policy. Equal rewards give zero reward preference.
- **Optimizer step:** one model update after any configured gradient accumulation.
  It is not one task or one rollout. A group here has four sampled completions.
- **LoRA:** small trainable adapter matrices added to a frozen model backbone.
  This reduces the trainable parameter count; the model still runs on cloud GPU.
- **Frozen reference:** the unchanged initial policy used as a comparison during
  training. Here the backbone with fresh adapters disabled is checked as reference.
- **KL and beta:** the sampled KL estimator measures departure from the reference;
  beta weights its penalty in training loss. Larger beta means a stronger penalty
  in this configuration, not a guaranteed better final policy.
- **Entropy:** how spread out the model's token choices are. Falling entropy can
  indicate increasingly concentrated outputs, but needs transcript context.
- **Checkpoint:** cloud-saved model/optimizer state allowing compatible training
  to resume. It stays in Drive rather than being downloaded to your Mac.
- **Runtime, GPU and CUDA:** the runtime is the computer executing the notebook.
  A GPU accelerates model work; CUDA is the required NVIDIA GPU platform here.
  Colab must provide an available CUDA GPU, with no automatic laptop fallback.
- **API:** a contract for software to send requests and receive responses.
  The sandbox has task issuance, submission and run-retrieval endpoints.
- **Public DTO:** an explicit response data shape that allowlists public fields,
  excluding server-owned expected answers and private task provenance.
- **Idempotent retry:** submitting the identical answer again does not create a
  different answer attempt. A changed answer in that run is rejected.
- **Cost cap:** the maximum accounted spend allowed. Unknown pricing blocks model
  evaluation; declaring a zero price requires checked free-tier eligibility.
- **Bundle:** the compact JSON/JSONL archive exported from cloud runs. Import
  validates paths, sizes, versions, checksums and splits; it loads no weights.
- **CI:** automated checks run by GitHub on code changes. Green CI confirms those
  checks passed, not that all cloud experiments or human reviews happened.
- **Static dashboard:** a website rendered from saved records. It is separate
  from the live sandbox server and makes no model calls when viewed.

Next: [the full project explanation](project-understanding.md) and
[interview preparation](interview-preparation.md).
