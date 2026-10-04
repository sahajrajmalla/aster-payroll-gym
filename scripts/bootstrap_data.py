"""Create the frozen, tiny synthetic corpus. No model libraries or network calls."""

import json
from pathlib import Path

from aster_gym.generator import generate_taskset, task_from_inputs, validate_splits, write_taskset
from aster_gym.reference import authoritative_schedules
from aster_gym.schemas import Answer, Task
from aster_gym.versions import GENERATOR_VERSION, REWARD_VERSION, rules_hash, stable_hash

ROOT = Path(__file__).resolve().parents[1]


def record():
    return {"kind": "payslip", "pay_date": "2030-01-31", "salary_records": [
        {"effective": "2030-01-01", "signed": True, "monthly_salary_cents": 300000}],
        "paid_days": 31, "period_days": 31, "bonus_cents": 0,
        "ytd_pensionable_cents": 0, "ytd_year": 2030, "schedules": authoritative_schedules()}


def main():
    partitions = {"train": generate_taskset(30, seed=1101, split="train"),
                  "validation": generate_taskset(15, seed=2201, split="validation"),
                  "evaluation": generate_taskset(120, seed=7001, split="evaluation")}
    seeds = []
    for year in (2029, 2030):
        for field in ("allowance_cents", "annual_ceiling_cents"):
            inputs = {"kind": "retrieval", "pay_date": f"{year}-06-15", "query_field": field,
                      "schedules": authoritative_schedules()}
            seeds.append(task_from_inputs(inputs, seed=7, difficulty=1, tags=["seed", "human_review_pending"]))
    for variant in range(4):
        inputs = record()
        if variant == 1:
            inputs["ytd_pensionable_cents"] = 1199000
            inputs["salary_records"][0]["monthly_salary_cents"] = 500000
        if variant == 2:
            inputs.update(pay_date="2030-04-30", period_days=30, paid_days=1,
                          ytd_pensionable_cents=1200000)
            inputs["salary_records"][0]["monthly_salary_cents"] = 100005
        if variant == 3:
            inputs.update(pay_date="2030-02-28", period_days=28, paid_days=14, bonus_cents=12345)
            inputs["salary_records"][0]["monthly_salary_cents"] = 600001
        seeds.append(task_from_inputs(inputs, seed=7, difficulty=2, tags=["seed", "human_review_pending"]))
    for variant in range(4):
        inputs = record()
        if variant == 0:
            del inputs["ytd_pensionable_cents"]
        elif variant == 1:
            inputs["salary_records"].append({"effective": "2030-01-01", "signed": True,
                                            "monthly_salary_cents": 400000})
        elif variant == 2:
            inputs.update(pay_date="2031-01-31", ytd_year=2031)
        else:
            del inputs["bonus_cents"]
        seeds.append(task_from_inputs(inputs, seed=7, difficulty=3, tags=["seed", "human_review_pending"]))

    # Explicitly authored presentations, independent of generator templates; user review still pending.
    transfer = []
    inputs = record()
    inputs.update(pay_date="2030-03-31", period_days=31, paid_days=23, bonus_cents=8765,
                  ytd_pensionable_cents=1177777)
    transfer.append(task_from_inputs(inputs, seed=99001, difficulty=2, tags=["transfer", "review_pending"],
        narrative="Finance handover: payment is scheduled for 31 March 2030. The attendance file counts 23 "
                  "paid calendar days. A bonus was approved separately. Reconcile the supplied records and "
                  "prepare all five payslip lines; do not treat an informal planning note as a salary change."))
    inputs = record()
    inputs["salary_records"] += [{"effective": "2030-01-15", "signed": True, "monthly_salary_cents": 444444},
                                {"effective": "2030-01-25", "signed": False, "monthly_salary_cents": 555555}]
    inputs["ytd_pensionable_cents"] = 1205000
    transfer.append(task_from_inputs(inputs, seed=99002, difficulty=2, tags=["transfer", "review_pending"],
        narrative="An unsigned offer differs from two signed records. Payroll has already exceeded the "
                  "year's pensionable ceiling. Read the dates and authority carefully and issue January's payslip."))
    inputs = record()
    inputs["salary_records"].append({"effective": "2030-01-01", "signed": True, "monthly_salary_cents": 320000})
    transfer.append(task_from_inputs(inputs, seed=99003, difficulty=3, tags=["transfer", "review_pending"],
        narrative="Two signed amendments arrived in separate attachments. Both start on the same date; neither "
                  "supersedes the other. Finance requests a payslip today. State whether this can be completed "
                  "and give a concrete next action using the supplied evidence."))
    inputs = record()
    del inputs["ytd_pensionable_cents"]
    inputs.update(pay_date="2030-09-30", period_days=30, paid_days=30)
    transfer.append(task_from_inputs(inputs, seed=99004, difficulty=3, tags=["transfer", "review_pending"],
        narrative="A September handover includes salary and attendance, but the prior earnings register has "
                  "not arrived. A note suggests assuming zero. Determine what the rules actually permit."))
    inputs = {"kind": "retrieval", "pay_date": "2029-12-30", "query_field": "allowance_cents",
              "schedules": authoritative_schedules()}
    transfer.append(task_from_inputs(inputs, seed=99005, difficulty=1, tags=["transfer", "review_pending"],
        narrative="For the payment dated 30 December 2029, extract only the monthly allowance from the "
                  "applicable schedule. An attachment also contains next year's schedule; do not use it early."))
    partitions.update(seeds=seeds, transfer=transfer)
    validate_splits(partitions)
    for split, tasks in partitions.items():
        write_taskset(tasks, ROOT / "data" / f"{split}.jsonl")
    manifest = {"generator_version": GENERATOR_VERSION, "reward_version": REWARD_VERSION,
                "rules_hash": rules_hash(), "human_review_status": "pending",
                "partitions": {name: {"n": len(tasks), "seeds": sorted({t.seed for t in tasks}),
                                      "ids": [t.id for t in tasks], "input_hashes": [t.input_hash for t in tasks],
                                      "hash": stable_hash([t.model_dump(mode="json") for t in tasks])}
                               for name, tasks in partitions.items()}}
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    schemas = ROOT / "schemas"
    schemas.mkdir(exist_ok=True)
    for name, model in (("task", Task), ("answer", Answer)):
        (schemas / f"{name}-v1.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
    review = ROOT / "reviews"
    review.mkdir(exist_ok=True)
    fidelity = [partitions["evaluation"][i] for i in (0, 1, 2, 4, 5, 7, 8, 10, 11, 14)]
    packet = {"instructions": "Read rules first. Independently calculate without looking at reference_result. "
                              "Then compare. Only the human reviewer can set reviewed=true.",
              "seed_reviews": [{"task_id": t.id, "inputs": t.inputs,
                                "reference_result": t.ground_truth.model_dump(), "reviewed": False,
                                "human_result": None, "reviewer": None, "notes": ""} for t in seeds],
              "fidelity_reviews": [{"task_id": t.id, "inputs": t.inputs,
                                     "reference_result": t.ground_truth.model_dump(), "reviewed": False,
                                     "human_result": None, "reviewer": None, "notes": ""} for t in fidelity],
              "transfer_reviews": [{"task_id": t.id, "prompt": t.prompt, "inputs": t.inputs,
                                     "reference_result": t.ground_truth.model_dump(), "reviewed": False,
                                     "human_result": None, "reviewer": None, "notes": ""} for t in transfer]}
    packet_path = review / "task-review-packet.json"
    current = json.loads(packet_path.read_text()) if packet_path.exists() else None
    already_reviewed = current and any(row.get("reviewed") for section in (
        "seed_reviews", "fidelity_reviews", "transfer_reviews") for row in current.get(section, []))
    if not already_reviewed:
        packet_path.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"partitions": {k: len(v) for k, v in partitions.items()}, "model_calls": 0}))


if __name__ == "__main__":
    main()
