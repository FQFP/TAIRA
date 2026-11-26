"""Apply a human correction to a saved failed TAIRA trajectory.

Example:
python scripts/apply_expert_correction.py --index 0 --correction "The plan skipped ..."
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.thought_distillation.pattern_distiller import PatternDistiller

PENDING = ROOT / "storage" / "thought_patterns" / "pending_failures.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=-1, help="Index in pending_failures.json; defaults to latest.")
    parser.add_argument("--correction", required=True, help="Human explanation of why the execution failed.")
    args = parser.parse_args()

    if not PENDING.exists():
        raise FileNotFoundError(f"No pending failure records: {PENDING}")
    records = json.loads(PENDING.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("There are no pending failure records.")
    index = args.index if args.index >= 0 else len(records) - 1
    if index >= len(records):
        raise IndexError(f"Failure index {index} is out of range.")

    record = records[index]
    pattern_key = record.get("used_pattern_key")

    distiller = PatternDistiller(str(ROOT / "storage" / "thought_patterns"))
    updated_key = distiller.distill_from_failure(
        record["task_route"], args.correction, pattern_key
    )
    if not updated_key:
        raise RuntimeError("Pattern refinement failed; the pending failure was kept.")
    records.pop(index)
    PENDING.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    action = "refined" if pattern_key else "created"
    print(f"Successfully {action} {updated_key}; removed corrected failure record {index}.")


if __name__ == "__main__":
    main()
