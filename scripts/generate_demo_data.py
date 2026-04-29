import csv
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.contracts import FEATURE_COLUMNS, ID_COLUMNS, TARGET_COLUMN


REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "simulated_data.csv"
PREDICTION_BATCH_PATH = PROJECT_ROOT / "data" / "demo" / "prediction_batch.csv"
TRUTH_BATCH_PATH = PROJECT_ROOT / "data" / "demo" / "truth_batch.csv"
DRIFTED_PREDICTION_BATCH_PATH = PROJECT_ROOT / "data" / "demo" / "drifted_prediction_batch.csv"
DRIFTED_TRUTH_BATCH_PATH = PROJECT_ROOT / "data" / "demo" / "drifted_truth_batch.csv"


def _student_row(student_id: int, rng: random.Random, drift: bool = False) -> dict:
    ability = rng.uniform(0.15, 0.95)
    engagement = rng.uniform(0.2, 0.95)
    instability = rng.uniform(0.0, 0.9)

    if drift:
        engagement *= rng.uniform(0.45, 0.75)
        instability = min(1.0, instability + rng.uniform(0.15, 0.35))

    row = {"id": student_id}
    term_risks = []

    for term in range(1, 8):
        fatigue = term * rng.uniform(0.005, 0.025)
        absence = min(
            0.95,
            max(0.0, 0.18 + instability * 0.35 - engagement * 0.18 + fatigue + rng.gauss(0, 0.045)),
        )
        suspension = max(
            0,
            int(round(instability * 2.4 + absence * 2.2 + rng.gauss(0, 0.8))),
        )
        mobility = max(
            0,
            int(round(instability * 3.0 + rng.gauss(0, 0.9))),
        )
        mark_base = 45 + ability * 45 + engagement * 12 - absence * 28 - suspension * 1.8
        marks = [
            min(100.0, max(0.0, mark_base + rng.gauss(0, 6.5)))
            for _ in range(4)
        ]

        row[f"absrate{term}"] = round(absence, 4)
        row[f"nsusp{term}"] = suspension
        row[f"mobility{term}"] = mobility
        for idx, mark in enumerate(marks, start=1):
            row[f"q{idx}mpa{term}"] = round(mark, 2)

        term_risks.append(absence * 0.55 + suspension * 0.08 + (60 - sum(marks) / 4) / 100)

    risk_signal = sum(term_risks) / len(term_risks)
    risk_signal += instability * 0.25
    risk_signal -= ability * 0.22
    risk_signal -= engagement * 0.18
    risk_signal += rng.gauss(0, 0.08)
    row[TARGET_COLUMN] = 1 if risk_signal > 0.22 else 0

    return row


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def main() -> None:
    rng = random.Random(42)
    truth_columns = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    prediction_columns = ID_COLUMNS + FEATURE_COLUMNS

    reference_rows = [_student_row(student_id, rng) for student_id in range(1, 221)]
    truth_rows = [_student_row(student_id, rng) for student_id in range(221, 281)]
    drifted_rows = [_student_row(student_id, rng, drift=True) for student_id in range(281, 341)]

    _write_csv(REFERENCE_PATH, reference_rows, truth_columns)
    _write_csv(PREDICTION_BATCH_PATH, truth_rows, prediction_columns)
    _write_csv(TRUTH_BATCH_PATH, truth_rows, truth_columns)
    _write_csv(DRIFTED_PREDICTION_BATCH_PATH, drifted_rows, prediction_columns)
    _write_csv(DRIFTED_TRUTH_BATCH_PATH, drifted_rows, truth_columns)

    print(f"Wrote {len(reference_rows)} rows to {REFERENCE_PATH}")
    print(f"Wrote {len(truth_rows)} rows to {PREDICTION_BATCH_PATH}")
    print(f"Wrote {len(truth_rows)} rows to {TRUTH_BATCH_PATH}")
    print(f"Wrote {len(drifted_rows)} rows to {DRIFTED_PREDICTION_BATCH_PATH}")
    print(f"Wrote {len(drifted_rows)} rows to {DRIFTED_TRUTH_BATCH_PATH}")


if __name__ == "__main__":
    main()
