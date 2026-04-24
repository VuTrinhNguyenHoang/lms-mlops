from pathlib import Path

import pandas as pd

from core.config import LOCAL_MERGED_TRAINING_DIR
from core.contracts import ID_COLUMNS, TRUTH_REQUIRED_COLUMNS
from data.loading import load_csv, save_csv
from data.validation import validate_truth_df


def _load_validated_truth(path: str):
    return validate_truth_df(load_csv(path))


def _default_output_path(batch_id: str) -> Path:
    return LOCAL_MERGED_TRAINING_DIR / f"{batch_id}.csv"


def build_retrain_dataset(
    reference_path: str,
    truth_paths: list[str],
    batch_id: str,
    output_path: str | None = None,
) -> dict:
    if not truth_paths:
        raise ValueError("At least one truth path is required to build retrain dataset")

    reference_df = _load_validated_truth(reference_path)
    truth_dfs = [_load_validated_truth(path) for path in truth_paths]
    truth_df = pd.concat(truth_dfs, ignore_index=True)

    merged_df = pd.concat(
        [reference_df, truth_df],
        ignore_index=True,
    )
    rows_before_dedup = len(merged_df)
    merged_df = (
        merged_df[TRUTH_REQUIRED_COLUMNS]
        .drop_duplicates(subset=ID_COLUMNS, keep="last")
        .reset_index(drop=True)
    )

    output_file = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(batch_id)
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(merged_df, str(output_file))

    return {
        "batch_id": batch_id,
        "reference_path": str(reference_path),
        "truth_paths": [str(path) for path in truth_paths],
        "output_path": str(output_file),
        "reference_rows": int(len(reference_df)),
        "truth_rows": int(len(truth_df)),
        "rows_before_dedup": int(rows_before_dedup),
        "training_rows": int(len(merged_df)),
        "duplicate_rows_dropped": int(rows_before_dedup - len(merged_df)),
    }
