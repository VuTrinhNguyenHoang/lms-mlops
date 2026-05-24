from pathlib import Path

from core.config import LOCAL_STORAGE_DIR


TRUTH_BATCH_DIR = LOCAL_STORAGE_DIR / "raw" / "truth"


def _batch_id_from_path(path: Path) -> str:
    return path.stem


def _truth_csv_paths() -> list[Path]:
    if not TRUTH_BATCH_DIR.exists():
        return []
    return [path for path in TRUTH_BATCH_DIR.glob("*.csv") if path.is_file()]


def latest_truth_path(exclude_batch_id: str | None = None) -> Path | None:
    paths = _truth_csv_paths()
    if exclude_batch_id is not None:
        paths = [
            path
            for path in paths
            if _batch_id_from_path(path) != exclude_batch_id
        ]

    if not paths:
        return None

    return max(paths, key=lambda path: path.stat().st_mtime)


def recent_truth_paths(
    max_batches: int,
    required_path: str | None = None,
) -> list[Path]:
    if max_batches < 1:
        raise ValueError("max_batches must be >= 1")

    newest_first = sorted(
        _truth_csv_paths(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    selected = newest_first[:max_batches]

    if required_path is not None:
        required = Path(required_path)
        if required not in selected:
            selected.insert(0, required)
            selected = selected[:max_batches]

    # Build datasets oldest-to-newest so duplicate IDs keep the newest truth row.
    return list(reversed(selected))
