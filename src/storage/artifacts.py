from pathlib import Path

from core.config import MINIO_MIRROR_ARTIFACTS, STORAGE_BACKEND


def object_storage_enabled() -> bool:
    return STORAGE_BACKEND == "minio" or MINIO_MIRROR_ARTIFACTS


def mirror_file_to_object_store(
    local_path: str | Path,
    object_key: str,
    content_type: str | None = None,
) -> dict | None:
    if not object_storage_enabled():
        return None

    from storage.minio_client import upload_file

    return upload_file(
        local_path=local_path,
        object_key=object_key,
        content_type=content_type,
    )


def compact_artifact_map(artifacts: dict[str, dict | None]) -> dict[str, dict]:
    return {name: info for name, info in artifacts.items() if info is not None}
