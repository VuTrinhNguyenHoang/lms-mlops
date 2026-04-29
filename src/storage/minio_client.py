from pathlib import Path
from mimetypes import guess_type

from minio import Minio

from core.config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
)


def get_minio_client() -> Minio:
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def ensure_bucket(client: Minio | None = None) -> None:
    client = client or get_minio_client()
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def upload_file(
    local_path: str | Path,
    object_key: str,
    content_type: str | None = None,
) -> dict:
    path = Path(local_path)
    client = get_minio_client()
    ensure_bucket(client)

    guessed_type, _ = guess_type(path.name)
    client.fput_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_key,
        file_path=str(path),
        content_type=content_type or guessed_type or "application/octet-stream",
    )

    return {
        "bucket": MINIO_BUCKET,
        "object_key": object_key,
        "endpoint": MINIO_ENDPOINT,
        "size_bytes": path.stat().st_size,
    }


def download_file(object_key: str, local_path: str | Path) -> dict:
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    client = get_minio_client()
    client.fget_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_key,
        file_path=str(path),
    )

    return {
        "bucket": MINIO_BUCKET,
        "object_key": object_key,
        "local_path": str(path),
    }
