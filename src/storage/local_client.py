from pathlib import Path
from shutil import copyfileobj
from typing import BinaryIO


def write_stream(stream: BinaryIO, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as output:
        copyfileobj(stream, output)

    return path


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
