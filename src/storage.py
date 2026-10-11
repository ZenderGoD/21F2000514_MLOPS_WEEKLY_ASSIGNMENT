"""Small helpers so the same code reads/writes GCS (gs://...) or a local folder (for testing)."""
import os

import fsspec


def join(*parts):
    return "/".join(p.strip("/") if i else p.rstrip("/") for i, p in enumerate(parts))


def open_file(path, mode="r"):
    if "w" in mode and not path.startswith("gs://"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return fsspec.open(path, mode).open()


def list_dirs(path):
    """Return the names of the immediate sub-folders of `path`."""
    fs, root = fsspec.core.url_to_fs(path)
    if not fs.exists(root):
        return []
    return sorted(os.path.basename(p.rstrip("/")) for p in fs.ls(root, detail=False) if fs.isdir(p))
