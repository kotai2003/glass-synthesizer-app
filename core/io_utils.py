"""
Filesystem helpers shared by the GUI app and CLI dump tool.

The texture loader is intentionally generic: it walks a directory recursively
and returns every file with a recognized image extension. This keeps the door
open for non-DTD texture sources (in-house defect photos etc., per Q4) without
changing the synthesis core.
"""
from __future__ import annotations

import os
from typing import Iterable, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_recursive(root: str) -> List[str]:
    """Return absolute paths of every image file under `root`, sorted."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"not a directory: {root}")
    out: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                out.append(os.path.abspath(os.path.join(dirpath, f)))
    out.sort()
    return out


def list_images_flat(root: str) -> List[str]:
    """Return image files directly in `root` (non-recursive), sorted."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"not a directory: {root}")
    out = [
        os.path.abspath(os.path.join(root, f))
        for f in os.listdir(root)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        and os.path.isfile(os.path.join(root, f))
    ]
    out.sort()
    return out


def ensure_dirs(*paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
