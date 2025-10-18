from __future__ import annotations
from pathlib import Path
from typing import Union

def smart_path(p: Union[str, Path]) -> str:
    """
    Local-only passthrough used by writers.py.
    Returns a string path for any input (Path or str).
    """
    return str(p)