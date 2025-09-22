"""
Fuka 5.0 — Space–Time Capacitor Substrate

This package runs the world/graph/substrate simulation, writes shards & volumes to GCS,
and provides utilities for env-driven configuration.

Public helpers exported here:
- __version__
- env_get(key, default=None)
- env_expand(value)                  # expand ${VARS} and $VARS in strings
- expand_env_in_dict(d: dict)        # recursively expand strings in nested dicts/lists
- load_json_with_env(path)           # read JSON file and expand env placeholders
"""

from __future__ import annotations
import os
import json
import re
from typing import Any, Dict

__all__ = [
    "__version__",
    "env_get",
    "env_expand",
    "expand_env_in_dict",
    "load_json_with_env",
]

__version__ = "5.0.0"

_ENV_PATTERN = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")

def env_get(key: str, default: str | None = None) -> str | None:
    """Read environment variable with optional default."""
    return os.environ.get(key, default)

def env_expand(value: str) -> str:
    """
    Expand $VARNAME or ${VARNAME} in a string using os.environ.
    Unset variables are replaced by empty string.
    """
    def _sub(m: re.Match[str]) -> str:
        var = m.group(1)
        return os.environ.get(var, "")
    return _ENV_PATTERN.sub(_sub, value)

def expand_env_in_dict(d: Any) -> Any:
    """
    Recursively expand environment placeholders in a nested structure.
    Applies only to str; leaves numbers/bools/None untouched.
    """
    if isinstance(d, dict):
        return {k: expand_env_in_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [expand_env_in_dict(v) for v in d]
    if isinstance(d, str):
        return env_expand(d)
    return d

def load_json_with_env(path: str) -> Dict[str, Any]:
    """
    Load JSON file then expand environment placeholders in all string values.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return expand_env_in_dict(obj)