"""Configuration loading and stable hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = ROOT / "configs" / "trace_ppi_frozen.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    with open(path or FROZEN_CONFIG) as handle:
        return yaml.safe_load(handle)


def override(config: Dict[str, Any], changes: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a deep copy with dotted-key overrides, e.g. {"assignment.quantile": 0.1}."""
    out = copy.deepcopy(config)
    for dotted, value in (changes or {}).items():
        node = out
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return out


def config_hash(config: Dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
