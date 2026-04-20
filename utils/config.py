from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
