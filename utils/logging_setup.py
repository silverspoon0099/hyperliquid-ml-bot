from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from .config import load_config, resolve_path

_configured = False


def get_logger(name: str = "ml-bot"):
    global _configured
    if _configured:
        return logger
    cfg = load_config().get("logging", {})
    log_dir = resolve_path(cfg.get("dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=cfg.get("level", "INFO"))
    logger.add(
        log_dir / f"{name}.log",
        level=cfg.get("level", "INFO"),
        rotation=cfg.get("rotation", "100 MB"),
        retention=cfg.get("retention", "30 days"),
        enqueue=True,
    )
    _configured = True
    return logger
