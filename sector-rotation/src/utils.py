from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config
