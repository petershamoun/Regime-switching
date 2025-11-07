from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "sector_rotation.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    cfg_path = Path(path) if path else default_config_path()
    cfg_path = cfg_path.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
