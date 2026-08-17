from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


ROOT = Path(__file__).resolve().parent


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["_root"] = str(config_path.resolve().parent)
    return cfg


def resolve_path(cfg: Dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_root"]) / path
