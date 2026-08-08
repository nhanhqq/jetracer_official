from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg, value):
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_root"]) / path
