from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent
def load_config(path=None):
    path=Path(path) if path else ROOT/'config.yaml'
    with path.open() as stream: cfg=yaml.safe_load(stream)
    cfg['_root']=str(path.parent.resolve()); return cfg

