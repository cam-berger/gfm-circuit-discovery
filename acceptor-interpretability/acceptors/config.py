from pathlib import Path
from types import SimpleNamespace

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(model_yaml="model.yaml", experiment_yaml="experiment.yaml"):
    """Load and merge configs/model.yaml + configs/experiment.yaml into one namespace."""
    with open(CONFIG_DIR / model_yaml) as f:
        model_cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / experiment_yaml) as f:
        exp_cfg = yaml.safe_load(f)
    return _to_namespace({"model": model_cfg, **exp_cfg})
