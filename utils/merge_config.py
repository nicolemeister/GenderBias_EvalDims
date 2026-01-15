# merge_config.py
from pathlib import Path
import sys, copy
import yaml   # pip install pyyaml
import tomllib  # py3.11+

def deepmerge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items(): out[k] = deepmerge(a.get(k), v)
        return out
    return copy.deepcopy(b)

def load_any(p):
    # Allow p to be either a Path or a string
    if isinstance(p, str):
        p = Path(p)
    if p.suffix in (".yml", ".yaml"):
        return yaml.safe_load(p.read_text())
    if p.suffix == ".toml":
        return tomllib.loads(p.read_text())
    raise ValueError(p)

def merge_config(base_path: Path, overlay_path: Path):
    base = load_any(base_path)
    overlay = load_any(overlay_path)
    return deepmerge(base, overlay)