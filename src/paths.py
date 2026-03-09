from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project_paths.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

_PATHS = _CFG["paths"]

def project_path(key: str) -> Path:
    return (PROJECT_ROOT / _PATHS[key]).resolve()