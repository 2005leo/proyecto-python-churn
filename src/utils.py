# src/utils.py
from pathlib import Path
import json

def save_metrics(d: dict, path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
