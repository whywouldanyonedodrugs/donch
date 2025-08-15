# research/modeling/feature_mask.py
import os
from pathlib import Path
from typing import Set, Iterable

DEFAULT_PATH = Path("research/config/disable_features.txt")

def load_disabled_features(path: str | None = None) -> Set[str]:
    """
    Reads a newline-delimited list of feature names to disable.
    Lines starting with '#' are ignored. Blank lines ignored.
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return set()
    out: Set[str] = set()
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out

def filter_feature_list(features: Iterable[str], disabled: Set[str]) -> list[str]:
    return [f for f in features if f not in disabled]
