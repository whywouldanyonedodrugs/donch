# /root/apps/donch/tools/_bootstrap.py
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_root() -> Path:
    """
    Ensure repo root is on sys.path so `import live` works even when running:
      python tools/foo.py
    (Because sys.path[0] becomes tools/ otherwise.)

    Safe to call multiple times.
    """
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # .../donch
    repo_root_str = str(repo_root)

    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    return repo_root

