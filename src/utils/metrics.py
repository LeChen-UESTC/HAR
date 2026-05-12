from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .distributed import is_main_process
from .config_utils import to_builtin


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    if not is_main_process():
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_builtin(payload), sort_keys=True) + "\n")
