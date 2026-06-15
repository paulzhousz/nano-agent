import json
from pathlib import Path
from typing import Any


def load_literature(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("literature library must be a list")
    return data


def select_literature(entries: list[dict[str, Any]], purpose: str) -> list[dict[str, Any]]:
    return [entry for entry in entries if purpose in entry.get("used_for", [])]
