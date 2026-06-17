"""文献库读取与筛选。"""

import json
from pathlib import Path
from typing import Any


def load_literature(path: Path) -> list[dict[str, Any]]:
    """读取文献 JSON，并确保顶层结构是列表。"""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("literature library must be a list")
    return data


def select_literature(entries: list[dict[str, Any]], purpose: str) -> list[dict[str, Any]]:
    """按用途标签筛选当前页面需要的文献条目。"""
    return [entry for entry in entries if purpose in entry.get("used_for", [])]
