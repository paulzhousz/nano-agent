from __future__ import annotations

import json
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.error import URLError
import urllib.request

SYSTEM_PROMPT = "请用清晰中文改写模型拥有的毒性预测解释。不要改变毒性等级、细胞活力、置信度或引用依据。"
PROTECTED_PREFIXES = ("预测毒性等级：", "预测细胞活力：", "模型置信度：")
ROOT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _extract_protected_lines(text: str) -> dict[str, str]:
    protected_lines: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        for prefix in PROTECTED_PREFIXES:
            if line.startswith(prefix):
                protected_lines[prefix] = line
    return protected_lines


def _preserves_core_conclusions(original: str, rewritten: str) -> bool:
    original_lines = _extract_protected_lines(original)
    rewritten_lines = _extract_protected_lines(rewritten)
    for prefix in PROTECTED_PREFIXES:
        original_line = original_lines.get(prefix)
        if original_line is None:
            continue
        if rewritten_lines.get(prefix) != original_line:
            return False
    return True


def load_env_file(path: Path = ROOT_ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key or key in os.environ or not value:
            continue
        os.environ[key] = value


def generate_llm_response(prompt: str) -> str | None:
    load_env_file()
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE")
    if not api_key or not api_base or not api_key.strip() or not api_base.strip():
        return None

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data: Any = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, UnicodeDecodeError, JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content and _preserves_core_conclusions(prompt, content):
            return content
    return None
