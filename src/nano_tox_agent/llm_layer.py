from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any
from urllib.error import URLError
import urllib.request

SYSTEM_PROMPT = "请用清晰中文改写模型拥有的毒性预测解释。不要改变毒性等级、细胞活力、置信度或引用依据。"


def generate_llm_response(prompt: str) -> str | None:
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE")
    if not api_key or not api_base:
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
    except (OSError, URLError, TimeoutError, JSONDecodeError):
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
        if content:
            return content
    return None
