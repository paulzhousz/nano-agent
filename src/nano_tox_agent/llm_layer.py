"""可选的 LLM 改写层。

LLM 只负责把模板解释改写得更适合阅读，不参与毒性预测本身；
同时会校验核心结论行，避免改写时篡改等级、活力或置信度。
"""

from __future__ import annotations

import json
import logging
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.error import URLError
import urllib.request

SYSTEM_PROMPT = "请用清晰中文改写模型拥有的毒性预测解释。不要改变毒性等级、细胞活力、置信度或引用依据。"
PROTECTED_PREFIXES = ("预测毒性等级：", "预测细胞活力：", "模型置信度：")
ROOT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
LOGGER = logging.getLogger(__name__)


def _normalize_protected_line(line: str) -> str:
    """把 Markdown 风格的列表/加粗清掉，便于比较关键结论行。"""
    normalized = line.strip()
    for prefix in ("- ", "* "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    normalized = normalized.replace("**", "")
    normalized = normalized.replace("__", "")
    return normalized.strip()


def _extract_protected_lines(text: str) -> dict[str, str]:
    """提取必须保持不变的核心结论行。"""
    protected_lines: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = _normalize_protected_line(raw_line)
        for prefix in PROTECTED_PREFIXES:
            if line.startswith(prefix):
                protected_lines[prefix] = line
    return protected_lines


def _preserves_core_conclusions(original: str, rewritten: str) -> bool:
    """确认 LLM 改写没有改动关键数值与标签。"""
    original_lines = _extract_protected_lines(original)
    rewritten_lines = _extract_protected_lines(rewritten)
    for prefix in PROTECTED_PREFIXES:
        original_line = original_lines.get(prefix)
        if original_line is None:
            continue
        if rewritten_lines.get(prefix) != original_line:
            return False
    return True


def load_env_file(path: Path | None = None) -> None:
    """从项目根目录 .env 补充可选的 LLM 配置。"""
    if os.getenv("LLM_DISABLE_DOTENV") == "1":
        LOGGER.info("LLM .env auto-load skipped because LLM_DISABLE_DOTENV=1")
        return
    path = path or ROOT_ENV_PATH
    if not path.exists():
        LOGGER.info("LLM .env file not found at %s", path)
        return

    loaded_keys: list[str] = []
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
        loaded_keys.append(key)
    if loaded_keys:
        LOGGER.info("LLM .env auto-loaded keys: %s", ", ".join(loaded_keys))


def generate_llm_response(prompt: str) -> str | None:
    """调用兼容 OpenAI Chat Completions 的接口改写解释文本。"""
    load_env_file()
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE")
    if not api_key or not api_base or not api_key.strip() or not api_base.strip():
        LOGGER.info("LLM rewrite skipped because API config is missing or blank")
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
            "User-Agent": "curl/8.7.1",
            "Accept": "*/*",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data: Any = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, UnicodeDecodeError, JSONDecodeError) as error:
        LOGGER.warning("LLM rewrite request failed: %s", error)
        return None

    if not isinstance(data, dict):
        LOGGER.warning("LLM rewrite response ignored because top-level JSON is not an object")
        return None

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        LOGGER.warning("LLM rewrite response ignored because choices is missing or empty")
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        LOGGER.warning("LLM rewrite response ignored because first choice is not an object")
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict):
        LOGGER.warning("LLM rewrite response ignored because message is not an object")
        return None

    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()
        # 只有在核心结论完全保留时，才接受改写结果覆盖模板解释。
        if content and _preserves_core_conclusions(prompt, content):
            LOGGER.info("LLM rewrite accepted")
            return content
        LOGGER.warning("LLM rewrite rejected because protected conclusion lines changed format or values")
        return None
    LOGGER.warning("LLM rewrite response ignored because content is missing or not a string")
    return None
