from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for name in [
        "OPENAI_API_KEY",
        "API_KEY",
        "OPENAI_BASE_URL",
        "BASE_URL",
        "LLM_MODEL",
        "MODEL",
        "LLM_MODE",
    ]:
        match = re.search(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=\s*([\"']?)(.*?)\1\s*$", text, re.M)
        if match and not os.getenv(name):
            os.environ[name] = match.group(2).strip()


def normalize_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        return "https://api.openai.com/v1"
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for OpenAI-compatible chat API.")
    parser.add_argument("--env-file", type=Path, default=ROOT / "doc" / "环境变量.txt")
    parser.add_argument("--message", default="简单介绍复旦大学")
    args = parser.parse_args()

    load_env_file(args.env_file)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    base_url = normalize_base_url(os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "")
    model = os.getenv("LLM_MODEL") or os.getenv("LLM_MODE") or os.getenv("MODEL") or "gpt-4o-mini"

    print(f"api_key_set: {bool(api_key)}")
    print(f"api_key_length: {len(api_key) if api_key else 0}")
    print(f"base_url: {base_url}")
    print(f"model: {model}")

    if not api_key:
        print("RESULT: failed - missing OPENAI_API_KEY or API_KEY")
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个简洁准确的中文助手。"},
                {"role": "user", "content": args.message},
            ],
            temperature=0.2,
        )
    except Exception as exc:
        print(f"RESULT: failed - {type(exc).__name__}: {exc}")
        return 3

    content = response.choices[0].message.content or ""
    print("RESULT: success")
    print("response:")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

