from __future__ import annotations

import argparse
import json
import os
import re
import socket
import ssl
from pathlib import Path
from urllib import error, parse, request

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


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def run_network_diagnostics(base_url: str, api_key: str, model: str, message: str) -> None:
    print("\n--- network diagnostics ---")
    parsed = parse.urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        print("invalid base_url: missing host")
        return

    print(f"host: {host}")
    print(f"port: {port}")

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = sorted({item[4][0] for item in infos})
        print(f"dns: ok -> {addresses}")
    except Exception as exc:
        print(f"dns: failed - {type(exc).__name__}: {exc}")
        return

    try:
        with socket.create_connection((host, port), timeout=10):
            print("tcp_connect: ok")
    except Exception as exc:
        print(f"tcp_connect: failed - {type(exc).__name__}: {exc}")
        return

    if parsed.scheme == "https":
        try:
            context = ssl.create_default_context()
            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host):
                    print("tls_handshake: ok")
        except Exception as exc:
            print(f"tls_handshake: failed - {type(exc).__name__}: {exc}")
            return

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个简洁准确的中文助手。"},
                {"role": "user", "content": message},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    url = chat_completions_url(base_url)
    print(f"raw_http_url: {url}")
    req = request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"raw_http_status: {response.status}")
            print(f"raw_http_content_type: {response.headers.get('Content-Type')}")
            print(f"raw_http_body_preview: {body[:1000].replace(chr(10), chr(92) + 'n')}")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"raw_http_error_status: {exc.code}")
        print(f"raw_http_body_preview: {body[:1000].replace(chr(10), chr(92) + 'n')}")
    except Exception as exc:
        print(f"raw_http_failed: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for OpenAI-compatible chat API.")
    parser.add_argument("--env-file", type=Path, default=ROOT / "doc" / "环境变量.txt")
    parser.add_argument("--message", default="简单介绍复旦大学")
    parser.add_argument("--diagnose", action="store_true", help="Run DNS/TCP/TLS/raw HTTP diagnostics.")
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
        cause = getattr(exc, "__cause__", None)
        context = getattr(exc, "__context__", None)
        if cause:
            print(f"cause: {type(cause).__name__}: {cause}")
        if context and context is not cause:
            print(f"context: {type(context).__name__}: {context}")
        run_network_diagnostics(base_url, api_key, model, args.message)
        return 3

    content = response.choices[0].message.content or ""
    print("RESULT: success")
    print("response:")
    print(content)
    if args.diagnose:
        run_network_diagnostics(base_url, api_key, model, args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
