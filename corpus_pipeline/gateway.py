"""Minimal async client for an OpenAI-compatible chat endpoint.
Set LLM_BASE_URL (e.g. https://your-gateway/v1) and LLM_API_KEY in the environment or in ~/.env."""
import asyncio, json, os
from pathlib import Path
import httpx

MODEL = os.environ.get("LLM_MODEL", "claude/claude-haiku-5-5")


def _env(name):
    if name not in os.environ:
        for line in (Path.home() / ".env").read_text().splitlines():
            if line.startswith(name + "="):
                os.environ[name] = line.split("=", 1)[1].strip()
    return os.environ[name]


def api_key():
    return _env("LLM_API_KEY")


async def chat(client, messages, max_tokens=16000, retries=4, reasoning_effort=None, model=None):
    """Returns (text, usage). Streams the answer (long non-streamed requests get cut by the gateway).
    Retries on network errors, 429/5xx and empty answers."""
    body = {"model": model or MODEL, "max_tokens": max_tokens, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    for attempt in range(retries):
        try:
            parts, usage = [], {}
            async with client.stream("POST", f"{_env('LLM_BASE_URL')}/chat/completions", json=body, timeout=600,
                                     headers={"Authorization": f"Bearer {api_key()}"}) as r:
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    if d.get("usage"):
                        usage = d["usage"]
                    for ch in d.get("choices") or []:
                        parts.append((ch.get("delta") or {}).get("content") or "")
            text = "".join(parts).strip()
            if text:
                return text, usage
        except (httpx.HTTPError, KeyError, ValueError) as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}: {type(e).__name__} {str(e)[:120]}", flush=True)
        await asyncio.sleep(5 * 2 ** attempt)
    raise RuntimeError("empty answer after retries")
