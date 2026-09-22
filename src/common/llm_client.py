#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DeepSeek（OpenAI 兼容接口）轻量封装 —— VeriCite 离线构建脚本共用。

只用标准库，避免为了写评测集装一堆依赖。
密钥从项目根目录 .env 读取（该文件已被 .gitignore 排除）。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> dict[str, str]:
    """读取 .env（KEY=VALUE，支持 # 注释），不覆盖已存在的环境变量。"""
    path = path or (ROOT / ".env")
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先由 .env.example 复制并填写密钥")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    }


class LLMClient:
    """极简 chat 封装：失败重试 + JSON 模式。"""

    def __init__(self, cfg: dict[str, str] | None = None, temperature: float = 0.3,
                 max_tokens: int = 2048):
        cfg = cfg or load_env()
        if not cfg["api_key"]:
            raise RuntimeError("DEEPSEEK_API_KEY 为空，无法调用 LLM")
        self.key = cfg["api_key"]
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, system: str, user: str, json_mode: bool = False,
             retries: int = 3, temperature: float | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = data.get("usage") or {}
                self.calls += 1
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:200]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                # 429/5xx 退避重试，4xx 其他错误直接放弃
                if exc.code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(4 * attempt)
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(3 * attempt)
        raise RuntimeError(f"LLM 调用失败: {last}")

    def chat_json(self, system: str, user: str, **kw) -> dict:
        """要求模型返回 JSON 对象，容错剥离 ```json 包裹。"""
        text = self.chat(system, user, json_mode=True, **kw).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def stats(self) -> str:
        return (f"调用 {self.calls} 次 | prompt {self.prompt_tokens} tok | "
                f"completion {self.completion_tokens} tok")


if __name__ == "__main__":
    c = LLMClient()
    print("配置:", c.base_url, c.model)
    print("连通性:", c.chat("你是有用的助手", "只回复两个字：可用")[:20])
    print(c.stats())
    sys.exit(0)
