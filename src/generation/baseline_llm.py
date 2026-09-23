#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""系统 baseline · 纯 LLM（无检索）—— 第 4.3 对比实验的第一列。

把 400 道题直接丢给 LLM，不给任何论文材料，看它凭自己的记忆怎么答：
  * 正常题上，它会不会编出根本不存在的引用？
  * 陷阱题上，问它一篇不存在的论文，它会不会顺着编？

prompt 与 RAG 版本共用（src/generation/prompts.py），只有"有没有注入检索材料"这一处差别，
这是对比实验公平性的前提。

输出 experiments/baseline_llm/：
    answers.jsonl   每题一条：回答原文 + 耗时 + 错误信息
    config.json     本次运行参数与 prompt 原文（论文附录要用，也保证可复现）

用法:
    python src/generation/baseline_llm.py --limit 5        # 试跑 5 条
    python src/generation/baseline_llm.py --workers 6      # 全量 400 条
    python src/generation/baseline_llm.py --retry-errors   # 只重跑上次失败的
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.llm_client import LLMClient  # noqa: E402
from generation.prompts import build_user, system_prompt  # noqa: E402

QUESTIONS = ROOT / "eval" / "v1" / "questions.jsonl"
RUNS = ROOT / "experiments"
OUT_DIR = RUNS / "baseline_llm"          # 口径 A：中性 prompt（主线对比实验）
OUT_DIR_STRICT = RUNS / "baseline_llm_strict"  # 口径 B：追加诚实性约束（消融用）
OUT = OUT_DIR / "answers.jsonl"
CONFIG = OUT_DIR / "config.json"
PROGRESS = OUT_DIR / ".progress"

NAME = "baseline_llm"
_lock = threading.Lock()


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or QUESTIONS
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_done(path: Path) -> dict[str, dict]:
    """已答过的题（按 qid 索引），用于断点续跑。"""
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("error"):
                done[r["qid"]] = r
    return done


def answer_one(client: LLMClient, rec: dict, temperature: float, system: str) -> dict:
    started = time.time()
    out = {
        "qid": rec["qid"],
        "type": rec["type"],
        "question_form": rec["question_form"],
        "difficulty": rec["difficulty"],
        "question": rec["question"],
        "system": NAME,
        "model": client.model,
        "temperature": temperature,
        "contexts": 0,
        "answer": "",
        "error": None,
        "elapsed": 0.0,
        "created": date.today().isoformat(),
    }
    try:
        out["answer"] = client.chat(system, build_user(rec["question"], contexts=None),
                                    temperature=temperature).strip()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["elapsed"] = round(time.time() - started, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="纯 LLM baseline 答题器")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    ap.add_argument("--workers", type=int, default=6, help="并发数")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--strict", action="store_true",
                    help="口径 B：追加诚实性约束（默认口径 A，中性 prompt）")
    ap.add_argument("--retry-errors", action="store_true", help="只重跑上次出错的题")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variant = "strict" if args.strict else "neutral"
    out_path = Path(args.out) if args.out else \
        ((OUT_DIR_STRICT if args.strict else OUT_DIR) / "answers.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    system = system_prompt(strict=args.strict)

    records = load_questions()
    if args.retry_errors and out_path.exists():
        bad = {json.loads(l)["qid"] for l in out_path.open(encoding="utf-8")
               if l.strip() and json.loads(l).get("error")}
        records = [r for r in records if r["qid"] in bad]
        print(f"只重跑上次失败的 {len(records)} 条")
    else:
        done = load_done(out_path)
        records = [r for r in records if r["qid"] not in done]
        if done:
            print(f"断点续跑：已完成 {len(done)} 条，本次待跑 {len(records)} 条")

    if args.limit:
        records = records[:args.limit]
    if not records:
        print("没有需要跑的题目 ✅")
        return 0

    client = LLMClient(temperature=args.temperature)
    (out_path.parent / "config.json").write_text(json.dumps({
        "system": NAME,
        "prompt_variant": variant,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": client.model,
        "base_url": client.base_url,
        "temperature": args.temperature,
        "workers": args.workers,
        "contexts": 0,
        "question_count": len(records),
        "questions_file": str(QUESTIONS.relative_to(ROOT)),
        "system_prompt": system,
        "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest()[:16],
        "note": "纯 LLM 无检索基线。user prompt 由 src/generation/prompts.build_user(question, contexts=None) 生成。"
                "prompt_variant=neutral 为口径 A（不叮嘱不要编造），strict 为口径 B（追加诚实性约束）。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"开始跑 {len(records)} 题 | 口径 {variant} | 并发 {args.workers} | 模型 {client.model}")
    ok = err = 0
    with out_path.open("a", encoding="utf-8", newline="\n") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(answer_one, client, r, args.temperature, system): r
                   for r in records}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with _lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if rec["error"]:
                err += 1
                print(f"  [{i}/{len(records)}] {rec['qid']} 失败: {rec['error'][:80]}")
            else:
                ok += 1
                if i % 20 == 0 or i == len(records):
                    print(f"  进度 {i}/{len(records)}（成功 {ok}，失败 {err}）")

    print(f"\n完成：成功 {ok}，失败 {err}")
    print(f"输出: {out_path}")
    print(client.stats())
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
