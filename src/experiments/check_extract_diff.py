#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抽取规则差分核查 —— 改完 `looks_like_title` 后必跑。

**为什么需要它**：2026-09-23 给 `looks_like_title` 加排除规则时，只验了"新增 0 条"，
就宣称"只收紧不放宽"。结果"句号收尾"那条把
`CommunityKG-RAG: A Retrieval-Augmented Generation Framework...`（省略号截断的**合法标题**）
一起删了 —— 假阴性，直到复核"删掉的是不是都该删"才暴露。
**只看新增、不看删除，等于没验。**

做法：用**旧门槛**（仅"含中文 / markdown / arXiv ID / 英文词 < 4"四条）
和**当前规则**各抽一遍全部答案，列出两个方向的差异：
  - 新增（当前抽到、旧规则没抽到）—— 正常应为 0；非 0 说明规则被放宽了，要查。
  - 删除（旧规则抽到、当前没抽到）—— 逐条人工确认是否都是正文噪声。

用法:
    python src/experiments/check_extract_diff.py
    python src/experiments/check_extract_diff.py --answers experiments/baseline_llm/answers.jsonl
退出码：有新增 → 1（提示放宽了），否则 0。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.textutil import (content_tokens, find_arxiv_ids, has_cjk,  # noqa: E402
                             norm_space, strip_latex)
from detection.extract import BAD_TOKENS, QUOTED, looks_like_title  # noqa: E402

DEFAULT_ANSWERS = [
    ROOT / "experiments" / "baseline_llm" / "answers.jsonl",
    ROOT / "experiments" / "baseline_llm_strict" / "answers.jsonl",
]


def extract_old(text: str) -> list[str]:
    """复现**加排除规则之前**的抽取门槛（只保留四条基础过滤）。"""
    out = []
    for m in QUOTED.finditer(text or ""):
        raw = norm_space(strip_latex(m.group(1)))
        if not raw or has_cjk(raw):
            continue
        if any(t in raw for t in BAD_TOKENS):
            continue
        if find_arxiv_ids(raw):
            continue
        if len(re.findall(r"[A-Za-z][A-Za-z'\-]+", raw)) < 4:
            continue
        out.append(raw)
    return out


def extract_new(text: str) -> list[str]:
    """当前规则（用被测模块自己的 looks_like_title，避免抄错）。"""
    out = []
    for m in QUOTED.finditer(text or ""):
        raw = norm_space(strip_latex(m.group(1)))
        if looks_like_title(raw):
            out.append(raw)
    return out


def why_excluded(raw: str) -> str:
    """当前规则为什么排除了它（便于人工判是噪声还是真标题）。"""
    reasons = []
    if re.search(r"(?<!\.)\.$", raw.rstrip()):
        reasons.append("单个句号收尾")
    if re.match(r"^(review|using|rewrite|please|note|remember|ensure|make sure|"
                r"answer the|based on|read the)\b", raw, re.I):
        reasons.append("祈使句开头")
    if raw == raw.lower() and ":" not in raw and len(content_tokens(raw)) < 6:
        reasons.append("全小写/无冒号/实词<6")
    return "、".join(reasons) or "（未命中新增规则，可能是基础门槛）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="append", default=None,
                    help="答案文件，可多次指定；默认跑两套口径")
    args = ap.parse_args()

    files = [Path(p) for p in args.answers] if args.answers else DEFAULT_ANSWERS
    added: list[tuple[str, str]] = []
    removed: list[tuple[str, str, str]] = []
    n = 0

    for f in files:
        if not f.exists():
            print(f"[跳过] 找不到 {f}")
            continue
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("qid") or "?"
            text = r.get("answer") or ""
            o, nw = set(extract_old(text)), set(extract_new(text))
            for x in sorted(nw - o):
                added.append((qid, x))
            for x in sorted(o - nw):
                removed.append((qid, x, why_excluded(x)))
            n += 1

    print(f"比对 {n} 条回答（{'、'.join(f.name for f in files)}）")
    print()
    print(f"【新增】当前抽到、旧规则没抽到：{len(added)} 条"
          f"{'  ← ⚠️ 规则被放宽了，要查' if added else '  ✅'}")
    for qid, x in added:
        print(f"    {qid}  {x[:100]}")
    print()
    print(f"【删除】旧规则抽到、当前没抽到：{len(removed)} 条  ← 逐条确认是否都是正文噪声")
    for qid, x, why in removed:
        print(f"    {qid}  [{why}]  {x[:100]}")

    return 1 if added else 0


if __name__ == "__main__":
    raise SystemExit(main())
