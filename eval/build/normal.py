#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正常题生成 —— 基于真实 arXiv 摘要，用 LLM 产出可溯源的问答对。

输入: data/arxiv_raw.jsonl（fetch_arxiv.py 产出）
输出: eval/v1/normal_candidates.jsonl（候选池，多于最终 250 条，便于筛掉不合格的）

关键约束（保证毕设可信度）:
  * 出题只能依据摘要原文，禁止引入摘要外的知识。
  * 每题必须给出 evidence —— 摘要中逐字出现的连续片段，脚本会做子串校验，
    不通过的打上 needs_review，绝不静默放进评测集。
  * 支持断点续跑：已处理过的 arxiv_id 记在 eval/v1/.normal_progress，重跑自动跳过。

用法:
    python eval/build/normal.py                # 全量（143 篇，每篇 2 题）
    python eval/build/normal.py --per-paper 2 --workers 5
    python eval/build/normal.py --limit 10     # 先试跑 10 篇
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))

from llm_client import LLMClient, load_env  # noqa: E402

DATA = ROOT / "data" / "arxiv_raw.jsonl"
OUT_DIR = ROOT / "eval" / "v1"
OUT = OUT_DIR / "normal_candidates.jsonl"
PROGRESS = OUT_DIR / ".normal_progress"

SYSTEM = """你是学术文献问答评测集的出题专家，正在为「学术文献研读问答系统」构建评测集。

你的任务：只依据我给出的论文摘要出题，题目和答案必须能被摘要原文完全支撑。

铁律（违反即作废）：
1. 严禁使用摘要之外的知识。你如果知道这篇论文的其他细节，也不许写进答案。
2. evidence 字段必须是从摘要中【逐字复制】的一段连续原文，不得改写、不得加省略号、不得拼接不相邻的句子。
3. 答案要具体：落到摘要里出现的方法名、模型名、数据集名、指标数值、结论。
4. 问题用中文提问，且必须能脱离论文上下文单独看懂：要带上论文的核心方法名或研究主题
   （例如「C$^2$-Cite 框架如何……」「面向忠实性评估的指标融合策略中……」），
   禁止出现「这篇论文」「该摘要」「所提出的方法」这类需要靠上下文才知道指什么的说法。
5. 不要问摘要里没有的信息（比如代码是否开源、作者单位、未来工作）。

只输出 JSON，不要任何解释文字。"""

USER_TMPL = """论文信息：
- arXiv ID: {aid}
- 标题: {title}
- 发表年份: {year}
- 学科分类: {cat}
- 摘要: {abstract}

请出 {n} 道问题，覆盖不同的问法（例如：方法细节、关键结论、具体数值、与已有方法的对比、设计动机）。
输出 JSON 格式：
{{
  "questions": [
    {{
      "question": "中文问题",
      "reference_answer": "中文答案，具体、可溯源",
      "evidence": "摘要中逐字复制的一段原文",
      "question_form": "factual | method | comparison | numeric | causal 五选一",
      "difficulty": "easy | medium | hard"
    }}
  ]
}}"""

VALID_FORMS = {"factual", "method", "comparison", "numeric", "causal"}
VALID_DIFF = {"easy", "medium", "hard"}

_lock = threading.Lock()


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def check_evidence(evidence: str, abstract: str) -> bool:
    """evidence 必须是摘要的逐字连续片段。"""
    if not evidence:
        return False
    ev = norm_space(evidence)
    ab = norm_space(abstract)
    if len(ev) < 20:
        return False
    return ev in ab


# 需要靠上下文才能看懂指代的说法：出现则标记待人工复核
VAGUE_PATTERNS = ("这篇论文", "该论文", "本文", "该摘要", "所提出的方法", "上述方法", "该方法")


def check_self_contained(question: str) -> bool:
    """问题应能脱离论文上下文单独看懂（不含含糊指代）。"""
    return not any(p in question for p in VAGUE_PATTERNS)


def gen_for_paper(client: LLMClient, paper: dict, n: int) -> list[dict]:
    user = USER_TMPL.format(
        aid=paper["arxiv_id"],
        title=paper["title"],
        year=paper["published"][:4],
        cat=paper["primary_category"],
        abstract=paper["abstract"],
        n=n,
    )
    recs: list[dict] = []
    try:
        data = client.chat_json(SYSTEM, user)
        items = data.get("questions") or []
    except Exception as exc:  # noqa: BLE001 - 单篇失败不影响整体
        return [{"_error": f"{paper['arxiv_id']}: {type(exc).__name__}: {exc}"}]

    for item in items:
        q = (item.get("question") or "").strip()
        a = (item.get("reference_answer") or "").strip()
        ev = (item.get("evidence") or "").strip()
        if not q or not a:
            continue
        form = (item.get("question_form") or "factual").strip().lower()
        diff = (item.get("difficulty") or "medium").strip().lower()
        if form not in VALID_FORMS:
            form = "factual"
        if diff not in VALID_DIFF:
            diff = "medium"
        ok = check_evidence(ev, paper["abstract"])
        clear = check_self_contained(q)
        issues = []
        if not ok:
            issues.append("evidence 未通过摘要逐字校验")
        if not clear:
            issues.append("问题含含糊指代，需改写成自洽问法")
        recs.append({
            "type": "normal",
            "question": q,
            "reference_answer": a,
            "gold_arxiv_ids": [paper["arxiv_id"]],
            "evidence": ev,
            "question_form": form,
            "difficulty": diff,
            "domain": paper["primary_category"],
            "trap": None,
            "detection_label": {"existence": "pass", "support": "pass", "traceability": "pass"},
            "expected_behavior": "引用真实存在的文献并给出被原文支撑的答案，附正确出处（arXiv ID）。",
            "annotator": "auto",
            "status": "draft" if not issues else "needs_review",
            "created": date.today().isoformat(),
            "notes": None if not issues else "；".join(issues),
            "_source_title": paper["title"],
            "_evidence_verified": ok,
            "_self_contained": clear,
        })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description="生成评测集正常题")
    ap.add_argument("--per-paper", type=int, default=2, help="每篇论文出几题，默认 2")
    ap.add_argument("--workers", type=int, default=5, help="并发数，默认 5")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（试跑用）")
    ap.add_argument("--fresh", action="store_true", help="忽略断点，从头重跑")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    papers = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    if args.limit:
        papers = papers[: args.limit]

    done: set[str] = set()
    if PROGRESS.exists() and not args.fresh:
        done = set(PROGRESS.read_text(encoding="utf-8").split())
    todo = [p for p in papers if p["arxiv_id"] not in done]
    print(f"论文总数 {len(papers)}，已完成 {len(done)}，本轮待处理 {len(todo)}")

    if not todo:
        print("无需处理，直接统计：")
    client_cfg = load_env()
    print(f"模型: {client_cfg['model']} @ {client_cfg['base_url']}")

    errors: list[str] = []
    written = 0

    def work(paper: dict) -> list[dict]:
        cli = LLMClient(client_cfg)
        return gen_for_paper(cli, paper, args.per_paper)

    with OUT.open("a", encoding="utf-8", newline="\n") as fout, \
            PROGRESS.open("a", encoding="utf-8", newline="\n") as fprog, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            paper = futures[fut]
            recs = fut.result()
            if recs and "_error" in recs[0]:
                errors.append(recs[0]["_error"])
                print(f"  [{i}/{len(todo)}] {paper['arxiv_id']} 失败")
                continue
            with _lock:
                for rec in recs:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                fout.flush()
                fprog.write(paper["arxiv_id"] + "\n")
                fprog.flush()
            flag = "OK" if all(r.get("_evidence_verified") for r in recs) else "部分待核"
            print(f"  [{i}/{len(todo)}] {paper['arxiv_id']} → {len(recs)} 题 [{flag}]")

    # 统计
    all_recs = [json.loads(l) for l in OUT.open(encoding="utf-8")] if OUT.exists() else []
    verified = sum(1 for r in all_recs if r.get("_evidence_verified"))
    clear = sum(1 for r in all_recs if r.get("_self_contained"))
    print("\n===== 正常题候选池 =====")
    print(f"本轮新增 {written} 条，候选池累计 {len(all_recs)} 条")
    print(f"evidence 逐字校验通过 {verified}/{len(all_recs)}"
          f"（通过率 {verified / max(len(all_recs), 1):.1%}）")
    print(f"问法自洽检查通过 {clear}/{len(all_recs)}"
          f"（通过率 {clear / max(len(all_recs), 1):.1%}）")
    forms: dict[str, int] = {}
    for r in all_recs:
        forms[r["question_form"]] = forms.get(r["question_form"], 0) + 1
    print(f"问法分布: {dict(sorted(forms.items(), key=lambda x: -x[1]))}")
    if errors:
        print(f"\n失败 {len(errors)} 篇：")
        for e in errors[:5]:
            print("  -", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
