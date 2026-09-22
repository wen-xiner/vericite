#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人工抽检抽样 —— 从 v1 评测集按类别分层抽样，导出便于逐条核对的文本。

为什么需要：评测集是自动构造的，论文里必须交代「抽检了多少、通过多少」。
本脚本按比例分层抽样（默认约 10%），输出带判定栏的检查单，人工勾选后再回填。

输入: eval/v1/questions.jsonl
输出: eval/v1/review_sample.txt   —— 可直接打印/对着屏幕核对的检查单
      eval/v1/review_sample.jsonl —— 同一批样本（便于程序化回填）

用法:
    python eval/build/sample_for_review.py               # 默认抽 40 条
    python eval/build/sample_for_review.py --normal 30 --t1 8 --t2 6 --t3 6
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "eval" / "v1"
QUESTIONS = V1 / "questions.jsonl"
OUT_TXT = V1 / "review_sample.txt"
OUT_JSONL = V1 / "review_sample.jsonl"

# 抽检时最需要盯的点，直接印在单子上，避免复核时漏看
CHECK_HINTS = {
    "normal": "① 参考答案是否真被 evidence 支撑 ② evidence 是否与摘要原文一致 ③ 问题是否会引出多解",
    "trap_nonexistent": "① 编造的标题是否够像真的（不会让模型一眼识破）② 参考答案是否明确否认其存在 ③ 标题是否意外撞上真实论文",
    "trap_unsupported": "① 追问的信息是否确实不在摘要中 ② 问题是否看起来合理（不会一眼假）③ 参考答案是否拒绝外推",
    "trap_mismatch": "① 被搬走的成果是否确实出自 source 论文 ② target 论文是否确实没做该工作 ③ 问题是否自然（不像刻意栽赃）",
}

SEED = 20260922


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="评测集人工抽检抽样")
    ap.add_argument("--normal", type=int, default=25)
    ap.add_argument("--t1", type=int, default=6)
    ap.add_argument("--t2", type=int, default=5)
    ap.add_argument("--t3", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(SEED)
    records = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]

    quotas = {"normal": args.normal, "trap_nonexistent": args.t1,
              "trap_unsupported": args.t2, "trap_mismatch": args.t3}
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r["type"], []).append(r)

    sample: list[dict] = []
    for t, q in quotas.items():
        pool = by_type.get(t, [])[:]
        rng.shuffle(pool)
        sample += pool[:q]

    lines = [
        "=" * 78,
        "VeriCite 评测集 v1 —— 人工抽检单",
        f"生成日期: {date.today().isoformat()} | 随机种子: {SEED}",
        f"抽样 {len(sample)} 条 / 全集 {len(records)} 条 "
        f"（{len(sample) / max(len(records), 1):.0%}）",
        "=" * 78,
        "核对方式：逐条判断「该题是否可用于正式实验」，在【判定】处填 通过 / 修改 / 废弃。",
        "全部通过率写入论文第 4 章评测集构建小节；不通过的题在 eval/v1/questions.jsonl 中把 status 改为 reviewed / rejected。",
        "",
    ]

    for i, r in enumerate(sample, 1):
        lines += [
            "-" * 78,
            f"[{i}/{len(sample)}] {r['qid']}  |  {r['type']}  |  {r['question_form']}  |  难度 {r['difficulty']}  |  领域 {r['domain']}",
            f"【问题】{norm_space(r['question'])}",
            f"【参考答案】{norm_space(r['reference_answer'])}",
        ]
        if r.get("evidence"):
            lines.append(f"【依据片段】{norm_space(r['evidence'])}")
        else:
            lines.append("【依据片段】（无 —— 陷阱题无真实来源）")
        lines.append(f"【应引用的真实来源】{r['gold_arxiv_ids'] or '无'}")
        if r.get("trap"):
            t = r["trap"]
            lines.append(f"【陷阱设计】{t.get('how', '')}")
            if t.get("target_paper"):
                lines.append(f"【被冒用论文】{t['target_paper']} 《{t.get('target_title')}》")
            if t.get("source_paper"):
                lines.append(f"【真正来源论文】{t['source_paper']} 《{t.get('source_title')}》")
        lines.append(f"【期望行为】{norm_space(r['expected_behavior'])}")
        lines.append(f"【机器校验】{json.dumps(r.get('verification', {}), ensure_ascii=False)}")
        if r.get("notes"):
            lines.append(f"【待核原因】{r['notes']}")
        lines.append(f"【重点核对】{CHECK_HINTS.get(r['type'], '')}")
        lines += ["【判定】通过 / 修改 / 废弃：____________", "【改动说明】____________________________", ""]

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"抽样完成：{len(sample)} 条")
    for t, q in quotas.items():
        print(f"  {t}: 抽 {min(q, len(by_type.get(t, [])))} / 全集 {len(by_type.get(t, []))}")
    print(f"\n检查单: {OUT_TXT}")
    print(f"同批样本: {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
