#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人工抽检结果回填 —— 把 review_sheet.csv 里的判定写回评测集，并产出抽检报告。

输入: eval/v1/review_sheet.csv   （人工填好「判定」列：通过 / 修改 / 废弃）
      eval/v1/questions.jsonl    （正式集）
输出: eval/v1/questions.jsonl    （原地更新 status 并补 review 字段）
      eval/v1/review_result.json （抽检报告，可直接用于论文第 4 章）

状态映射（保持 status 枚举语义干净）:
    通过 → status = reviewed                （抽检通过，可计入正式实验）
    修改 → status = draft + needs_edit      （小改后可用，留给 v1.1 修订）
    废弃 → status = rejected                （不可用）

通过率口径：通过率 = 通过条数 / 抽检总条数。「修改」与「废弃」都不计入通过。

必须由人填判定 —— 本脚本只做回填与统计，不生成判定，否则论文中的
「人工抽检通过率」即为伪造。

用法:
    python eval/build/record_review.py
    python eval/build/record_review.py --verdict-source "用户口述：全部通过"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "eval" / "v1"
SHEET = V1 / "review_sheet.csv"
QUESTIONS = V1 / "questions.jsonl"
RESULT = V1 / "review_result.json"

ALLOWED = {"通过", "修改", "废弃"}


def main() -> int:
    ap = argparse.ArgumentParser(description="回填人工抽检结果")
    ap.add_argument("--verdict-source", default="人工逐条判定（review_sheet.csv）",
                    help="判定来源说明，写进报告，便于论文交代数据出处")
    ap.add_argument("--reviewer", default="温惠锦", help="抽检人")
    args = ap.parse_args()

    if not SHEET.exists():
        print(f"找不到 {SHEET}，请先跑 sample_for_review.py", file=sys.stderr)
        return 1

    verdicts: dict[str, dict] = {}
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = (row.get("qid") or "").strip()
            if not qid:
                continue
            verdicts[qid] = {
                "verdict": (row.get("判定") or "").strip(),
                "note": (row.get("备注") or "").strip(),
                "type": (row.get("类型") or "").strip(),
            }

    missing = [q for q, v in verdicts.items() if v["verdict"] not in ALLOWED]
    if missing:
        print(f"以下 {len(missing)} 条还没填判定（或填写不规范，只接受 通过/修改/废弃）：")
        for q in missing[:20]:
            print("  -", q)
        print("\n填好后重跑本脚本。")
        return 1

    counts = Counter(v["verdict"] for v in verdicts.values())
    by_type = Counter(v["type"] for v in verdicts.values())
    type_pass = Counter(v["type"] for v in verdicts.values() if v["verdict"] == "通过")

    # 回填 questions.jsonl
    records = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    touched = 0
    for r in records:
        v = verdicts.get(r["qid"])
        if not v:
            continue
        touched += 1
        verdict = v["verdict"]
        if verdict == "通过":
            r["status"] = "reviewed"
        elif verdict == "废弃":
            r["status"] = "rejected"
        else:  # 修改
            r["status"] = "draft"
        r["review"] = {
            "verdict": verdict,
            "needs_edit": verdict == "修改",
            "note": v["note"] or None,
            "reviewer": args.reviewer,
            "round": 1,
            "date": date.today().isoformat(),
        }

    if touched != len(verdicts):
        print(f"[注意] 抽检表 {len(verdicts)} 条，但正式集里只匹配到 {touched} 条，请检查 qid 是否有变更")

    with QUESTIONS.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_sample = len(verdicts)
    n_pass = counts.get("通过", 0)
    result = {
        "round": 1,
        "date": date.today().isoformat(),
        "reviewer": args.reviewer,
        "verdict_source": args.verdict_source,
        "scope": {
            "sampled": n_sample,
            "full_set": len(records),
            "ratio": round(n_sample / max(len(records), 1), 4),
            "sampling": "按类别分层随机抽样（正常题 25 / T1 6 / T2 5 / T3 4），随机种子 20260922",
            "note": "本次人工抽检仅覆盖抽样到的题目；未抽到的题目只经过程序化校验（逐字/arXiv 核验），"
                    "不得表述为「全部人工复核」。",
        },
        "counts": {k: counts.get(k, 0) for k in ("通过", "修改", "废弃")},
        "pass_rate": round(n_pass / max(n_sample, 1), 4),
        "by_type_sampled": dict(by_type),
        "by_type_passed": dict(type_pass),
        "failed_qids": [q for q, v in verdicts.items() if v["verdict"] != "通过"] or [],
        "questions_updated": touched,
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== 抽检回填完成 =====")
    print(f"抽检 {n_sample} 条 / 全集 {len(records)} 条（{result['scope']['ratio']:.0%}）")
    print(f"判定：通过 {counts.get('通过', 0)}　修改 {counts.get('修改', 0)}　废弃 {counts.get('废弃', 0)}")
    print(f"通过率：{result['pass_rate']:.1%}")
    print(f"分工种通过情况：{dict(type_pass)}")
    if result["failed_qids"]:
        print(f"未通过的题：{result['failed_qids']}")
    print(f"\n已更新: {QUESTIONS}")
    print(f"抽检报告: {RESULT}")
    print("\n提醒：报告中已写明本次仅为分层抽检，未覆盖的题目只是过了程序化校验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
