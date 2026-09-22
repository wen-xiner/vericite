#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人工抽检抽样 —— 从 v1 评测集按类别分层抽样，导出可实际核对的检查单与回填表。

为什么需要：评测集是自动构造的，论文里必须交代「抽检了多少、通过多少、怎么判的」。
本脚本按比例分层抽样（默认约 10%，40 条），并**带上论文摘要原文**——
没有原文，人根本没法核对「依据是否真支撑答案」「追问的信息是否真不在摘要里」。

输入: eval/v1/questions.jsonl + data/arxiv_raw.jsonl（取摘要做核对上下文）
输出: eval/v1/review_sample.txt   —— 阅读用检查单（摘要中依据片段已用【】标出）
      eval/v1/review_sheet.csv   —— 回填用判定表（Excel 可直接打开填 判定/备注）
      eval/v1/review_sample.jsonl —— 同批样本的原始记录（便于程序化回填）

判定口径（三选一，填在 review_sheet.csv 的「判定」列）：
    通过 = 该题可直接用于正式实验
    修改 = 题干/答案/依据需小改后可用（请在备注写清改什么）
    废弃 = 不能用于实验（如依据不足、表述歧义、陷阱不成立）

为什么判定必须人工做：这是论文中"人工抽检"数据的唯一来源。
AI 自评不能替代，否则该数据即为伪造。

用法:
    python eval/build/sample_for_review.py               # 默认抽 40 条
    python eval/build/sample_for_review.py --normal 30 --t1 8 --t2 6 --t3 6
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "eval" / "v1"
QUESTIONS = V1 / "questions.jsonl"
CORPUS = ROOT / "data" / "arxiv_raw.jsonl"
OUT_TXT = V1 / "review_sample.txt"
OUT_CSV = V1 / "review_sheet.csv"
OUT_JSONL = V1 / "review_sample.jsonl"

SEED = 20260922

# 抽检时最需要盯的点，直接印在单子上，避免复核时漏看
CHECK_HINTS = {
    "normal": "参考答案是否真被依据支撑、依据是否与摘要一致、问题是否唯一解",
    "trap_nonexistent": "编造标题是否够像真的（不会一眼识破）、参考答案是否明确否认其存在",
    "trap_unsupported": "追问的信息是否确实不在摘要中、问题是否看起来合理、参考答案是否拒绝外推",
    "trap_mismatch": "被搬走的成果是否确实出自来源论文、被冒用论文是否确实没做该工作、问题是否自然",
}


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# 中文（含全角标点）逐字断行，英文/数字按整词断行，避免把单词切成两半
CJK = r"\u3000-\u9fff\uff00-\uffef"
TOKEN_RE = re.compile(rf"[{CJK}]|[^\s{CJK}]+")


def find_span(text: str, evidence: str | None) -> tuple[int, int] | None:
    """在摘要中定位依据片段的字符区间（容忍空白差异）。"""
    if not evidence:
        return None
    tokens = [re.escape(t) for t in norm_space(evidence).split(" ") if t]
    if not tokens:
        return None
    m = re.search(r"\s+".join(tokens), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def wrap(text: str, width: int = 74, indent: str = "  ",
         hl_span: tuple[int, int] | None = None) -> list[str]:
    """折行输出。hl_span 覆盖到的行前面加 ▸ 标记（整行标记，不会把标记切断）。"""
    out: list[str] = []
    for para in (text or "").split("\n"):
        if not para:
            out.append("")
            continue
        line = ""
        last_cjk = False
        start: int | None = None
        end: int | None = None

        def flush() -> None:
            nonlocal line, start, end
            if not line:
                return
            hit = (hl_span and start is not None and end is not None
                   and start < hl_span[1] and end > hl_span[0])
            out.append(indent + ("▸ " if hit else "  ") + line)
            line, start, end = "", None, None

        for m in TOKEN_RE.finditer(para):
            tok = m.group(0)
            # 注意：不能只看 len(tok)==1，单个英文字母长度也是 1，会误判成中文
            is_cjk = bool(re.fullmatch(rf"[{CJK}]", tok))
            sep = "" if (not line or is_cjk or last_cjk) else " "
            if line and len(line) + len(sep) + len(tok) > width:
                flush()
                sep = ""
            if not line:
                start = m.start()
            line += sep + tok
            end = m.end()
            last_cjk = is_cjk
            if len(line) >= width:
                flush()
        flush()
    return out or [indent]


def group_head(title: str) -> list[str]:
    return ["", "=" * 78, title, "=" * 78]


def main() -> int:
    ap = argparse.ArgumentParser(description="评测集人工抽检抽样")
    ap.add_argument("--normal", type=int, default=25)
    ap.add_argument("--t1", type=int, default=6)
    ap.add_argument("--t2", type=int, default=5)
    ap.add_argument("--t3", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(SEED)
    records = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    corpus = {p["arxiv_id"]: p for p in
              (json.loads(l) for l in CORPUS.open(encoding="utf-8"))}

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
        "VeriCite 评测集 v1 —— 人工抽检单（阅读用）",
        f"生成日期: {date.today().isoformat()} | 随机种子: {SEED}",
        f"抽样 {len(sample)} 条 / 全集 {len(records)} 条 （{len(sample) / max(len(records), 1):.0%}）",
        "=" * 78,
        "判定四步（正常题照这个走；陷阱题的判要点见每条末尾）：",
        "  ① 这题要的是什么信息？（数量 / 方法名 / 结论 / 对比关系，先定下「要什么」）",
        "  ② 摘要里能不能找到对应的原句？（找不到支撑句 = 无据，判废弃）",
        "  ③ 依据片段是否与摘要逐字一致？（机器已比过，扫一眼确认没标错行）",
        "  ④ 摘要里有没有第二个可能被误认的答案？（有则判修改 —— 给题干补限定词）",
        "  四项全过 → 通过",
        "",
        "不通过的典型情形：",
        "  摘要找不到支撑答案的句子（答案像是模型自己推的）        → 废弃",
        "  依据片段与摘要对不上（机器标错了行 / 断章取义）          → 废弃（或改 evidence）",
        "  答案比摘要多说了内容（摘要只给方向，答案却给了具体数值）  → 废弃",
        "  题干可能对应多个答案（缺限定词，如未写「出版物」）        → 修改（补限定词）",
        "  问题脱离上下文看不懂（出现「该论文」「所提方法」）        → 修改",
        "  答案方向对但表述含糊、没法判对错（如「效果更好」）        → 修改",
        "",
        "怎么用：",
        "  1. 逐条读题目、参考答案、依据片段，摘要原文里依据所在行前有 ▸ 标记。",
        "  2. 把判定填进 review_sheet.csv 的「判定」列：通过 / 修改 / 废弃（备注写清原因）。",
        "  3. 填完交回，按判定回填 questions.jsonl 的 status（reviewed / rejected）并算通过率。",
        "",
    ]

    for i, r in enumerate(sample, 1):
        qid, rtype = r["qid"], r["type"]
        lines += [
            "-" * 78,
            f"[{i}/{len(sample)}] {qid}  |  {rtype}  |  {r['question_form']}  |  难度 {r['difficulty']}  |  领域 {r['domain']}",
            "",
            "【问题】",
        ]
        lines += wrap(norm_space(r["question"]))
        lines += ["", "【参考答案】"]
        lines += wrap(norm_space(r["reference_answer"]))

        trap = r.get("trap") or {}
        if rtype == "normal":
            aid = (r["gold_arxiv_ids"] or [None])[0]
            paper = corpus.get(aid, {})
            lines += ["", f"【依据出处】arXiv:{aid} 《{paper.get('title', '')}》", "", "【依据片段】"]
            lines += wrap(norm_space(r["evidence"]))
            lines += ["", "【论文摘要原文（▸ 标出依据所在行）】"]
            ab = norm_space(paper.get("abstract", ""))
            lines += wrap(ab, hl_span=find_span(ab, r.get("evidence")))
        elif rtype == "trap_nonexistent":
            lines += [
                "",
                f"【编造标题】{trap.get('fabricated_title')}",
                f"【编造 ID】{trap.get('fabricated_id')}（已由脚本向 arXiv 复核：不存在）",
                "",
                "【摘要原文】无 —— 该文献不存在，本题考察系统会不会硬编内容。",
                "",
                "【判要点】这个标题读起来像不像一篇真实的 arXiv 论文？如果一眼就假，题目太弱，建议废弃。",
            ]
        elif rtype == "trap_unsupported":
            aid = trap.get("target_paper")
            paper = corpus.get(aid, {})
            lines += [
                "",
                f"【被追问的真实论文】arXiv:{aid} 《{paper.get('title', '')}》",
                "",
                "【论文摘要原文】",
            ]
            lines += wrap(paper.get("abstract", ""))
            lines += [
                "",
                f"【已验证不在摘要中的关键概念】{json.dumps(r.get('verification', {}).get('absent_keywords_confirmed_not_in_abstract', []), ensure_ascii=False)}",
                "",
                "【判要点】读完整段摘要，确认问题要问的那件事确实没写。",
            ]
        else:
            src = corpus.get(trap.get("source_paper"), {})
            tgt = corpus.get(trap.get("target_paper"), {})
            lines += [
                "",
                f"【成果真正来源】arXiv:{trap.get('source_paper')} 《{src.get('title', '')}》",
                "【来源论文摘要（▸ 标出被搬走的片段所在行）】",
            ]
            sab = norm_space(src.get("abstract", ""))
            lines += wrap(sab, hl_span=find_span(sab, r.get("evidence")))
            lines += [
                "",
                f"【被冒用论文】arXiv:{trap.get('target_paper')} 《{tgt.get('title', '')}》",
                "【被冒用论文摘要】",
            ]
            lines += wrap(tgt.get("abstract", ""))
            lines += ["", "【判要点】确认被搬走的成果确实属于来源论文、且被冒用论文摘要里没有该工作。"]

        lines += [
            "",
            f"【期望行为】{norm_space(r['expected_behavior'])}",
            f"【机器校验】{json.dumps(r.get('verification', {}), ensure_ascii=False)}",
        ]
        if r.get("notes"):
            lines.append(f"【待核原因】{r['notes']}")
        lines += [f"【重点核对】{CHECK_HINTS.get(rtype, '')}", "", f"【判定】通过 / 修改 / 废弃：____________", ""]

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    # 回填用判定表：Excel 打开填「判定」「备注」两列，其余列只读参考
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "类型", "问法", "难度", "问题", "参考答案",
                    "依据片段", "应引用来源", "待核原因", "判定", "备注"])
        for r in sample:
            w.writerow([
                r["qid"], r["type"], r["question_form"], r["difficulty"],
                norm_space(r["question"]), norm_space(r["reference_answer"]),
                norm_space(r.get("evidence") or "(无)"),
                ";".join(r["gold_arxiv_ids"]) or "(无)",
                r.get("notes") or "",
                "", "",
            ])

    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"抽样完成：{len(sample)} 条 / 全集 {len(records)} 条")
    for t, q in quotas.items():
        print(f"  {t}: 抽 {min(q, len(by_type.get(t, [])))} / 全集 {len(by_type.get(t, []))}")
    print(f"\n阅读用检查单（带摘要）: {OUT_TXT}")
    print(f"回填用判定表（Excel）: {OUT_CSV}")
    print(f"同批样本（程序回填用）: {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
