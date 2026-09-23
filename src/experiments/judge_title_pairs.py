#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标题对语义判官 —— 给 `exists_variant` 加一份**可复现**的佐证。

## 背景：词面路线已经试尽
`exists_variant` 是"精确检索 0 命中、但放宽检索命中高度相似标题"这一态，属**软证据**。
判它是"同一篇、只是写法有偏差"还是"同主题的另一篇"，词面指标全部不可分（2026-09-23 实测）：

| 试过的判据 | 结论 |
|---|---|
| 词面 F1 阈值（含提高到 0.8） | 真阳性可低到 0.625，假阳性可高到 0.769 —— 区间重叠 |
| 5 词 AND 检索 | 有一个词写错就 0 命中（`CommunityKG-RAG: … Framework...` 被误判） |
| 覆盖率（模型词被候选覆盖比例） | 假 `QA-RAG` 0.78 vs 真 `Mitigating…via Dynamic Trust Calibration` 0.80 |
| 独特词（DF）对齐 | 会把**没有独特词的真标题**全杀（`Hallucination in Natural Language Generation: A Survey`）|
| 缩写展开否决 | 能挡 `FenCE`/`QA-RAG`，但**误杀 `ALCE:`**（ALCE 是真实系统名，只是不在标题里）|
| 最长公共连续词片段 | `FenCE` 与候选仍有 6 词连续重合 |

所以改用语义判官：把「模型标题 + 候选标题」成对给 LLM，问是否指同一篇。

## 本脚本**不改任何判定**
按 2026-09-23 的决定（方案"甲"），判官结果只作**佐证**：
产出 `experiments/cache/title_pair_judge.json`，并与人工逐条复核的标签比对一致率。
**不修改 `judged.jsonl` / `metrics.json`，不重跑第一层** —— 所以主指标不受影响。

## 判据纪律
- 只给判官**两条标题**，不给 arXiv ID、不给问题、不给摘要 —— 避免任何泄漏。
- 判官是评估者、不是真值。n=18 条已全部经人工逐条复核，本脚本报的是**一致率**，
  两者不一致的条目要人工再看。
- 提示词是中性的一般性说明（允许语序/缩写/漏词等写法偏差，但主题相近≠同一篇），
  **不点名任何具体案例**，避免把答案喂给判官。

用法:
    python src/experiments/judge_title_pairs.py            # 跑全量（18 条，可断点续跑）
    python src/experiments/judge_title_pairs.py --dry      # 只列标题对，不调 LLM
    python src/experiments/judge_title_pairs.py --summary  # 只看已有结果的一致率
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.llm_client import LLMClient  # noqa: E402

OUT = ROOT / "experiments" / "cache" / "title_pair_judge.json"
CACHE = ROOT / "experiments" / "cache" / "arxiv_exists.json"
JUDGED = [
    ROOT / "experiments" / "baseline_llm" / "judged.jsonl",
    ROOT / "experiments" / "baseline_llm_strict" / "judged.jsonl",
]

# 判官只看这两条标题，别的一律不给。
SYSTEM = """你是学术引用核查员。用户会给你两条论文标题，你要判断它们是否指**同一篇论文**。

判为「同一篇」的情形（标题写法有偏差，但指向同一篇）：
- 语序不同、大小写/连字符/单复数差异
- 省略或改写了副标题、漏掉个别词
- 自造缩写与全称互相对应（如 APEFT ↔ Atomic Preference Enhanced Factuality Tuning）
- 给标题加了系统名/缩写前缀

判为「不是同一篇」的情形：
- 只是研究主题相近，实际是两篇不同的论文
- 缩写或专名对不上（例如一个叫 X，另一个叫 Y，且无对应关系）
- 关键实词不同，导致指向的贡献/数据集/方法不是同一个

只输出 JSON，不要解释以外的文字：
{"same_paper": true/false, "reason": "一句话依据"}"""

# 人工逐条复核结果（2026-09-23）。按 `norm_title` 前缀匹配；未列出的判为「同一篇」。
# 依据：18 条全部看过原始 `display`（`norm_title` 会丢大小写，缩写差异看不出来）。
MANUAL_NOT_SAME = {
    "fence a framework for evaluating the factual consistency":
        "真实是 Just ClozE! …；缩写 FenCE ≠ ClozE",
    "qa rag a hybrid retrieval augmented generation framework":
        "真实是 RAG-BioQA: …；缩写 QA-RAG ≠ RAG-BioQA",
    "federated learning for retrieval augmented generation":
        "真实是 Privacy-Preserving Federated **Embedding** Learning for Localized RAG，漏了 Embedding",
}

# 判官与人工不一致的 2 条 —— 人工复核后**维持原判**，理由是「口径分歧」而非判官出错。
# 这两条都是"主干逐字一致、只是尾部机制名/范围词被模型写错"：
#   判官按「这个标题能否**唯一确定**这篇论文」判 → 尾部不同 → 不是；
#   人工按「被引的这篇文献**是否真实存在**」判 → 主干一致 → 是。
# 这正是 `exists_variant` 这一态不可消除的模糊：它同时夹在两个问题中间。
MANUAL_DISPUTED = {
    "mitigating hallucination in large vision language models via dynamic":
        "前 9 词与真实标题逐字一致，只有尾部 mechanism 被写成 Dynamic Trust Calibration"
        "（真实是 Adaptive Attention Calibration）—— 模型记错机制名，但指向的文献确实存在",
    "self adaptive self adaptive retrieval augmented generation":
        "模型这一处写漏了 Multimodal（同一回答另一处写对了：SAM-RAG: Self-adaptive "
        "Multimodal Retrieval-Augmented Generation），指向的仍是同一篇",
}


def collect_pairs() -> dict[str, dict]:
    """从两套判定里取 exists_variant 的标题对，按 norm_title 去重。

    注意：候选标题（`match`）与 F1 不在 `judged.jsonl` 里 —— 宽松复核的结论存在
    `arxiv_exists.json` 的 `title_variant` 段，判定文件里只落了一个 status。所以要 join 缓存。
    """
    variants = json.loads(CACHE.read_text(encoding="utf-8")).get("title_variant", {})
    pairs: dict[str, dict] = {}
    for path in JUDGED:
        if not path.exists():
            print(f"[跳过] 找不到 {path}")
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for c in r.get("citations") or []:
                if c.get("kind") != "title" or c.get("status") != "exists_variant":
                    continue
                key = c.get("value") or ""
                v = variants.get(key) or {}
                pairs.setdefault(key, {
                    "model_title": c.get("display") or key,
                    "candidate": v.get("match") or "",
                    "f1": v.get("f1"),
                })
    return pairs


def manual_label(key: str) -> tuple[bool, str]:
    for prefix, why in MANUAL_NOT_SAME.items():
        if key.startswith(prefix):
            return False, why
    return True, ""


def summarize(cache: dict) -> int:
    done = {k: v for k, v in cache.items() if "same_paper" in v}
    if not done:
        print("还没有判官结果。")
        return 0
    agree = disagree = 0
    print(f"判官已判 {len(done)} 条\n")
    for key, v in sorted(done.items(), key=lambda x: (x[1]["same_paper"], x[0])):
        man, why = manual_label(key)
        ok = bool(v["same_paper"]) == man
        agree += ok
        disagree += (not ok)
        flag = "一致" if ok else "⚠️ 不一致"
        print(f"  [{flag}] 判官={'同一篇' if v['same_paper'] else '不是'} 人工={'同一篇' if man else '不是'}")
        print(f"      模型标题: {v.get('model_title', key)[:78]}")
        print(f"      候选标题: {(v.get('candidate') or '')[:78]}")
        print(f"      判官依据: {str(v.get('reason'))[:78]}")
        if not ok:
            print(f"      人工依据: {why or '（人工判同一篇）'}")
            for prefix, note in MANUAL_DISPUTED.items():
                if key.startswith(prefix):
                    print(f"      ⚖️ 裁定（人工维持原判）: {note}")
        print()
    n = agree + disagree
    judge_same = sum(1 for v in done.values() if v["same_paper"])
    manual_same = sum(1 for k in done if manual_label(k)[0])
    print(f"一致率 {agree}/{n} = {agree / n * 100:.1f}%（不一致 {disagree} 条）")
    print(f"判为「同一篇」：判官 {judge_same}/{n}，人工 {manual_same}/{n}")
    if disagree:
        print("不一致条目均为「人工判同一篇、判官判不是」，属**口径分歧**：")
        print("  判官按「该标题能否唯一确定这篇论文」判，人工按「被引文献是否真实存在」判。")
        print(f"  若改用判官的严口径，该态 18 → {judge_same} 条、精度 {judge_same / n * 100:.0f}%，")
        print(f"  判官判「不是」的 {n - judge_same} 条会回流 `fabricated`"
              f"（fabricated 141 → ~{141 + (n - judge_same)}）。")
        print("  **本次不采用**（方案甲：判官只做佐证，不改判定与指标）。")
    return 0 if disagree == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只列标题对，不调 LLM")
    ap.add_argument("--summary", action="store_true", help="只看已有结果的一致率")
    ap.add_argument("--limit", type=int, default=0, help="只判前 N 条（冒烟用）")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    if args.summary:
        return summarize(cache)

    pairs = collect_pairs()
    print(f"`exists_variant` 标题对（去重后）：{len(pairs)} 条")
    todo = {k: v for k, v in pairs.items() if "same_paper" not in cache.get(k, {})}
    print(f"已有判官结论：{len(pairs) - len(todo)} 条 | 待判：{len(todo)} 条")
    if args.limit:
        todo = dict(list(todo.items())[:args.limit])
        print(f"--limit 生效，本次只判 {len(todo)} 条")
    print()

    if args.dry:
        for k, v in pairs.items():
            print(f"  模型: {v['model_title'][:76]}")
            print(f"  候选: {(v['candidate'] or '')[:76]}   F1={v['f1']}")
            print()
        print("（--dry，未调 LLM）")
        return 0

    if not todo:
        print("无需调用。")
        return summarize(cache)

    client = LLMClient(temperature=0.0)
    for i, (k, v) in enumerate(todo.items(), 1):
        prompt = (f"【标题 A（模型写的）】\n{v['model_title']}\n\n"
                  f"【标题 B（候选真实标题）】\n{v['candidate'] or '（无）'}\n\n"
                  f"它们是否指同一篇论文？")
        try:
            out = client.chat_json(SYSTEM, prompt)
            rec = {
                "same_paper": bool(out.get("same_paper")),
                "reason": str(out.get("reason") or "")[:300],
                "model_title": v["model_title"],
                "candidate": v["candidate"],
                "f1": v["f1"],
            }
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {k[:56]}（{type(exc).__name__}: {exc}）")
            continue
        cache[k] = rec
        OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        man, _ = manual_label(k)
        ok = rec["same_paper"] == man
        print(f"  {i}/{len(todo)} 判官={'同一篇' if rec['same_paper'] else '不是'} "
              f"人工={'同一篇' if man else '不是'} {'✓' if ok else '⚠️ 不一致'}  {k[:52]}")

    print(f"\n{client.stats()}")
    print(f"结果已写 {OUT.relative_to(ROOT)}（**未改动任何判定与指标**）\n")
    return summarize(cache)


if __name__ == "__main__":
    raise SystemExit(main())
