"""一次性脚本：把二次核验里「人工确认成立」的标题回灌为 exists_variant。

背景：宽松复核用 5 词 AND 检索，模型把标题压成核心词拼串 / 换语序时 F1 只有 0.6~0.7，
卡在 0.7 阈值下被误判成 fabricated。二次核验（verify_unmatched.py）用更宽的检索式
找回了一批真实对应，这里把确认成立的回灌进 title_variant，让重跑第一层时自动改判。

判定标准（保守）：匹配标题必须包含模型标题的**独特词组**（自造词、长短语），
仅"同主题"不算。逐条依据见 AUTO_CONFIRMED 的 reason。

用法: python backfill_second_pass.py          # 执行回灌
      python backfill_second_pass.py --dry    # 只看会改哪些
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "experiments" / "cache" / "arxiv_exists.json"
SECOND = ROOT / "experiments" / "cache" / "title_second_pass.json"

# key 前缀 → 判定依据（前缀匹配，避免手打整串出错）
AUTO_CONFIRMED = {
    "mitigating hallucination in large vision language models via":
        "前缀逐字一致，仅结尾 mechanism 不同（via Dynamic… vs via Adaptive Attention）",
    "towards fine grained citation in verifiable generation":
        "核心词集合完全一致（fine-grained / citation / verifiable / generation），仅语序不同",
    "hetarag hybrid retrieval augmented generation":
        "自造词 HetaRAG 对上，hetarag + hybrid + heterogeneous 三项吻合",
    "data importance learning retrieval augmented llm":
        "独特词组 data importance learning 逐字一致",
    "apeft atomic preference enhanced factuality tuning":
        "长词组 atomic preference enhanced factuality tuning 完全一致，APEFT 是对应缩写",
}

# 被否掉的候选（留证，别再回灌）：
#   etrag a general framework for relation extraction with retrieval
#     → 曾误回灌为 exists_variant（F1 0.769），但 2026-09-23 实测 `ti:"ETRAG"` 命中 0 条，
#       即世上没有以 ETRAG 命名的论文；真实对应标题 Retrieval-Augmented Generation-based
#       Relation Extraction 里根本没有 ETRAG。0.769 全部来自 retrieval/augmented/
#       generation/relation/extraction 这些**主题通用词**，属"同主题的另一篇"，应留 fabricated。
#       —— 这条恰好暴露了 F1 单指标的病灶：主题词重叠就能把相似度顶到 0.7 以上。


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只打印，不写回")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    second = json.loads(SECOND.read_text(encoding="utf-8"))
    variants = cache["title_variant"]

    changed = []
    for prefix, reason in AUTO_CONFIRMED.items():
        keys = [k for k in second if k.startswith(prefix)]
        if not keys:
            print(f"[找不到] {prefix}")
            continue
        for k in keys:
            s = second[k]
            if not s.get("found"):
                print(f"[跳过·二次核验未找到对应] {k[:60]}")
                continue
            v = variants.setdefault(k, {})
            before = v.get("similar")
            v["match"] = s.get("match")
            v["f1"] = s.get("f1")
            v["checked"] = True
            v["similar"] = True
            v["second_pass"] = {"probe": s.get("probe"), "note": reason}
            changed.append((k, before, s.get("f1"), s.get("match"), reason))

    print(f"\n本次回灌 {len(changed)} 条：\n")
    for k, before, f1, match, reason in changed:
        print(f"  {k[:64]}")
        print(f"     similar: {before} → True （二次核验 F1={f1}）")
        print(f"     匹配到 : {(match or '')[:76]}")
        print(f"     依据   : {reason}")

    if args.dry:
        print("\n（--dry，未写回）")
        return 0

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写回 {CACHE.relative_to(ROOT)}")
    print("接着重跑：python src/experiments/run_eval.py --answers … --refresh-layer1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
