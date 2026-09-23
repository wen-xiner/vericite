#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第一层复核：对「查无此文」的标题做二次核验，估计假阳性率。

为什么需要这一步（2026-09-22/23）：
    宽松复核（existence.py 的 lenient_title_match）用「5 个最长实词的 AND」检索，
    真实标题若不含其中某个词就会 0 结果 → F1=0 → 被判 fabricated（编造）。
    实测在 11 条抽样里就有 1 条属误判：
        「measuring and reducing llm hallucination without gold standards」
        实际是「Self-Known and Self-Unknown: Measuring and Reducing LLM Hallucination」（F1 0.800）
    所以 fabricated 的计数含假阳性，论文里报数必须注明这是软证据。

做法：把检索式放宽两档再试一遍，任一档出现词面 F1 ≥ 0.6 的候选，就说明
    「该研究方向真实存在、只是模型写的标题不对」——属属性错误，不是凭空捏造。
        AND3 —— 只取 3 个最长实词做 ti: 交集（比 5 词松）
        all  —— 全字段包含检索（匹配摘要/正文，比标题检索松）
    两档都没到 0.6 才认定为「未见真实对应」，仍按编造计。

结果增量写入 experiments/cache/title_second_pass.json，可反复跑、中断续跑。

用法:
    python verify_unmatched.py                 # 跑全部待核验标题
    python verify_unmatched.py --limit 30      # 只跑前 30 个
    python verify_unmatched.py --summary       # 只看统计，不发请求
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.arxiv_api import ArxivApi  # noqa: E402
from common.textutil import content_tokens, norm_title, token_f1  # noqa: E402

CACHE = ROOT / "experiments" / "cache" / "title_second_pass.json"
TRIGGER = 0.6  # 放宽检索后达到这个相似度，认为「真实文献存在、标题写错了」


def load_records() -> list[dict]:
    out = []
    for tag in ("baseline_llm", "baseline_llm_strict"):
        p = ROOT / "experiments" / tag / "judged.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def trap_titles() -> set[str]:
    """评测集自带的伪标题：已知必不存在，是构造陷阱，不参与假阳性复核。"""
    out = set()
    q = ROOT / "eval" / "v1" / "questions.jsonl"
    with q.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ft = (json.loads(line).get("trap") or {}).get("fabricated_title")
            if ft:
                out.add(norm_title(ft))
    return out


def collect_targets(records: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
    """收集被判「查无此文」的标题（去重），并记下它出现在哪些题里。"""
    traps = trap_titles()
    where: dict[str, list[str]] = {}
    for r in records:
        for c in r.get("citations") or []:
            if c.get("kind") != "title" or c.get("status") != "fabricated":
                continue
            key = norm_title(c.get("display") or c.get("value") or "")
            if not key or key in traps:
                continue
            where.setdefault(key, []).append(r["qid"])
    return list(where), where


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_cache(data: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="对「查无此文」的标题做二次核验")
    ap.add_argument("--limit", type=int, default=None, help="本次最多核验多少个")
    ap.add_argument("--summary", action="store_true", help="只打印统计，不发请求")
    ap.add_argument("--interval", type=float, default=3.0, help="每次请求后的间隔秒数")
    args = ap.parse_args()

    records = load_records()
    targets, where = collect_targets(records)
    cache = load_cache()

    done = [t for t in targets if t in cache]
    todo = [t for t in targets if t not in cache]
    print(f"被引记录 {len(records)} 条 | 判「查无此文」的去重标题 {len(targets)} 个"
          f"（已核验 {len(done)}）")

    hit = [t for t in done if cache[t].get("found")]
    print(f"已核验 {len(done)} 个：其中 {len(hit)} 个在放宽检索后找到真实对应"
          f"（假阳性率 {len(hit) / len(done):.1%}）" if done else "已核验 0 个")

    if args.summary:
        for t in sorted(hit, key=lambda x: -cache[x]["f1"]):
            print(f"\n  f1={cache[t]['f1']:.3f} [{cache[t]['probe']}] {t[:66]}")
            print(f"        → {cache[t]['match'][:76]}")
            print(f"        出现在: {', '.join(where.get(t, []))}")
        return 0

    if not todo:
        print("没有待核验的标题 ✅")
        return 0

    if args.limit:
        todo = todo[:args.limit]
    est = len(todo) * (args.interval * 2 + 2)
    print(f"本次核验 {len(todo)} 个，预计 {est / 60:.0f} 分钟（已核验的跳过，可中断续跑）\n")

    api = ArxivApi(quiet=True)
    for n, t in enumerate(todo, 1):
        toks = sorted(set(content_tokens(t)), key=len, reverse=True)
        probes = []
        if len(toks) >= 3:
            probes.append(("AND3", " AND ".join(f"ti:{x}" for x in toks[:3])))
        probes.append(("all", f'all:"{t}"'))

        best, best_f1, best_probe = None, 0.0, "-"
        for tag, q in probes:
            for cand in api.search_titles(q, 5):
                f = token_f1(content_tokens(t), content_tokens(cand))
                if f > best_f1:
                    best, best_f1, best_probe = cand, f, tag
            time.sleep(args.interval)

        cache[t] = {"match": best, "f1": round(best_f1, 3), "probe": best_probe,
                    "found": best_f1 >= TRIGGER, "checked": True,
                    "qids": where.get(t, [])}
        save_cache(cache)
        mark = "★ 找到真实对应" if cache[t]["found"] else ("~ 部分相似" if best_f1 >= 0.4 else "· 无")
        print(f"  {n:>3}/{len(todo)} {mark} f1={best_f1:.3f} [{best_probe}] {t[:56]}")

    found = [t for t in targets if cache.get(t, {}).get("found")]
    checked = [t for t in targets if t in cache]
    print(f"\n===== 二次核验完成：{len(checked)}/{len(targets)} =====")
    print(f"  放宽检索后找到真实对应：{len(found)} 个"
          f"（占已核验 {len(found) / max(len(checked), 1):.1%}）")
    print(f"  这些属于「研究方向真实、模型标题写错」的属性错误，不应计为凭空捏造。")
    print(f"  缓存: {CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
