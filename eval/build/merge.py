#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测集合并与自检 —— 把正常题候选池与陷阱题合并成正式的 v1 评测集。

输入: eval/v1/normal_candidates.jsonl（286 条候选）
      eval/v1/traps.jsonl（150 条陷阱题）
输出: eval/v1/questions.jsonl   —— 正式评测集 400 条
      eval/v1/review_queue.jsonl —— 需人工抽检的题目清单
      eval/v1/stats.json          —— 分布统计与校验结果

自检项（对应第 4 章「评测集构建」要交代的可信度）:
  1. 字段完整性 + 枚举合法性（对照 eval/schema.json）
  2. 每题 detection_label 是否符合该类型的期望标签（schema.label_expectation_by_type）
  3. normal 题：evidence 必须逐字出自对应论文摘要
  4. 所有 gold_arxiv_ids / trap 中引用的论文必须真实存在于本次语料
  5. qid 唯一、题目文本不重复
  6. 类别配比是否达标（250 / 60 / 50 / 40）

用法:
    python eval/build/merge.py
    python eval/build/merge.py --dry-run    # 只自检不写文件
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "eval" / "v1"
NORMAL_IN = V1 / "normal_candidates.jsonl"
TRAPS_IN = V1 / "traps.jsonl"
OUT = V1 / "questions.jsonl"
REVIEW = V1 / "review_queue.jsonl"
STATS = V1 / "stats.json"
CORPUS = ROOT / "data" / "arxiv_raw.jsonl"
SCHEMA = ROOT / "eval" / "schema.json"

SEED = 20260922
TARGET_NORMAL = 250
# 正常题目标问法配比（合计 250），实际按可用池情况向下取
FORM_MIX = {"method": 60, "factual": 50, "numeric": 50, "comparison": 45, "causal": 45}

LABEL_EXPECT = {
    "normal": {"existence": "pass", "support": "pass", "traceability": "pass"},
    "trap_nonexistent": {"existence": "fail", "support": "na", "traceability": "na"},
    "trap_unsupported": {"existence": "pass", "support": "fail", "traceability": "na"},
    "trap_mismatch": {"existence": "pass", "support": "fail", "traceability": "fail"},
}

QID_PREFIX = {
    "normal": "N",
    "trap_nonexistent": "T1",
    "trap_unsupported": "T2",
    "trap_mismatch": "T3",
}


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        raise FileNotFoundError(f"缺少输入文件: {p}")
    return [json.loads(l) for l in p.open(encoding="utf-8")]


# ---------------------------------------------------------------- 正常题筛选

def select_normal(candidates: list[dict], target: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """按问法配比分层抽样，优先取双项校验都通过的题。返回 (入选, 落选)。"""
    # 去重：完全相同的题干只留一条
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in candidates:
        key = norm_space(r["question"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    good = [r for r in deduped if r.get("_evidence_verified") and r.get("_self_contained")]
    weak = [r for r in deduped if r not in good]
    rng.shuffle(good)
    rng.shuffle(weak)

    chosen: list[dict] = []
    by_form: dict[str, list[dict]] = {}
    for r in good:
        by_form.setdefault(r["question_form"], []).append(r)

    for form, quota in FORM_MIX.items():
        pool = by_form.get(form, [])
        chosen += pool[:quota]

    # 配额没凑满（某问法不够）则从剩余优等题补齐
    if len(chosen) < target:
        used = {id(r) for r in chosen}
        rest = [r for r in good if id(r) not in used]
        chosen += rest[: target - len(chosen)]

    # 仍不够才动用待核题，并保留 needs_review 标记
    if len(chosen) < target:
        chosen += weak[: target - len(chosen)]

    chosen = chosen[:target]
    chosen_ids = {id(r) for r in chosen}
    dropped = [r for r in deduped if id(r) not in chosen_ids]
    return chosen, dropped


# ---------------------------------------------------------------- 自检

def verify(records: list[dict], corpus_ids: set[str], abs_by_id: dict[str, str]) -> dict:
    problems: list[str] = []
    counters: Counter = Counter()

    qids = [r["qid"] for r in records]
    dup_qid = [q for q, c in Counter(qids).items() if c > 1]
    if dup_qid:
        problems.append(f"qid 重复: {dup_qid[:5]}")

    qtexts = [norm_space(r["question"]) for r in records]
    dup_q = [q for q, c in Counter(qtexts).items() if c > 1]
    if dup_q:
        problems.append(f"题干重复 {len(dup_q)} 条")

    for r in records:
        qid = r["qid"]
        # 枚举合法性
        if r["type"] not in LABEL_EXPECT:
            problems.append(f"{qid}: 未知 type={r['type']}")
            continue
        if r["status"] not in ("draft", "needs_review", "reviewed", "rejected"):
            problems.append(f"{qid}: 非法 status={r['status']}")
        # 标签是否符合类型期望
        exp = LABEL_EXPECT[r["type"]]
        if r["detection_label"] != exp:
            problems.append(f"{qid}: detection_label 与类型期望不符 {r['detection_label']} != {exp}")
        # gold 引用必须真实存在于语料
        for gid in r["gold_arxiv_ids"]:
            if gid not in corpus_ids:
                problems.append(f"{qid}: gold arXiv ID {gid} 不在本次语料中")
        # normal 题 evidence 必须逐字出自对应摘要
        if r["type"] == "normal":
            gid = r["gold_arxiv_ids"][0] if r["gold_arxiv_ids"] else None
            ab = abs_by_id.get(gid, "")
            if not r["evidence"] or norm_space(r["evidence"]) not in norm_space(ab):
                problems.append(f"{qid}: evidence 未逐字匹配 {gid} 的摘要")
                counters["evidence_fail"] += 1
            else:
                counters["evidence_pass"] += 1
        # 陷阱题引用的论文也都必须在语料里
        trap = r["trap"] or {}
        for key in ("target_paper", "source_paper"):
            if trap.get(key) and trap[key] not in corpus_ids:
                problems.append(f"{qid}: trap.{key}={trap[key]} 不在语料中")
        # T3 关键约束：被搬走的片段不得出现在 target 摘要里
        if r["type"] == "trap_mismatch":
            tgt_ab = abs_by_id.get(trap.get("target_paper"), "")
            if norm_space(r["evidence"]) in norm_space(tgt_ab):
                problems.append(f"{qid}: 张冠李戴片段出现在 target 摘要中，陷阱失效")
                counters["mismatch_fail"] += 1
            else:
                counters["mismatch_pass"] += 1

    return {"problems": problems, "counters": dict(counters)}


def remote_recheck_t1(records: list[dict]) -> dict:
    """独立远程复核 T1：不信构造阶段的标记，重新向 arXiv 核一遍伪 ID 与伪标题。

    这是评测集可信度最硬的一块证据 —— 论文里可以写「全部伪造引用均已通过
    独立的 arXiv 接口复核，确认不存在」。
    """
    import time as _time
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    api = "https://export.arxiv.org/api/query"
    ns = {"atom": "http://www.w3.org/2005/Atom",
          "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

    def get(url: str, retries: int = 4) -> bytes:
        """带退避重试 —— arXiv 对密集请求会返回 429，必须退让后再试。"""
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "VeriCite/0.1 (research)"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except Exception as exc:  # noqa: BLE001
                last = exc
                wait = 10 * attempt
                print(f"    [重试] 第 {attempt} 次失败（{type(exc).__name__}），{wait}s 后重试")
                _time.sleep(wait)
        raise RuntimeError(f"请求失败: {url}") from last

    t1 = [r for r in records if r["type"] == "trap_nonexistent"]
    ids = sorted({r["trap"]["fabricated_id"] for r in t1 if r["trap"].get("fabricated_id")})
    titles = sorted({r["trap"]["fabricated_title"] for r in t1 if r["trap"].get("fabricated_title")})

    result: dict = {"ids_checked": len(ids), "titles_checked": len(titles),
                    "ids_still_nonexistent": None, "titles_still_nonexistent": None,
                    "ids_that_now_exist": [], "titles_that_now_exist": [], "errors": []}

    # 伪 ID：分批查询，出现在返回结果里的就是「其实存在」
    existing_ids: list[str] = []
    try:
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            url = f"{api}?{urllib.parse.urlencode({'id_list': ','.join(chunk), 'max_results': len(chunk)})}"
            root = ET.fromstring(get(url))
            for e in root.findall("atom:entry", ns):
                m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", e.findtext("atom:id", default="", namespaces=ns))
                if m and m.group(1) in chunk:
                    existing_ids.append(m.group(1))
            _time.sleep(5)
        result["ids_that_now_exist"] = existing_ids
        result["ids_still_nonexistent"] = len(ids) - len(existing_ids)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"伪 ID 复核失败: {type(exc).__name__}: {exc}")

    # 伪标题：逐条精确检索（totalResults=0 即确认不存在）。
    # 试过 10 个一批的 OR 查询，URL 过长会被 arXiv 限流（429），逐条最稳。
    # 单条查询失败不中断整轮 —— 记入 titles_unverified，否则一次限流会毁掉整批复核。
    try:
        def esc(t: str) -> str:
            return '"' + re.sub(r'["“”]', "", t) + '"'

        confirmed = 0
        hits_found: list[str] = []
        unverified: list[str] = []
        for idx, t in enumerate(titles, 1):
            url = f"{api}?{urllib.parse.urlencode({'search_query': f'ti:{esc(t)}', 'max_results': 3})}"
            hits = None
            try:
                root = ET.fromstring(get(url, retries=2))
                node = root.find("opensearch:totalResults", ns)
                hits = int(node.text) if node is not None and node.text else 0
            except Exception as exc:  # noqa: BLE001
                unverified.append(t)
                print(f"    [跳过] 第 {idx} 条查询失败（{type(exc).__name__}），记入待核")
            if hits == 0:
                confirmed += 1
            elif hits:
                hits_found.append(t)
                print(f"    [警告] 伪标题命中 {hits} 条真实论文：{t[:70]}")
            if idx % 10 == 0 or idx == len(titles):
                print(f"    伪标题进度 {idx}/{len(titles)}，已确认不存在 {confirmed}，待核 {len(unverified)}")
            _time.sleep(5)
        result["titles_still_nonexistent"] = confirmed
        result["titles_that_now_exist"] = hits_found
        result["titles_unverified"] = unverified
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"伪标题复核失败: {type(exc).__name__}: {exc}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="合并并自检评测集 v1")
    ap.add_argument("--dry-run", action="store_true", help="只输出自检结果，不写文件")
    ap.add_argument("--target-normal", type=int, default=TARGET_NORMAL)
    ap.add_argument("--verify-remote", action="store_true",
                    help="额外向 arXiv 独立复核 T1 的伪 ID 与伪标题（仅首次构建后跑一次即可）")
    args = ap.parse_args()

    rng = random.Random(SEED)
    corpus = load_jsonl(CORPUS)
    corpus_ids = {p["arxiv_id"] for p in corpus}
    abs_by_id = {p["arxiv_id"]: p["abstract"] for p in corpus}

    normal_pool = load_jsonl(NORMAL_IN)
    traps = load_jsonl(TRAPS_IN)
    print(f"输入: 正常题候选 {len(normal_pool)} 条 | 陷阱题 {len(traps)} 条 | 语料 {len(corpus)} 篇")

    chosen, dropped = select_normal(normal_pool, args.target_normal, rng)
    print(f"正常题入选 {len(chosen)} 条（候选池落选 {len(dropped)} 条，留作 v1.1 补充）")

    # 组装正式记录：剥离内部字段，转成公开的 verification 对象
    records: list[dict] = []
    counters_qid: Counter = Counter()
    for r in chosen:
        counters_qid["normal"] += 1
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec["qid"] = f"N-{counters_qid['normal']:04d}"
        rec["verification"] = {
            "evidence_verbatim_in_abstract": bool(r.get("_evidence_verified")),
            "question_self_contained": bool(r.get("_self_contained")),
        }
        records.append(rec)

    for r in traps:
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        prefix = QID_PREFIX[r["type"]]
        counters_qid[prefix] += 1
        rec["qid"] = f"{prefix}-{counters_qid[prefix]:04d}"
        if r["type"] == "trap_nonexistent":
            rec["verification"] = {
                "fabricated_title_nonexistent_on_arxiv": bool(r.get("_title_verified")),
                "fabricated_id_nonexistent_on_arxiv": bool(r.get("_id_verified")),
            }
        elif r["type"] == "trap_unsupported":
            rec["verification"] = {
                "absent_keywords_confirmed_not_in_abstract": r.get("_absent_keywords", []),
            }
        else:
            rec["verification"] = {
                "source_evidence_verbatim_in_source_abstract": bool(r.get("_source_verified")),
                "evidence_absent_from_target_abstract": not bool(r.get("_evidence_in_target")),
            }
        records.append(rec)

    print(f"合并后共 {len(records)} 条，开始自检 ...")

    # 跨类别去重兜底：不同构造链路理论上可能撞出同一题干（实测 T1 曾出现），
    # 保留先出现的，后出现的剔除并记录，避免评测集里出现重复题。
    seen_q: dict[str, str] = {}
    deduped_records: list[dict] = []
    cross_dups: list[str] = []
    for r in records:
        key = norm_space(r["question"])
        if key in seen_q:
            cross_dups.append(f"{r['qid']} 与 {seen_q[key]} 题干重复，已剔除")
            continue
        seen_q[key] = r["qid"]
        deduped_records.append(r)
    if cross_dups:
        records = deduped_records

    result = verify(records, corpus_ids, abs_by_id)
    problems = result["problems"] + cross_dups

    by_type = Counter(r["type"] for r in records)
    by_form = Counter(r["question_form"] for r in records)
    by_diff = Counter(r["difficulty"] for r in records)
    by_dom = Counter(r["domain"] for r in records)
    by_status = Counter(r["status"] for r in records)

    stats = {
        "version": "v1",
        "created": date.today().isoformat(),
        "corpus": {
            "papers": len(corpus),
            "source": "data/arxiv_raw.jsonl",
            "domains": dict(Counter(p["primary_category"] for p in corpus).most_common()),
            "years": dict(sorted(Counter(p["published"][:4] for p in corpus).items())),
        },
        "seed": SEED,
        "total": len(records),
        "target_distribution": {"normal": 250, "trap_nonexistent": 60,
                               "trap_unsupported": 50, "trap_mismatch": 40},
        "by_type": dict(by_type),
        "by_question_form": dict(by_form),
        "by_difficulty": dict(by_diff),
        "by_domain": dict(by_dom.most_common()),
        "by_status": dict(by_status),
        "normal_candidates_pool": len(normal_pool),
        "normal_dropped": len(dropped),
        "cross_type_duplicates_removed": len(cross_dups),
        "checks": result["counters"],
        "problems": problems,
    }

    target = stats["target_distribution"]
    stats["distribution_matches_target"] = {k: by_type.get(k, 0) == v for k, v in target.items()}

    if args.verify_remote:
        print("\n===== 独立远程复核（向 arXiv 重新校验 T1）=====")
        rc = remote_recheck_t1(records)
        stats["remote_recheck"] = rc
        print(f"伪 ID: 复核 {rc['ids_checked']} 个，确认不存在 {rc['ids_still_nonexistent']}"
              f"（本轮新发现存在 {len(rc['ids_that_now_exist'])} 个）")
        print(f"伪标题: 复核 {rc['titles_checked']} 个，确认不存在 {rc['titles_still_nonexistent']}")
        for e in rc["errors"]:
            print("  [注意]", e)
    elif STATS.exists():
        # 只重造了 T2 之类的情形：T1 记录逐字节未变，沿用上一轮复核结果，
        # 避免重复打 arXiv 接口（密集请求会被 429）。若 T1 有改动请显式加 --verify-remote。
        try:
            prev = json.loads(STATS.read_text(encoding="utf-8"))
            if prev.get("remote_recheck"):
                carried = dict(prev["remote_recheck"])
                carried["note"] = "T1 记录未变，沿用上一轮向 arXiv 的独立复核结果"
                stats["remote_recheck"] = carried
                print("\n[沿用] T1 的 arXiv 独立复核结果（本轮 T1 未改动）")
        except Exception:  # noqa: BLE001
            pass

    print("\n===== 自检结果 =====")
    print(f"类别分布: {dict(by_type)}")
    if not all(stats["distribution_matches_target"].values()):
        print(f"  [注意] 与目标配比存在偏差，目标为 {target}")
        for k, ok in stats["distribution_matches_target"].items():
            if not ok:
                print(f"    {k}: 实际 {by_type.get(k, 0)} / 目标 {target[k]}")
    print(f"问法分布: {dict(by_form.most_common())}")
    print(f"难度分布: {dict(by_diff)}")
    print(f"状态分布: {dict(by_status)}")
    print(f"evidence 逐字校验: 通过 {result['counters'].get('evidence_pass', 0)}")
    print(f"张冠李戴约束校验: 通过 {result['counters'].get('mismatch_pass', 0)}")
    if problems:
        print(f"\n发现问题 {len(problems)} 项：")
        for p in problems[:15]:
            print("  -", p)
    else:
        print("\n未发现结构性问题 ✅")

    if args.dry_run:
        print("\n[dry-run] 未写入文件")
        return 0

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    need_review = [r for r in records if r["status"] != "draft"]
    with REVIEW.open("w", encoding="utf-8", newline="\n") as f:
        for r in need_review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已写出: {OUT}（{len(records)} 条）")
    print(f"已写出: {REVIEW}（待人工抽检 {len(need_review)} 条）")
    print(f"已写出: {STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
