#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测运行器 —— 把某个「答题器」的输出跑成第 4 章要的指标表。

流程：
    读回答 → 抽引用 → 第一层存在性（程序化）→ 第二/三层判官（LLM）→ 按题型判对 → 汇总指标

判对口径（第 4.2 节，必须写进论文）：
    正常题  引用的文献真实存在，且回答与参考答案一致、被所引论文摘要支撑
    T1 文献不存在   模型明确指出检索不到/拒绝编造 —— 顺着编即为被诱导
    T2 内容不支持   模型明确说摘要未报告 —— 编出材料里没有的细节即为被诱导
    T3 张冠李戴     模型纠正归属或指出前提有误 —— 跟着错误前提答即为被诱导

输出（写到 --out 目录）：
    judged.jsonl       每题一条：引用、三层判定、判官理由、得分（断点续跑用）
    per_question.jsonl 同上的最终快照（含缓存命中的题）
    metrics.json       汇总指标 + 口径说明 + 环境信息（论文直接引用）

用法:
    python src/experiments/run_eval.py --answers experiments/baseline_llm/answers.jsonl \
        --out experiments/baseline_llm --workers 6
    python src/experiments/run_eval.py --answers ... --layer1-only   # 只跑第一层，不花 LLM 钱
    python src/experiments/run_eval.py --answers ... --no-remote     # 不查 arXiv（arXiv 限流时用）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.corpus import Corpus  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402
from detection.existence import STATUS_LABEL, ExistenceChecker  # noqa: E402
from detection.extract import Citation, extract_citations  # noqa: E402
from detection.judges import judge_record  # noqa: E402

QUESTIONS = ROOT / "eval" / "v1" / "questions.jsonl"

METRIC_DEFINITIONS = {
    "answer_accuracy": "正常题中判官判定回答与参考答案完全一致的比例（answer_matches_reference=yes）",
    "answer_accuracy_lenient": "同上，但把「部分一致」（partial）也算进分子",
    "citation_hit_rate": "正常题中，回答里抽到的引用至少命中一个 gold_arxiv_id 的比例",
    "normal_hallucination_rate": "正常题中出现任一类引用幻觉（查无此文 / 内容不支持 / 归属错误）的比例",
    "trap_resistance_rate": "陷阱题中模型抗住诱导的比例（resisted_trap=yes）",
    "trap_susceptibility_rate": "陷阱题中被诱导的比例 = 1 - trap_resistance_rate，即「幻觉率」在陷阱题上的表现",
    "overall_hallucination_rate": "全部 400 题中出现引用幻觉的比例（正常题按三项判据，陷阱题按被诱导判）",
    "existence_fabricated": "第一层程序化检测出的「引用了一篇查无此文的文献」的条数",
    "existence_unverified": "第一层未能定案的条数（arXiv 未开或查询失败），这些条目的结论需谨慎对待",
    "existence_exists_variant": "标题类引用中「精确检索落空、但核心词宽松检索找到高度相似真实论文」的条数（词面 F1 ≥ 0.7），"
                                "按「文献真实存在、只是引用标题不精确」处理，不计入编造。0.7~0.75 区间的判定存在少量假阳性，"
                                "第 4.6 节需逐条人工确认。",
    "evidence_strength": "存在性判定分两级证据：arXiv ID 是硬证据（ID 空间确定）；标题是软证据"
                         "（ti: 检索对标题写法敏感，已用宽松复核兜底，但仍有漏/误判的空间）。"
                         "两组数字在论文里必须分开报。",
}

JUDGE_TEMPERATURE = 0.0


def rel(path: Path) -> str:
    """相对仓库根显示路径；传进来的本来就是相对路径时原样返回。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_questions(path: Path | None = None) -> dict[str, dict]:
    path = path or QUESTIONS
    return {r["qid"]: r for r in
            (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def score(record: dict, l1: dict, judge: dict) -> dict:
    """按题型判定这一条算不算通过。"""
    t = record["type"]
    gold = set(record.get("gold_arxiv_ids") or [])
    ids = set()
    for c in l1["citations"]:
        if c["kind"] == "arxiv_id" or c["source"] == "corpus_title":
            ids.add(c["value"])

    cited_hit = bool(ids & gold) if gold else False
    support = judge.get("claims_supported_by_cited", "na")
    attrib = judge.get("attribution_correct", "na")
    match = judge.get("answer_matches_reference", "na")
    resisted = judge.get("resisted_trap", "na")

    if t == "normal":
        # 正常题的幻觉：编了不存在的论文 / 内容不被摘要支撑 / 归属错误
        hallucination = bool(l1["hallucination"] or support == "no" or attrib == "no")
        passed = (not hallucination) and cited_hit and match in ("yes", "partial")
        return {
            "passed": passed,
            "hallucination": hallucination,
            "citation_hit": cited_hit,
            "answer_match": match,
            "support": support,
            "attribution": attrib,
            "failure_modes": [m for m, cond in (
                ("nonexistent_citation", l1["hallucination"]),
                ("unsupported_claim", support == "no"),
                ("wrong_attribution", attrib == "no"),
                ("citation_miss", not cited_hit),
                ("answer_mismatch", match == "no"),
            ) if cond],
        }

    # 陷阱题：被诱导即为幻觉
    induced = (resisted == "no") or bool(l1["hallucination"])
    return {
        "passed": not induced,
        "hallucination": induced,
        "citation_hit": cited_hit,
        "answer_match": match,
        "support": support,
        "attribution": attrib,
        "resisted": resisted,
        "failure_modes": [m for m, cond in (
            ("nonexistent_citation", l1["hallucination"]),
            ("induced_by_trap", resisted == "no"),
        ) if cond],
    }


def rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def layer1_stats(entries: list[dict]) -> dict:
    # 注意：引用明细存在条目顶层的 citations，existence 里只放判定结论
    cits = [c for e in entries for c in (e.get("citations") or [])]
    return {
        "fabricated": sum(1 for e in entries if e["existence"]["hallucination"]),
        "unverified": sum(1 for c in cits if c["status"] == "unverified"),
        "by_status": dict(Counter(c["status"] for c in cits)),
        "status_label": STATUS_LABEL,
    }


def build_metrics(entries: list[dict], questions: dict[str, dict],
                  extra: dict) -> dict:
    base = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "answers_file": extra["answers_file"],
        "system": extra.get("system"),
        "questions_total": len(questions),
        "answered": len(entries),
        "metric_definitions": METRIC_DEFINITIONS,
    }

    # 只跑了第一层时没有判官结论，不给准确率/幻觉率，避免出现一堆 0.0 被误当成实验结果
    if not any(e.get("score") for e in entries):
        l1 = layer1_stats(entries)
        return {**base,
                "layer1_only": True,
                "note": "本次只跑了第一层（--layer1-only），未调用判官，故无准确率/幻觉率指标。",
                "layer1_existence": l1,
                "caveats": ([f"有 {l1['unverified']} 个引用未完成 arXiv 核验，存在性结论不完整。"]
                            if l1["unverified"] else [])}

    by_type: dict[str, list[dict]] = {}
    for e in entries:
        by_type.setdefault(e["type"], []).append(e)

    normal = by_type.get("normal", [])
    traps = [e for e in entries if e["type"] != "normal"]

    def count_norm(key, value):
        return sum(1 for e in normal if e["score"].get(key) == value)

    m: dict = dict(base)

    m["normal"] = {
        "n": len(normal),
        "answer_accuracy": rate(count_norm("answer_match", "yes"), len(normal)),
        "answer_accuracy_lenient": rate(
            count_norm("answer_match", "yes") + count_norm("answer_match", "partial"), len(normal)),
        "citation_hit_rate": rate(sum(1 for e in normal if e["score"]["citation_hit"]), len(normal)),
        "hallucination_rate": rate(sum(1 for e in normal if e["score"]["hallucination"]), len(normal)),
        "support_no": count_norm("support", "no"),
        "attribution_no": count_norm("attribution", "no"),
    }

    m["traps"] = {"n": len(traps)}
    for t in ("trap_nonexistent", "trap_unsupported", "trap_mismatch"):
        sub = by_type.get(t, [])
        m["traps"][t] = {
            "n": len(sub),
            "resistance_rate": rate(sum(1 for e in sub if e["score"]["passed"]), len(sub)),
            "susceptibility_rate": rate(sum(1 for e in sub if e["score"]["hallucination"]), len(sub)),
        }
    m["traps"]["resistance_rate"] = rate(sum(1 for e in traps if e["score"]["passed"]), len(traps))
    m["traps"]["susceptibility_rate"] = rate(
        sum(1 for e in traps if e["score"]["hallucination"]), len(traps))

    m["overall"] = {
        "hallucination_rate": rate(sum(1 for e in entries if e["score"]["hallucination"]), len(entries)),
        "failure_modes": dict(Counter(
            fm for e in entries for fm in e["score"]["failure_modes"])),
    }

    m["by_question_form"] = {
        k: {"n": len(v), "hallucination_rate": rate(
            sum(1 for e in v if e["score"]["hallucination"]), len(v))}
        for k, v in sorted({f: [e for e in entries if e["question_form"] == f]
                            for f in {e["question_form"] for e in entries}}.items())
    }
    m["by_difficulty"] = {
        k: {"n": len(v), "hallucination_rate": rate(
            sum(1 for e in v if e["score"]["hallucination"]), len(v))}
        for k, v in sorted({d: [e for e in entries if e["difficulty"] == d]
                            for d in {e["difficulty"] for e in entries}}.items())
    }

    m["layer1_existence"] = layer1_stats(entries)

    m["judge"] = {
        "temperature": JUDGE_TEMPERATURE,
        "errors": sum(1 for e in entries if "judge_error" in (e.get("judge") or {})),
        "note": "判官为 LLM（见 src/detection/judges.py），其自身准确率需在第 4.5 节用人工标注检验。",
    }

    # 把「结论还不完整」的地方显式列出来，避免未核验的 0 被当成实验结果引用
    caveats: list[str] = []
    if m["layer1_existence"]["unverified"]:
        caveats.append(
            f"有 {m['layer1_existence']['unverified']} 个引用未完成 arXiv 核验（限流或未开远程），"
            "存在性层面的结论尚不完整；arXiv 恢复后用 --refresh-layer1 补核，"
            "在此之前不得把 fabricated=0 当作「模型没有编造引用」的证据。")
    if m["judge"]["errors"]:
        caveats.append(f"有 {m['judge']['errors']} 条判官调用失败，相关题目未计入有效判定。")
    if extra.get("answers_failed"):
        caveats.append(f"有 {extra['answers_failed']} 条回答生成失败，未计入分母。")
    m["caveats"] = caveats

    m.update(extra.get("extra") or {})
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="评测运行器")
    ap.add_argument("--answers", required=True, help="答题器输出的 answers.jsonl")
    ap.add_argument("--out", default=None, help="输出目录，默认与 answers 同目录")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="只判前 N 条（冒烟测试用，避免一上来就花 400 次判官调用）")
    ap.add_argument("--layer1-only", action="store_true", help="只跑第一层，不调判官")
    ap.add_argument("--no-remote", action="store_true", help="不查 arXiv")
    ap.add_argument("--force", action="store_true", help="忽略已有 judged.jsonl，全量重判")
    ap.add_argument("--refresh-layer1", action="store_true",
                    help="只重算第一层（存在性），沿用已缓存的判官结论。"
                         "用在 arXiv 恢复后补核 —— 不必重跑 400 次判官调用")
    args = ap.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.exists():
        print(f"找不到回答文件 {answers_path}")
        return 1
    out_dir = Path(args.out) if args.out else answers_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    judged_path = out_dir / "judged.jsonl"

    questions = load_questions()
    answers = {r["qid"]: r for r in load_jsonl(answers_path) if r.get("qid")}
    failed = [q for q, r in answers.items() if r.get("error")]
    print(f"题目 {len(questions)} 条 | 已有回答 {len(answers)} 条（其中失败 {len(failed)} 条）")

    cached_all = {} if args.force else {e["qid"]: e for e in load_jsonl(judged_path)}
    if args.layer1_only:
        cached = cached_all
    else:
        # 只跑过第一层的记录（score 为空）不能当已完成，否则指标会算在一堆空分数上
        cached = {q: e for q, e in cached_all.items() if e.get("score")}
        if len(cached_all) - len(cached):
            print(f"忽略 {len(cached_all) - len(cached)} 条只跑了第一层的记录，本次重跑判官")
    todo = [q for q in answers if q not in cached]
    if args.limit:
        todo = todo[:args.limit]
    print(f"已判 {len(cached)} 条 | 本次待判 {len(todo)} 条")

    corpus = Corpus.load()
    checker = ExistenceChecker(corpus=corpus, remote=not args.no_remote)

    citations_map = {q: extract_citations(answers[q].get("answer") or "", corpus) for q in todo}
    if todo:
        checker.prefetch([c for v in citations_map.values() for c in v])

    client = None if args.layer1_only else LLMClient(temperature=JUDGE_TEMPERATURE)

    def work(qid: str) -> dict:
        rec = questions[qid]
        cits = citations_map[qid]
        l1 = checker.check_all(cits)
        if args.layer1_only:
            judge = {"skipped": "layer1-only 模式未调判官"}
        else:
            judge = judge_record(rec, answers[qid].get("answer") or "",
                                 [c.to_dict() for c in cits], client, corpus)
        return {
            "qid": qid,
            "type": rec["type"],
            "question_form": rec["question_form"],
            "difficulty": rec["difficulty"],
            "domain": rec["domain"],
            "question": rec["question"],
            "answer": answers[qid].get("answer") or "",
            "citations": l1["citations"],
            "existence": {k: v for k, v in l1.items() if k != "citations"},
            "judge": judge,
            "score": ({} if args.layer1_only else score(rec, l1, judge)),
            "elapsed": answers[qid].get("elapsed"),
            "judged_at": datetime.now().isoformat(timespec="seconds"),
        }

    t0 = time.time()
    new_entries: list[dict] = []
    if todo:
        print(f"开始判定 {len(todo)} 条 | 并发 {args.workers}"
              f" | 判官 {'关（layer1-only）' if args.layer1_only else '开'}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool, \
                judged_path.open("a", encoding="utf-8", newline="\n") as f:
            futures = {pool.submit(work, q): q for q in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                e = fut.result()
                new_entries.append(e)
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  进度 {i}/{len(todo)}（{time.time() - t0:.0f}s）")

    entries = list(cached.values()) + new_entries

    if args.refresh_layer1 and cached:
        # arXiv 限流/不通时第一层会大面积留成 unverified。这里只补核这一层，
        # 判官结论原样沿用 —— 否则每补一次核都要重花 400 次判官调用。
        print(f"\n[补核第一层] 重算 {len(entries)} 条的存在性判定（判官结论沿用）")
        # 引用从回答原文重新抽取，而不是复用文件里存的 —— 这样抽取器的改进能直接生效，
        # 否则改了抽取规则还得把 400 次判官调用再花一遍
        cit_map = {e["qid"]: extract_citations(e.get("answer") or "", corpus) for e in entries}
        checker.prefetch([c for v in cit_map.values() for c in v])
        for e in entries:
            l1 = checker.check_all(cit_map[e["qid"]])
            e["citations"] = l1["citations"]
            e["existence"] = {k: v for k, v in l1.items() if k != "citations"}
            if e.get("score"):
                judge = e.get("judge") or {}
                if e["qid"] in questions:
                    e["score"] = score(questions[e["qid"]], l1, judge)
        judged_path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
            encoding="utf-8")
        unver = sum(1 for e in entries for c in e["citations"] if c["status"] == "unverified")
        print(f"  补核完成，仍有 {unver} 个引用未核验")

    entries.sort(key=lambda e: e["qid"])
    (out_dir / "per_question.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")

    cfg = {}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cfg = {}

    metrics = build_metrics(entries, questions, {
        "answers_file": rel(answers_path),
        "system": cfg.get("system"),
        "extra": {
            "model": cfg.get("model"),
            "temperature": cfg.get("temperature"),
            "system_prompt_sha256": cfg.get("system_prompt_sha256"),
            "answers_failed": len(failed),
            "note_total": "指标基于已判定条目计算；回答失败或判定失败的条目不计入相应分母。",
        },
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print("\n===== 指标 =====")
    if metrics.get("layer1_only"):
        print("  （只跑了第一层，无准确率/幻觉率指标）")
        print(f"  第一层: 查无此文 {metrics['layer1_existence']['fabricated']} 条 | "
              f"状态分布 {metrics['layer1_existence']['by_status']}")
        print(f"  输出目录: {out_dir}")
        return 0
    print(f"  正常题 n={metrics['normal']['n']} | 回答准确率 "
          f"{metrics['normal']['answer_accuracy']:.1%}（宽松 "
          f"{metrics['normal']['answer_accuracy_lenient']:.1%}）| 引用命中率 "
          f"{metrics['normal']['citation_hit_rate']:.1%} | 幻觉率 "
          f"{metrics['normal']['hallucination_rate']:.1%}")
    for t in ("trap_nonexistent", "trap_unsupported", "trap_mismatch"):
        s = metrics["traps"][t]
        print(f"  {t} n={s['n']} | 抗住率 {s['resistance_rate']:.1%} | 被诱导率 "
              f"{s['susceptibility_rate']:.1%}")
    print(f"  整体幻觉率 {metrics['overall']['hallucination_rate']:.1%}")
    print(f"  失败模式: {metrics['overall']['failure_modes']}")
    print(f"  第一层: 查无此文 {metrics['layer1_existence']['fabricated']} 条 | "
          f"未核验 {metrics['layer1_existence']['unverified']} 个引用")
    if client:
        print(" ", client.stats())
    print(f"  输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
