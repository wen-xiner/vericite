#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""幻觉检测 · 第一层：引用存在性（existence）。

这是三层检测里唯一**完全不依赖 LLM**、可逐字节复现的一层，也是 VeriCite 的
核心卖点：模型编造的文献可以直接被程序判死，不需要人工标注。

判定优先级：
  1. 被引文献在本地语料库里            → in_corpus      pass
  2. 不在库里，但 arXiv 接口确认存在    → exists_outside pass（超出本次知识库，但不是幻觉）
  3. 不在库里，arXiv 也查不到           → fabricated     fail（引用幻觉）
  4. 未开远程核验 / 查询失败            → unverified     不下结论

为什么必须有第 2 档：如果只拿「在不在我们 143 篇语料里」当标准，模型引用一篇真实
但语料外的论文也会被误判成幻觉。只有 arXiv 也没有，才叫编造。

证据强度不同，报告里要分开看：ID 判定是硬证据（arXiv 的 ID 空间确定），
标题判定是软证据（ti: 检索对标题写法敏感，可能把真实论文判成查不到）。

用法:
    python src/detection/existence.py --self-test            # 用评测集自检本层
    python src/detection/existence.py --self-test --limit 20
    python src/detection/existence.py --self-test --no-remote # 只用本地语料
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
from common.corpus import Corpus  # noqa: E402
from common.textutil import norm_title  # noqa: E402
from detection.extract import Citation, extract_citations  # noqa: E402

QUESTIONS = ROOT / "eval" / "v1" / "questions.jsonl"

STATUS_PASS = {"in_corpus": True, "exists_outside": True, "exists_variant": True,
               "fabricated": False}
STATUS_LABEL = {
    "in_corpus": "语料内",
    "exists_outside": "语料外但真实存在",
    "exists_variant": "标题有偏差但文献真实存在",
    "fabricated": "查无此文（幻觉）",
    "unverified": "未核验",
}


class ExistenceChecker:
    def __init__(self, corpus: Corpus | None = None, api: ArxivApi | None = None,
                 remote: bool = True):
        self.corpus = corpus if corpus is not None else Corpus.load()
        self.remote = remote
        self.api = api if api is not None else (ArxivApi() if remote else None)

    def prefetch(self, citations: list[Citation]) -> None:
        """预取：把需要远程核验的引用一次性查完（单线程顺序查，避免并发撞限流）。

        不做这件事的话，每个引用都会单独发一次 arXiv 请求 —— 60 条 T1 就是 60 次请求，
        必然被 429 限流。ID 批量查（每批 50 个）只要 1~2 次请求；标题必须逐条查，
        但可以在并发判定开始前集中做完，之后全部走缓存。
        """
        if not self.remote or self.api is None:
            return
        unknown_ids: list[str] = []
        unknown_titles: list[str] = []
        for c in citations:
            if c.kind == "arxiv_id":
                if self.corpus.get(c.value) is None and c.value not in self.api.ids:
                    unknown_ids.append(c.value)
            elif c.source != "corpus_title":
                key = norm_title(c.display or c.value)
                # 精确检索还没做过 → 要查；精确检索落空但还没做宽松复核 → 也要补做
                if key not in self.api.titles or (
                        self.api.titles.get(key) is False
                        and key not in self.api.title_variant):
                    unknown_titles.append(c.display or c.value)
        if unknown_ids:
            self.api.exists_ids(unknown_ids)
        if unknown_titles:
            self._log(f"  标题核验 {len(unknown_titles)} 个（精确检索 + 落空后的宽松复核）")
        for n, t in enumerate(unknown_titles, 1):
            # 宽松复核也在这里做完（单线程），避免并发判定时一起打 arXiv 和写缓存文件
            if self.api.title_hits(t) == 0:
                self.api.lenient_title_match(t)
            if n % 20 == 0 and n < len(unknown_titles):
                # 持续查询会被 arXiv 按 IP 限流（实测 3 秒间隔撑不住上百次连查，
                # 约 100 次后就被 429 打断）。每 20 条多歇一会儿，慢一点但不会中途失败。
                self._log(f"    已查 {n}/{len(unknown_titles)}，暂停 "
                          f"{self.api.batch_pause:.0f}s 避免触发限流")
                time.sleep(self.api.batch_pause)

    def _log(self, msg: str) -> None:
        if self.api is not None and not self.api.quiet:
            print(msg)

    def check_citation(self, c: Citation) -> dict:
        """判定单个引用。返回 {status, pass, evidence}。"""
        if c.kind == "arxiv_id":
            paper = self.corpus.get(c.value)
            if paper is not None:
                return {"status": "in_corpus", "pass": True,
                        "evidence": f"命中本地语料（{paper.primary_category}）"}
            if not self.remote:
                return {"status": "unverified", "pass": None, "evidence": "未开远程核验"}
            result = self.api.exists_ids([c.value])
            if c.value not in result:
                return {"status": "unverified", "pass": None, "evidence": "arXiv 查询失败"}
            if result[c.value]:
                return {"status": "exists_outside", "pass": True,
                        "evidence": "arXiv 上真实存在，但不在本地语料内"}
            return {"status": "fabricated", "pass": False,
                    "evidence": "arXiv id_list 查询无返回（ID 空间确定，硬证据）"}

        # 标题类引用
        if c.source == "corpus_title":
            return {"status": "in_corpus", "pass": True, "evidence": "与本地语料标题整串一致"}
        if not self.remote:
            return {"status": "unverified", "pass": None, "evidence": "未开远程核验"}
        hits = self.api.title_hits(c.display or c.value)
        if hits is None:
            return {"status": "unverified", "pass": None, "evidence": "arXiv 标题检索失败"}
        if hits > 0:
            return {"status": "exists_outside", "pass": True,
                    "evidence": f"arXiv ti: 检索命中 {hits} 条"}
        # 精确检索 0 命中时不能直接判编造：模型凭记忆写标题常有副标题/连字符/缩写前缀偏差。
        # 宽松复核的结果只从缓存读（网络请求集中在 prefetch 里做）；还没做过就明确标未核验，
        # 绝不能因为"复核没做完"就默认当成编造 —— 那会凭空造出幻觉。
        var = self.api.lookup_variant(c.display or c.value)
        if var is None:
            return {"status": "unverified", "pass": None,
                    "evidence": "标题精确检索 0 命中，宽松复核尚未完成（等 --prewarm 跑完再判定）"}
        if var.get("similar"):
            return {"status": "exists_variant", "pass": True,
                    "evidence": f"宽松检索命中高度相似的真实论文（词面 F1 {var['f1']}）："
                                f"{(var.get('match') or '')[:90]}"}
        return {"status": "fabricated", "pass": False,
                "evidence": f"精确检索 0 命中，核心词宽松检索最相似度仅 {var.get('f1')}"
                            "（软证据，对标题写法敏感）"}

    def check_all(self, citations: list[Citation]) -> dict:
        """判定一组引用，给出本层总判定。"""
        checked = []
        for c in citations:
            r = self.check_citation(c)
            checked.append({**c.to_dict(), **r})
        fabricated = [x for x in checked if x["status"] == "fabricated"]
        unverified = [x for x in checked if x["status"] == "unverified"]
        if fabricated:
            verdict = "fail"
        elif unverified or not checked:
            verdict = "unverified"
        else:
            verdict = "pass"
        return {
            "layer": "existence",
            "verdict": verdict,
            "hallucination": bool(fabricated),
            "citations": checked,
            "fabricated": [x["display"] for x in fabricated],
        }


def _self_test(limit: int | None, remote: bool) -> int:
    """用评测集自检：T1 的伪引用必须被判 fabricated，其它类的真实引用必须 pass。"""
    records = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8") if l.strip()]
    checker = ExistenceChecker(remote=remote)

    cases: list[tuple[str, str, Citation]] = []
    for r in records:
        if r["type"] == "trap_nonexistent":
            if r["trap"].get("fabricated_id"):
                cases.append(("T1伪ID", "fabricated",
                              Citation("arxiv_id", r["trap"]["fabricated_id"],
                                       r["trap"]["fabricated_id"], "explicit_id")))
            if r["trap"].get("fabricated_title"):
                cases.append(("T1伪标题", "fabricated",
                              Citation("title", "", r["trap"]["fabricated_title"], "quoted_title")))
        for aid in r["gold_arxiv_ids"]:
            cases.append(("真实引用", "not_fabricated",
                          Citation("arxiv_id", aid, aid, "explicit_id")))

    if limit:
        # 按类型截断，保证每类都有样本
        buckets: dict[str, list] = {}
        for c in cases:
            buckets.setdefault(c[0], []).append(c)
        cases = [c for v in buckets.values() for c in v[:limit]]

    print(f"自检 {len(cases)} 个引用对象（远程核验: {'开' if remote else '关'}）")
    checker.prefetch([c for _, _, c in cases])
    stats: dict[str, dict[str, int]] = {}
    wrong: list[str] = []
    for label, expect, cit in cases:
        res = checker.check_citation(cit)
        s = stats.setdefault(label, {"n": 0, "ok": 0, "unverified": 0})
        s["n"] += 1
        if res["status"] == "unverified":
            s["unverified"] += 1
            continue
        ok = (res["status"] == "fabricated") if expect == "fabricated" else (res["status"] != "fabricated")
        s["ok"] += 1 if ok else 0
        if not ok:
            wrong.append(f"{label} {cit.display[:70]} → {STATUS_LABEL[res['status']]}")

    print("\n===== 第一层（存在性）自检结果 =====")
    for label, s in sorted(stats.items()):
        judged = s["n"] - s["unverified"]
        rate = f"{s['ok']}/{judged}" if judged else "无可判定样本"
        print(f"  {label}: {rate} 判定正确（未核验 {s['unverified']}）")
    if wrong:
        print("\n判定不符预期:")
        for w in wrong[:15]:
            print("  -", w)
    else:
        print("\n全部符合预期 ✅" if not any(v["unverified"] for v in stats.values())
              else "\n符合预期的样本全部正确 ✅（有未核验样本，见上）")
    print("\n", checker.api.stats() if checker.api else "（未启用远程核验）")
    return 0 if not wrong else 1


def _prewarm(paths: list[str], remote: bool) -> int:
    """预热模式：从已有的逐题判定文件里收集引用，把 arXiv 核验结果灌进缓存。

    为什么单独做一个入口：arXiv 是 IP 级限流，冷却期可达一小时。
    补核过程可能被打断，所以缓存增量保存、可以反复跑，跑几次都行 —— 已查过的直接命中缓存。
    全部查完后，再用 run_eval.py --refresh-layer1 把判定结果刷进指标（那个不花 LLM 钱）。
    """
    checker = ExistenceChecker(remote=remote)
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[跳过] 找不到 {path}")
            continue
        n = 0
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # 从回答原文重新抽取，而不是读文件里存的 citations ——
            # 这样抽取器改进（比如过滤正文噪声）能直接生效，不必重跑判官
            for c in extract_citations(rec.get("answer") or "", checker.corpus):
                key = (c.kind, c.display)
                if key in seen:
                    continue
                seen.add(key)
                citations.append(c)
                n += 1
        print(f"{path}: 收集到 {n} 个去重引用")

    todo_ids = [c.value for c in citations
                if c.kind == "arxiv_id" and checker.corpus.get(c.value) is None]
    todo_titles = [c.display for c in citations
                   if c.kind == "title" and c.source != "corpus_title"]
    print(f"\n待核验：ID {len(todo_ids)} 个（批量查，每批 50）、标题 {len(todo_titles)} 个（逐条查）")
    if todo_titles:
        print(f"标题按 3 秒间隔算，最坏情况约 {len(todo_titles) * 3 // 60} 分钟；"
              f"已缓存的不会重复查，中断了直接再跑一次即可。")

    checker.prefetch(citations)

    st = checker.api.stats() if checker.api else {}
    print("\n===== 预热结果 =====")
    print(f"  缓存 ID {st.get('cached_ids')} 个 | 缓存标题 {st.get('cached_titles')} 个")
    print(f"  本轮请求 {st.get('requests_this_run')} 次")
    if st.get("unverified_ids"):
        print(f"  仍未核验 ID {len(st['unverified_ids'])} 个（下一轮会重试）")
    if st.get("unverified_titles"):
        print(f"  仍未核验标题 {len(st['unverified_titles'])} 个（下一轮会重试）")
    left = len(st.get("unverified_ids") or []) + len(st.get("unverified_titles") or [])
    print("  ✅ 引用核验缓存已就绪，接着跑 run_eval.py --refresh-layer1 刷新指标"
          if left == 0 else f"  ⚠️ 还有 {left} 个未核验，arXiv 可能又限流了，稍后再跑一次本命令")
    print(f"  缓存文件: {checker.api.cache_path if checker.api else '（未启用远程核验）'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="引用存在性检测（幻觉第一层）")
    ap.add_argument("--self-test", action="store_true", help="用评测集自检本层")
    ap.add_argument("--limit", type=int, default=None, help="每类最多取多少条自检")
    ap.add_argument("--no-remote", action="store_true", help="不查 arXiv，只用本地语料")
    ap.add_argument("--prewarm", nargs="+", metavar="JSONL",
                    help="从逐题判定文件里收集引用并预热 arXiv 核验缓存（可反复跑）")
    args = ap.parse_args()

    if args.prewarm:
        return _prewarm(args.prewarm, remote=not args.no_remote)
    if args.self_test:
        return _self_test(args.limit, remote=not args.no_remote)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
