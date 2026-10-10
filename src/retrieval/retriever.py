#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""向量召回 —— 毕设大纲 3.3「向量召回（top-k）」的第一段。

载入 `corpus/index/<tag>/` 的 faiss 索引与元数据，把查询编码后做 top-k 召回。
本模块只管**召回**，不做重排序、不做生成。

用法:
    # 单条查询
    python src/retrieval/retriever.py --query "检索增强生成如何缓解大模型幻觉？" --top-k 5

    # 在评测集上算 Recall@k / MRR（只统计带 gold_arxiv_ids 的题）
    python src/retrieval/retriever.py --eval --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.textutil import norm_arxiv_id  # noqa: E402

DEFAULT_ROOT = ROOT / "corpus" / "index"
DEFAULT_QUESTIONS = ROOT / "eval" / "v1" / "questions.jsonl"


class Retriever:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        cfg_path = self.index_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"索引目录不完整: {self.index_dir}（缺 config.json）")
        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))

        self.meta: list[dict] = []
        with (self.index_dir / "meta.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))

        import faiss

        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        if self.index.ntotal != len(self.meta):
            raise ValueError(
                f"索引与元数据条数不符: {self.index.ntotal} vs {len(self.meta)}"
            )
        self._emb = None

    @classmethod
    def load(cls, tag: str | None = None, root: Path = DEFAULT_ROOT) -> "Retriever":
        root = Path(root)
        if tag is None:
            cands = sorted(p for p in root.iterdir() if (p / "config.json").exists()) \
                if root.exists() else []
            if not cands:
                raise FileNotFoundError(f"{root} 下没有索引，先跑 src/retrieval/build_index.py")
            if len(cands) > 1:
                names = ", ".join(p.name for p in cands)
                raise ValueError(f"存在多个索引，请用 --tag 指定：{names}")
            return cls(cands[0])
        return cls(root / tag)

    @property
    def embedder(self):
        if self._emb is None:
            from retrieval.embedder import Embedder

            self._emb = Embedder(
                self.config["model"],
                max_seq_length=self.config.get("max_seq_length", 512),
            )
        return self._emb

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.search_batch([query], top_k)[0]

    def search_batch(self, queries: list[str], top_k: int = 5) -> list[list[dict]]:
        import numpy as np

        vecs = np.asarray(self.embedder.encode(queries, show_progress=False), dtype="float32")
        scores, idxs = self.index.search(vecs, top_k)

        out: list[list[dict]] = []
        for row_scores, row_idx in zip(scores, idxs):
            hits: list[dict] = []
            for rank, (s, i) in enumerate(zip(row_scores, row_idx), start=1):
                if i < 0:
                    continue
                m = self.meta[i]
                hits.append({
                    "rank": rank,
                    "score": float(s),
                    "arxiv_id": m["arxiv_id"],
                    "title": m["title"],
                    "primary_category": m.get("primary_category", ""),
                })
            out.append(hits)
        return out


def evaluate(retriever: Retriever, questions_path: Path, top_k: int) -> dict:
    qs: list[dict] = []
    with questions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))

    usable = [q for q in qs if q.get("gold_arxiv_ids")]
    print(f"评测题 {len(qs)} 条 | 带 gold_arxiv_ids 的 {len(usable)} 条（其余为伪文献陷阱题，无金标来源）")

    queries = [q["question"] for q in usable]
    all_hits = retriever.search_batch(queries, top_k)

    k_values = [1, 3, 5, 10]
    k_values = [k for k in k_values if k <= top_k]
    counts = {k: 0 for k in k_values}
    rr_sum = 0.0
    by_type: dict[str, list[int]] = {}
    misses: list[dict] = []

    for q, hits in zip(usable, all_hits):
        golds = {norm_arxiv_id(g) for g in q["gold_arxiv_ids"]}
        hit_ranks = [h["rank"] for h in hits if h["arxiv_id"] in golds]
        first = min(hit_ranks) if hit_ranks else 0
        rr_sum += (1.0 / first) if first else 0.0
        if not first:
            misses.append({
                "qid": q.get("qid"),
                "type": q.get("type"),
                "gold": sorted(golds),
                "top1": hits[0]["arxiv_id"] if hits else None,
            })
        t = q.get("type", "?")
        by_type.setdefault(t, [0, 0])
        by_type[t][1] += 1
        if first:
            by_type[t][0] += 1
        for k in k_values:
            if first and first <= k:
                counts[k] += 1

    n = max(len(usable), 1)
    result = {
        "n_questions": len(usable),
        # 溯源：这一组指标对应哪份索引、哪版语料。语料 sha256 必须留，
        # 否则换过语料后指标说不清是谁的（项目纪律：代码/数据/指标要自洽）。
        "index_tag": retriever.index_dir.name,
        "model": retriever.config["model"],
        "doc_mode": retriever.config["doc_mode"],
        "n_docs": retriever.config["n_docs"],
        "dim": retriever.config.get("dim"),
        "top_k": top_k,
        "source_sha256": retriever.config.get("source_sha256"),
        "index_built_at": retriever.config.get("built_at"),
        "recall_at": {k: round(counts[k] / n, 4) for k in k_values},
        "mrr": round(rr_sum / n, 4),
        "by_type": {
            t: {"hit": v[0], "total": v[1], "rate": round(v[0] / max(v[1], 1), 4)}
            for t, v in sorted(by_type.items())
        },
        # 一条都没召回的题（逐条留痕，便于人工看是不是检索真错还是金标问题）
        "misses": misses,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="VeriCite 向量召回")
    ap.add_argument("--tag", default=None, help="索引目录名（默认取唯一一个）")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--query", default=None, help="单条查询文本")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--eval", action="store_true", help="在评测集上算 Recall@k / MRR")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--json-out", type=Path, default=None, help="把评估结果写 JSON")
    args = ap.parse_args()

    r = Retriever.load(args.tag, args.root)
    print(f"索引: {r.index_dir.name} | 模型 {r.config['model']} | "
          f"doc_mode={r.config['doc_mode']} | {r.config['n_docs']} 篇 | "
          f"dim={r.config['dim']} | 语料 sha {r.config['source_sha256'][:12]}…")

    if args.eval:
        res = evaluate(r, args.questions, args.top_k)
        print("\n===== 召回评估 =====")
        for k, v in res["recall_at"].items():
            print(f"  Recall@{k:<3} {v:.1%}")
        print(f"  MRR       {res['mrr']:.4f}")
        print("  按题型:")
        for t, v in res["by_type"].items():
            print(f"    {t:<18} {v['hit']}/{v['total']}  {v['rate']:.1%}")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  结果已写 {args.json_out}")
        return 0

    if not args.query:
        ap.error("需要 --query 或 --eval 之一")
    hits = r.search(args.query, args.top_k)
    print(f"\n查询: {args.query}")
    for h in hits:
        print(f"  [{h['rank']}] {h['score']:.4f}  {h['arxiv_id']}  {h['title'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
