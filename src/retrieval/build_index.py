#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""知识库向量索引构建 —— 毕设大纲 3.2（arXiv 拉取 / PDF 解析 / 切分向量化）的最后一环。

**当前档位：摘要级（document-level）**
--------------------------------------
一个文档单元 = 一篇论文，向量取自「标题 + 摘要」。不依赖 PDF 解析，先打通
「向量召回 → 引用约束生成 → 三层幻觉检测」的端到端闭环。

⚠️ 已知装置局限（论文里必须写明）
--------------------------------
`eval/v1/questions.jsonl` 的 400 条题里，250 道正常题的证据**逐字取自摘要**
（`verification.evidence_verbatim_in_abstract` 全为 True，2026-09-23 全量核实）。
出题粒度 = 摘要，若知识库粒度也用摘要，**检索层会偏容易**，Recall@k 天然偏高，
不能用来论证"检索策略优劣"。全文级（PDF 解析 + 切片）留作后续对照实验。

用法:
    python src/retrieval/build_index.py                          # 默认 bge-m3 + 标题+摘要
    python src/retrieval/build_index.py --doc-mode abstract      # 只摘要（消融用）
    python src/retrieval/build_index.py --tag my-tag             # 自定义产物目录名
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.textutil import norm_arxiv_id, norm_space  # noqa: E402
from retrieval.embedder import DEFAULT_MODEL, Embedder, model_slug  # noqa: E402

DEFAULT_SOURCE = ROOT / "data" / "arxiv_raw.jsonl"
DEFAULT_OUT_ROOT = ROOT / "corpus" / "index"

# 文档文本构造方式 → 消融实验的一维（大纲 4.4「切分粒度」在摘要级下的对应物）
DOC_MODES = ("title_abstract", "abstract")


def build_doc_text(rec: dict, mode: str) -> str:
    title = norm_space(rec.get("title", ""))
    abstract = norm_space(rec.get("abstract", ""))
    if mode == "abstract":
        return abstract
    if mode == "title_abstract":
        return f"{title}\n{abstract}"
    raise ValueError(f"未知 doc-mode: {mode}")


def load_source(path: Path) -> list[dict]:
    recs: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"语料为空: {path}")
    return recs


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 VeriCite 摘要级向量索引")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="语料 jsonl 路径")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="索引根目录")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="嵌入模型（须支持中英跨语）")
    ap.add_argument("--doc-mode", choices=DOC_MODES, default="title_abstract")
    ap.add_argument("--tag", default=None, help="产物子目录名，默认 <model>-<doc-mode>")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=512)
    args = ap.parse_args()

    import faiss  # 延迟导入：--help / 语法检查不该依赖装好 faiss
    import numpy as np

    recs = load_source(args.source)
    texts = [build_doc_text(r, args.doc_mode) for r in recs]
    print(f"语料 {len(recs)} 篇 | doc-mode={args.doc_mode} | 模型={args.model}")

    t0 = time.time()
    emb = Embedder(args.model, max_seq_length=args.max_seq_length)
    vecs = np.asarray(emb.encode(texts, batch_size=args.batch_size), dtype="float32")
    n, dim = vecs.shape
    print(f"编码完成：{n} × {dim}（{time.time() - t0:.1f}s）")

    # 归一化向量 + 内积 == 余弦相似度
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    tag = args.tag or f"{model_slug(args.model)}-{args.doc_mode}"
    out_dir = args.out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))

    with (out_dir / "meta.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(recs):
            f.write(json.dumps({
                "idx": i,
                "arxiv_id": norm_arxiv_id(r["arxiv_id"]),
                "title": norm_space(r.get("title", "")),
                "primary_category": r.get("primary_category", ""),
                "published": (r.get("published") or "")[:10],
                "abs_len": len(norm_space(r.get("abstract", ""))),
                "doc_chars": len(texts[i]),
            }, ensure_ascii=False) + "\n")

    config = {
        "tag": tag,
        "model": args.model,
        "dim": dim,
        "doc_mode": args.doc_mode,
        "n_docs": n,
        "max_seq_length": args.max_seq_length,
        "normalize_embeddings": True,
        "index_type": "IndexFlatIP",
        "unit": "paper",
        "source": str(args.source.relative_to(ROOT).as_posix()),
        "source_sha256": sha256_of(args.source),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n===== 索引构建完成 =====")
    print(f"目录: {out_dir}")
    print(f"向量: {n} × {dim} | 类型: IndexFlatIP（归一化后内积 = 余弦）")
    print(f"语料 sha256: {config['source_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
