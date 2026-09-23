#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""语料库访问 —— 载入 data/arxiv_raw.jsonl，支持按 ID / 按标题查找。

这是「引用存在性检测」的第一层依据：只要被引文献就在我们自己的知识库里，
它就一定真实存在，不需要任何外部查询。不在库里的才去 arXiv 核验
（见 src/common/arxiv_api.py 与 src/detection/existence.py）。

用法:
    from common.corpus import Corpus
    c = Corpus.load()
    c.get("2409.01864v2")        # 带版本号也能查到
    c.find_title(模型回答原文)     # 回答里照抄了语料标题就能认出来
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.textutil import norm_arxiv_id, norm_space, norm_title  # noqa: E402

DEFAULT_PATH = ROOT / "data" / "arxiv_raw.jsonl"


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    published: str
    pdf_url: str
    authors: str


class Corpus:
    def __init__(self, papers: list[Paper]):
        self.papers = list(papers)
        self._by_id = {p.arxiv_id: p for p in self.papers}
        # 归一化标题 → arxiv_id。同一归一化标题只留第一篇，避免重复计数。
        self._by_title: dict[str, str] = {}
        for p in self.papers:
            t = norm_title(p.title)
            if t and t not in self._by_title:
                self._by_title[t] = p.arxiv_id

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Corpus":
        path = Path(path or DEFAULT_PATH)
        if not path.exists():
            raise FileNotFoundError(f"未找到语料 {path}，请先运行 src/ingest/fetch_arxiv.py")
        papers: list[Paper] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                papers.append(Paper(
                    arxiv_id=norm_arxiv_id(r["arxiv_id"]),
                    title=norm_space(r.get("title", "")),
                    abstract=norm_space(r.get("abstract", "")),
                    primary_category=r.get("primary_category", ""),
                    published=(r.get("published") or "")[:10],
                    pdf_url=r.get("pdf_url", ""),
                    authors=r.get("authors", ""),
                ))
        return cls(papers)

    def __len__(self) -> int:
        return len(self.papers)

    def __iter__(self):
        return iter(self.papers)

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def get(self, arxiv_id: str) -> Paper | None:
        return self._by_id.get(norm_arxiv_id(arxiv_id))

    def has(self, arxiv_id: str) -> bool:
        return self.get(arxiv_id) is not None

    def find_title(self, text: str) -> list[tuple[str, str]]:
        """在文本里找语料中的标题（要求整串照抄），返回 [(arxiv_id, 归一化标题)]。

        模型引用论文时通常会把标题原样写出来，所以整串子串匹配已经够用；
        不做模糊匹配是刻意的 —— 宁可漏配，也不要把不相干的标题算成命中。
        """
        target = norm_title(text)
        if not target:
            return []
        hits: list[tuple[str, str]] = []
        seen: set[str] = set()
        for t, aid in self._by_title.items():
            if t and t in target and aid not in seen:
                seen.add(aid)
                hits.append((aid, t))
        return hits

    def title_of(self, arxiv_id: str) -> str:
        p = self.get(arxiv_id)
        return p.title if p else ""

    def abstract_of(self, arxiv_id: str) -> str:
        p = self.get(arxiv_id)
        return p.abstract if p else ""


if __name__ == "__main__":
    c = Corpus.load()
    print(f"语料 {len(c)} 篇 | 分类 {len({p.primary_category for p in c})} 种")
    demo = c.find_title("本文提出了一种方法，参见 C$^2$-Cite: Contextual-Aware Citation Generation for Attributed Large Language Models 一文。")
    print("标题匹配自测:", demo)
