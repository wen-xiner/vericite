#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""引用抽取 —— 从系统回答里抽出「它声称引用了什么」。

存在性检测和溯源检测都以这一步为前提：先知道它引了哪些文献，才能判真假。

抽两类：
  * arxiv_id —— 形如 arXiv:2411.00437，或裸写 2403.09040（含旧式 cs/0701001）
  * title    —— 两种来源
       source=corpus_title  与本地语料标题整串对得上（说明它是照抄了真论文标题）
       source=quoted_title  引号/书名号里的英文串（可能是编造的标题）

注意：**抽到 ≠ 有问题**。模型回答「这篇论文我检索不到」时也会出现标题甚至伪 ID，
是否构成幻觉交给检测层判断（存在性 + 判官），抽取层不下结论。

已知限制：模型若既不加引号、又不照抄语料标题地提到一篇不存在的论文（例如自行改写
标题），本层抓不到。这类漏检会在第 4 章作为检测器局限说明。
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.textutil import (find_arxiv_ids, has_cjk, norm_space,  # noqa: E402
                             norm_title, strip_latex)

OPEN_Q = "\u300a\u300c\u300e\u201c\"'\u2018"
CLOSE_Q = "\u300b\u300d\u300f\u201d\"'\u2019"
QUOTED = re.compile("[" + OPEN_Q + "]([^" + OPEN_Q + CLOSE_Q + "\n]{15,220})[" + CLOSE_Q + "]")

# 引号里的东西不都是论文标题。下面这些是实测抓到的噪声：
#   「** 或 **arXiv:2304.09848,」  —— 模型正文的普通引号，里面是 markdown 星号 + 引用列表
#   「**most** of the documents **mention** **risk X**」 —— 模型在复述指令
# 这类串送去 arXiv 查必然查不到，会被判成「编造」——凭空造出幻觉。
# 所以宁可漏，也不能把正文噪声当标题。
BAD_TOKENS = ("**", "*", "`", "|", "\\", "：", "->")


def looks_like_title(raw: str) -> bool:
    """粗筛：这一串是否像一篇英文论文标题。"""
    if not raw or has_cjk(raw):
        return False
    if any(t in raw for t in BAD_TOKENS):
        return False
    if find_arxiv_ids(raw):          # 里面带 arXiv ID，多半是在列举引用而不是标题
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+", raw)
    return len(words) >= 4


@dataclass
class Citation:
    kind: str      # 'arxiv_id' | 'title'
    value: str     # ID 类=归一化后的 arXiv ID；标题类=norm_title(标题)（corpus_title 时存的是 arXiv ID）
    display: str   # 原文形态，报告里给人看
    source: str    # 'explicit_id' | 'corpus_title' | 'quoted_title'

    def to_dict(self) -> dict:
        return asdict(self)


def cited_ids(citations: list[Citation]) -> set[str]:
    """从引用列表里取出能确定的 arXiv ID（显式写出的 + 照抄语料标题换回来的）。"""
    ids: set[str] = set()
    for c in citations:
        if c.kind == "arxiv_id":
            ids.add(c.value)
        elif c.source == "corpus_title":
            ids.add(c.value)
    return ids


def extract_citations(text: str, corpus=None) -> list[Citation]:
    """抽出回答里所有可识别的引用。corpus 传入时会额外识别「照抄语料标题」的情况。"""
    text = text or ""
    out: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, value: str, display: str, source: str) -> None:
        key = (kind, value)
        if not value or key in seen:
            return
        seen.add(key)
        out.append(Citation(kind, value, display, source))

    for i in find_arxiv_ids(text):
        add("arxiv_id", i, i, "explicit_id")

    if corpus is not None:
        for aid, _ in corpus.find_title(text):
            add("title", aid, norm_space(corpus.title_of(aid))[:160], "corpus_title")

    for m in QUOTED.finditer(text):
        raw = norm_space(strip_latex(m.group(1)))
        if not looks_like_title(raw):
            continue
        add("title", norm_title(raw), raw, "quoted_title")

    return out


def extract_from_records(records, text_key: str = "answer", corpus=None) -> None:
    """给一批记录原地打上 citations 字段（供离线批处理用）。"""
    for r in records:
        r["citations"] = [c.to_dict() for c in extract_citations(r.get(text_key) or "", corpus)]


if __name__ == "__main__":
    demo = (
        "该研究由 RAGGED（arXiv:2403.09040）提出，另见《Citation-Aware Retrieval "
        "Augmentation with Provenance-Weighted Decoding for Faithful Attribution》。"
    )
    for c in extract_citations(demo):
        print(c)
