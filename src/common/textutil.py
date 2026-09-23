#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文本工具 —— 归一化、分词、词面相似度、arXiv ID 识别。只用标准库。

用途：语料标题匹配、引用抽取，以及「支持度 / 溯源」检测的离线词面版本。

重要限制：词面相似度只在**同语言**之间才有意义。
本项目的论断是中文、论文摘要是英文，中文论断算出来的词面相似度必然接近 0，
所以跨语言的支持度判定走 LLM 判官（src/detection/judges.py），
词面版本只用于英文-英文的离线对照（见第 4 章检测方法对比）。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

CJK = "\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef"

STOPWORDS = frozenset("""
a an the of for and or to in on at by with from as is are was were be been being
this that these those it its their there here we our you your they them he she his her
not no nor can could may might must shall should will would do does did done has have had
having than then such which who whom whose what when where why how all any both each few
more most other some only own same so too very just now also using used use based via
toward towards into over under between among during et al fig eq sec
""".split())

ARXIV_NEW = re.compile(r"(?<!\d)(\d{4})\.(\d{4,5})(?!\d)")
ARXIV_OLD = re.compile(r"(?<![a-z0-9])([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?!\d)", re.I)


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def strip_latex(s: str) -> str:
    """去掉标题里的 LaTeX 片段（$^2$、\\alpha 之类），否则标题匹配会失败。"""
    s = re.sub(r"\$[^$]{0,40}\$", " ", s or "")
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    return s


def norm_title(s: str) -> str:
    """标题归一化：去 LaTeX、转小写、只留字母数字与中文 —— 用于标题匹配。

    两边都用同一个函数处理，标点差异就不会导致漏配。
    """
    s = strip_latex(s)
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"[^0-9a-z" + CJK + r"]+", " ", s)
    return norm_space(s)


def latin_tokens(s: str, min_len: int = 2) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) >= min_len]


def content_tokens(s: str) -> list[str]:
    """去停用词后的英文词元。中文文本会得到接近空的结果，这是已知限制。"""
    return [t for t in latin_tokens(s) if t not in STOPWORDS]


def is_latin_dominant(s: str, threshold: float = 0.7) -> bool:
    """文本是否以拉丁字母为主 —— 用来判断词面比较是否适用。"""
    letters = sum(1 for ch in s if ch.isalpha())
    if not letters:
        return False
    latin = sum(1 for ch in s if "a" <= ch.lower() <= "z")
    return latin / letters >= threshold


def has_cjk(s: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", s or "") is not None


def token_f1(a: list[str], b: list[str]) -> float:
    """多重集 F1。"""
    ca, cb = Counter(a), Counter(b)
    if not ca or not cb:
        return 0.0
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    p = overlap / sum(ca.values())
    r = overlap / sum(cb.values())
    return 2 * p * r / (p + r)


def window_f1(claim: str, source: str, window: int | None = None) -> float:
    """论断 vs 长文的最佳窗口 F1 —— 论断通常只对应长文里的一小段。"""
    c, s = content_tokens(claim), content_tokens(source)
    if not c or not s:
        return 0.0
    if window is None:
        window = max(len(c) * 3, 40)
    if len(s) <= window:
        return token_f1(c, s)
    step = max(window // 4, 1)
    best = 0.0
    for i in range(0, len(s) - window + 1, step):
        f = token_f1(c, s[i:i + window])
        if f > best:
            best = f
    return best


def find_arxiv_ids(text: str) -> list[str]:
    """文本里出现的 arXiv ID（归一化去版本号，按出现顺序去重）。"""
    out: list[str] = []
    seen: set[str] = set()
    for m in ARXIV_NEW.finditer(text or ""):
        i = f"{m.group(1)}.{m.group(2)}"
        if i not in seen:
            seen.add(i)
            out.append(i)
    for m in ARXIV_OLD.finditer(text or ""):
        i = m.group(1)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def norm_arxiv_id(raw: str) -> str:
    """归一化 arXiv ID：去版本号与包裹符号。旧式 math.GT/0309136 原样保留。"""
    s = norm_space(raw).strip(".,;:()[]{}<>\"'")
    return re.sub(r"v\d+$", "", s, flags=re.I)
