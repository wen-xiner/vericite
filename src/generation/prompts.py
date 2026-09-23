#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成侧提示词 —— 纯 LLM 与 RAG 共用同一套。

这是第 4.3 对比实验的公平性前提：两个系统的 system prompt 完全一致，
差别只在 user prompt 里有没有注入【检索到的论文材料】。
任何一方单独加「注意不要编造」之类的提示，对比就不成立了。

两种口径（第 4.4 消融实验的一个维度：检索 vs 提示约束各贡献多少）：
    neutral（默认，主线用）  只要求「回答具体、给出依据 ID」，不叮嘱「不要编造」。
                             让模型自然表现，幻觉率才是真实水平，RAG 的抑制效果才看得出来。
    strict                   追加诚实性约束（材料里没有就说没有、不确定就说不确定）。
                             用来回答「光靠 prompt 约束能压掉多少幻觉」。

踩过的坑（2026-09-22）：strict 的约束若写成「如果给出的论文材料里没有报告…」，
纯 LLM 手里没有材料，会把「我没有材料」读成「材料未报告」，正常题 93% 直接拒答，
测出来的不是幻觉率而是「被 prompt 逼闭嘴的比例」。所以 strict 的措辞必须场景无关。
"""

from __future__ import annotations

from pathlib import Path

_BASE = """你是 VeriCite 学术文献研读助手，负责帮助用户读懂人工智能领域的学术论文。
请用中文回答。

要求：
1. 回答要具体，落到论文提出的方法、模块、数据集、指标数值或结论上。
2. 凡是依据某篇论文作答，请在文中给出该论文的 arXiv ID 与标题。
"""

# 严重踩坑记录（2026-09-22）：这里原本写成「（形如 2411.00437）」，拿一个真实 ID 当格式示例。
# 后果：模型在 400 题里有 107 题直接照抄这个示例 ID，再各自配一个与题目主题贴合的编造标题。
# 它污染了两条结论：①「模型 24% 的引用指向同一篇无关论文」纯属示例泄漏，不是模型行为；
# ②「ID 层 0 编造」也不成立 —— 模型抄的示例 ID 恰好是语料里的真论文，当然查得到。
# 所以 prompt 里**绝不能出现任何具体的 arXiv ID**，格式要求只用文字描述。

# 口径 B 追加的诚实性约束。措辞刻意避开「给出的材料」这类只有 RAG 才成立的指代。
_STRICT = """3. 回答必须有依据：如果你无法确认某个说法有文献支撑，请明确说明"没有找到依据"。
4. 如果不确定，直接说明不确定，不要用推测填空。
"""


def system_prompt(strict: bool = False) -> str:
    return _BASE + (_STRICT if strict else "")


def build_user(question: str, contexts: list[dict] | None = None,
               max_abstract: int = 2200) -> str:
    """拼 user prompt。contexts 为空＝纯 LLM；给了材料＝RAG。

    contexts 元素: {"arxiv_id": str, "title": str, "abstract": str}
    """
    if not contexts:
        return (
            "请回答下面的问题。\n"
            "如果你依据某篇论文作答，请给出它的 arXiv ID 与标题。\n\n"
            f"【问题】\n{question}"
        )

    blocks = []
    for c in contexts:
        ab = (c.get("abstract") or "").strip()
        if len(ab) > max_abstract:
            ab = ab[:max_abstract] + "…"
        blocks.append(f"[{c['arxiv_id']}] {c.get('title', '')}\n摘要：{ab}")
    material = "\n\n".join(blocks)

    return (
        "以下是从文献库中检索到的论文材料，请据以回答用户的问题。\n"
        "引用时请给出对应的 arXiv ID。\n\n"
        f"【论文材料】\n{material}\n\n"
        f"【问题】\n{question}"
    )


def module_path() -> Path:
    return Path(__file__).resolve()
