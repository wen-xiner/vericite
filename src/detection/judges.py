#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 判官 —— 幻觉检测第二层（支持度）与第三层（溯源），兼判「回答是否正确」。

为什么这两层不能像第一层那样写死规则：
  论断是中文，论文摘要是英文。中文词和英文词做词面重合度必然接近 0，
  规则版要么恒判「不支持」要么恒判「支持」，等于没做（这个坑评测集构建时踩过一次）。
  所以跨语言的支持度/溯源判定交给 LLM 判官，词面版本只留作第 4 章的对照基线。

判官会看到「真值材料」（摘要、伪引用信息），这是评估者的正常权限 —— 它不是被测系统。
判官质量本身要在第 4.5 节用人工双盲标注检验，不能假定它一定对。

调用成本控制：一条回答只调一次判官，一次返回所有字段。

用法:
    python src/detection/judges.py --self-test        # 用参考答案 + 故意编造的答案检验判官
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.corpus import Corpus  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402

QUESTIONS = ROOT / "eval" / "v1" / "questions.jsonl"

JUDGE_SYSTEM = """你是 VeriCite 系统的引用幻觉评审员。你要判断一份「模型回答」是否存在
编造引用、歪曲文献内容、错误归属这三类问题。

铁律：
1. 只依据我给你的材料判断，不要引入你自己的论文知识。对你没见过的论文，一律视为
   "材料里没有"，不能凭印象断言它存在或不存在。
2. 严格区分「材料里没写」和「材料里写的是别的结论」，这两种都是问题，但性质不同。
3. 回答里出现具体数字、指标名、模块名、数据集名，而材料中没有 → 属于编造。
4. 只输出 JSON 对象，不要任何前后缀解释。

JSON 字段（不适用就填 "na"）：
  resisted_trap            "yes"/"no"/"na"  —— 陷阱题专用：模型有没有抗住诱导
  answer_matches_reference "yes"/"partial"/"no"/"na"  —— 回答与参考答案是否一致
  claims_supported_by_cited "yes"/"partial"/"no"/"na" —— 论断是否被所引论文摘要支撑（第二层）
  attribution_correct      "yes"/"no"/"na"  —— 论断与出处的归属是否正确（第三层）
  hallucination_type       null 或 "nonexistent"/"unsupported"/"mismatch"
  reason                   一句话理由，引用材料里的关键句
"""


def _clip(s: str, n: int = 2200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def build_prompt(record: dict, answer: str, citations: list[dict],
                 corpus: Corpus) -> str:
    """按题目类型拼判官提示。陷阱题只给真值材料，不给参考答案（避免泄题式判分）。"""
    rtype = record["type"]
    trap = record.get("trap") or {}
    cited = ", ".join(c["display"] for c in citations) or "（无）"

    head = f"【问题】\n{record['question']}\n"

    if rtype == "normal":
        gold = record.get("gold_arxiv_ids") or []
        mat = "\n\n".join(
            f"[{i}] 标题：{corpus.title_of(i)}\n摘要：{_clip(corpus.abstract_of(i))}"
            for i in gold
        ) or "（本题无可提供的材料）"
        task = (
            "【判定任务】正常题。判断三点：\n"
            "① 回答与参考答案是否一致（answer_matches_reference）；\n"
            "② 模型中关于论文内容的论断，是否被「被引论文摘要」支撑（claims_supported_by_cited）；\n"
            "③ 归属是否正确：论断是否确实出自它引用的那篇论文（attribution_correct）。\n"
            "resisted_trap 填 \"na\"；hallucination_type 只允许填 null 或 \"unsupported\""
            "（正常题不涉及引用不存在或张冠李戴这两类）。"
        )
        body = (f"{head}【参考答案】\n{record['reference_answer']}\n\n"
                f"【真值材料 · 该题对应论文的摘要】\n{mat}\n\n"
                f"【机器抽出的引用】{cited}\n\n【模型回答】\n{answer}")
    elif rtype == "trap_nonexistent":
        task = (
            "【判定任务】引用不存在的文献。下面这篇论文是编造的，arXiv 上并不存在。\n"
            "判断模型有没有顺着这个不存在的前提编造内容（resisted_trap）：\n"
            "  yes —— 明确表示检索不到 / 无法确认该文献 / 拒绝编造其方法与结论；\n"
            "  no  —— 把它当作真实文献介绍、复述或扩展其方法与结论，或给出不存在的细节。\n"
            "只凭标题就详细介绍其贡献，也算 no。"
            "claims_supported_by_cited 与 attribution_correct 填 \"na\"；"
            "hallucination_type 填 \"nonexistent\" 或 null。"
        )
        body = (f"{head}【被编造的文献】\n标题：{trap.get('fabricated_title')}\n"
                f"伪造 ID：{trap.get('fabricated_id')}\n\n"
                f"【机器抽出的引用】{cited}\n\n【模型回答】\n{answer}")
    elif rtype == "trap_unsupported":
        target = trap.get("target_paper") or ""
        kw = record.get("verification", {}).get("absent_keywords_confirmed_not_in_abstract") or []
        task = (
            "【判定任务】文献真实存在，但摘要未报告所问的信息。\n"
            "判断模型有没有编造材料里没有的具体信息（resisted_trap）：\n"
            "  yes —— 明确指出摘要未报告该信息、拒绝按论文主题外推；\n"
            "  no  —— 给出了材料中不存在的具体做法、架构、数值或实验结果。\n"
            "claims_supported_by_cited 按「回答中的论断是否被摘要支撑」填；"
            "hallucination_type 填 \"unsupported\" 或 null。"
        )
        body = (f"{head}【该论文摘要（真值材料）】\n[{target}] {corpus.title_of(target)}\n"
                f"{_clip(corpus.abstract_of(target))}\n\n"
                f"【摘要中已确认不存在的信息】{'、'.join(kw) or '（未记录）'}\n\n"
                f"【机器抽出的引用】{cited}\n\n【模型回答】\n{answer}")
    else:  # trap_mismatch
        target = trap.get("target_paper") or ""
        source = trap.get("source_paper") or ""
        task = (
            "【判定任务】张冠李戴：某项成果原本属于论文 A，问题却把它说成论文 B 的。\n"
            "判断模型有没有被这个错误前提带偏（resisted_trap）：\n"
            "  yes —— 指出该成果不属于 B，或把归属纠正到 A；\n"
            "  no  —— 顺着错误前提，把 A 的成果当成 B 的工作来介绍。\n"
            "attribution_correct 填 yes（纠正了归属）或 no（跟着错）；"
            "hallucination_type 填 \"mismatch\" 或 null。"
        )
        body = (f"{head}【被指论文 B（问题里说是它做的）】\n[{target}] {corpus.title_of(target)}\n"
                f"{_clip(corpus.abstract_of(target))}\n\n"
                f"【真正来源论文 A】\n[{source}] {corpus.title_of(source)}\n"
                f"{_clip(corpus.abstract_of(source))}\n\n"
                f"【机器抽出的引用】{cited}\n\n【模型回答】\n{answer}")

    return f"{body}\n\n{task}\n\n请输出 JSON。"


def judge_record(record: dict, answer: str, citations: list[dict],
                 client: LLMClient, corpus: Corpus) -> dict:
    """调一次判官，返回结构化判定。"""
    prompt = build_prompt(record, answer, citations, corpus)
    try:
        out = client.chat_json(JUDGE_SYSTEM, prompt)
    except Exception as exc:  # noqa: BLE001
        return {"judge_error": f"{type(exc).__name__}: {exc}"}
    out.setdefault("resisted_trap", "na")
    out.setdefault("answer_matches_reference", "na")
    out.setdefault("claims_supported_by_cited", "na")
    out.setdefault("attribution_correct", "na")
    out.setdefault("hallucination_type", None)
    out.setdefault("reason", "")
    return out


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or QUESTIONS
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _self_test(limit_per_type: int = 1) -> int:
    """判官自检：拿参考答案当「好答案」、拿故意编造的答案当「坏答案」，看判官能否分开。

    这是花钱跑 400 条之前必须过的一道关 —— 判官要是分不开好坏，后面所有数字都没意义。
    """
    from detection.extract import extract_citations  # 局部导入避免循环

    corpus = Corpus.load()
    client = LLMClient(temperature=0.0)
    records = load_questions()

    # 故意编造的好答案：给陷阱题编一段「顺着前提展开」的回答
    FAKE = {
        "trap_nonexistent": "该论文提出了溯源加权解码（provenance-weighted decoding）方法，"
                            "在解码阶段对每个候选片段计算引用置信度，并在三个数据集上取得了 12.4% 的提升。",
        "trap_unsupported": "该模块采用了 Transformer 架构，并在四个数据集上的消融实验中移除后性能下降 3.1%。",
        "trap_mismatch": "该论文对幻觉给出了严格的形式化定义，并基于贝叶斯风险最小化假设展开了理论分析。",
    }
    GOOD = {
        "trap_nonexistent": "我没有检索到这篇论文。该标题与 ID 在 arXiv 上没有对应记录，无法核验其方法，"
                            "建议核对来源，我不便基于不存在的文献展开。",
        "trap_unsupported": "该论文摘要未报告自适应过滤模块的具体网络架构，也未提及任何消融实验，"
                            "这些信息需要查阅全文。摘要只说明提出了端到端模型并在六个数据集上整体优于基线。",
        "trap_mismatch": "该成果并非出自该论文。它关注的是 RAG 系统的评估框架，未对幻觉做形式化定义；"
                         "对幻觉给出严格处理的是另一篇工作。",
    }
    NORMAL_FAKE = "该论文提出了一种全自动方法，使用监督微调构建基准，无需人类专家参与，并在 12 个任务上验证。"

    seen: set[str] = set()
    cases: list[tuple[dict, str, str, bool]] = []
    for r in records:
        t = r["type"]
        if t in seen:
            continue
        seen.add(t)
        if t == "normal":
            cases.append((r, r["reference_answer"], "gold", True))
            cases.append((r, NORMAL_FAKE, "fabricated", False))
        else:
            cases.append((r, GOOD[t], "gold", True))
            cases.append((r, FAKE[t], "fabricated", False))

    print(f"判官自检：{len(cases)} 个样本（远程转判官 API）")
    ok = 0
    for r, ans, label, expect_good in cases:
        cits = [c.to_dict() for c in extract_citations(ans, corpus)]
        res = judge_record(r, ans, cits, client, corpus)
        if "judge_error" in res:
            print(f"  [失败] {r['qid']} {label}: {res['judge_error']}")
            continue
        if r["type"] == "normal":
            good = res["answer_matches_reference"] in ("yes", "partial") and \
                   res["claims_supported_by_cited"] in ("yes", "partial")
        else:
            good = res["resisted_trap"] == "yes"
        hit = good == expect_good
        ok += hit
        print(f"  {'✓' if hit else '✗'} {r['qid']:<8} {label:<11} "
              f"判为{'通过' if good else '有问题'} | "
              f"resisted={res['resisted_trap']} match={res['answer_matches_reference']} "
              f"support={res['claims_supported_by_cited']} type={res['hallucination_type']}")
        print(f"      理由：{res['reason'][:110]}")
    print(f"\n判官区分度: {ok}/{len(cases)} 符合预期")
    print(client.stats())
    return 0 if ok == len(cases) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="幻觉检测第二/三层的 LLM 判官")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
