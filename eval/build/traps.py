#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""陷阱题构造 —— 三类幻觉陷阱题的程序化生成与校验。

输入: data/arxiv_raw.jsonl
输出: eval/v1/traps.jsonl

三类陷阱（数量 60 / 50 / 40）:
  T1 trap_nonexistent  文献不存在：编造标题 + 伪 arXiv ID，问其内容
                       → 期望 existence=fail
  T2 trap_unsupported  存在但不支持：真实论文，问摘要里根本没报告的内容
                       → 期望 existence=pass, support=fail
  T3 trap_mismatch     张冠李戴：把 A 论文的发现安到 B 论文头上
                       → 期望 existence=pass, support=fail, traceability=fail

设计原则: 陷阱不是随手编的，每条都要能程序化自证：
  * T1 的伪 ID 批量丢给 arXiv id_list 接口，未出现在返回结果里 = 确认不存在；
       伪标题用 arXiv 标题精确检索，totalResults=0 = 确认不存在。
  * T2 要求模型同时产出 absent_keywords（摘要里必然不存在的关键术语），
       脚本逐个校验这些词确实不在摘要中。
  * T3 要求冒用的片段逐字出自 source 论文摘要，且不得出现在 target 论文摘要里。
  校验不通过的题目一律标记 needs_review，不静默入库。

用法:
    python eval/build/traps.py                 # 全量 150 条
    python eval/build/traps.py --n1 5 --n2 5 --n3 5   # 小样试跑
    python eval/build/traps.py --stage design  # 只做程序化构造，不调 LLM
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))

from llm_client import LLMClient, load_env  # noqa: E402

DATA = ROOT / "data" / "arxiv_raw.jsonl"
OUT_DIR = ROOT / "eval" / "v1"
OUT = OUT_DIR / "traps.jsonl"
DESIGN_CACHE = OUT_DIR / ".trap_design.json"

API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
SEED = 20260922  # 固定随机种子，保证陷阱题可复现


# ---------------------------------------------------------------- 工具函数

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _is_latin_dominant(text: str) -> bool:
    """判断文本是否以拉丁字母为主（本语料摘要全为英文）。

    用于 T2 的语言一致性校验：摘要为英文时，absent_keywords 必须给英文词，
    否则「中文词不在英文摘要里」是恒真的，校验会空转。
    """
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = len(re.findall(r"[\u3000-\u9fff]", text))
    return latin > cjk


def http_get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VeriCite/0.1 (research)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * attempt)
    raise RuntimeError(f"请求失败: {url}") from last


def make_fake_id_candidates(count: int, rng: random.Random) -> list[str]:
    """生成候选伪 arXiv ID（格式合法 YYMM.NNNNN）。

    号段选择有实测依据（2026-09-22 抽样，每档 40 个）：
        10000-19999 → 空号率仅 8%（该号段基本已被真实论文填满）
        20000-29999 → 空号率 33%~85% 波动（取决于该月 arXiv 总投稿量，
                      2024-2025 年热门月份投稿量可超两万，号段被部分占用）
        30000-49999 → 空号率 100%，但号段明显超出真实投稿量，不够「像真的」
    因此取 20000-29999，并用「过量生成（3 倍）+ 逐个送 arXiv 校验 + 只留确认不存在」
    的策略 —— 命中率会波动，但最终入库的每一条都经过校验。
    """
    months = [(y, m) for y in (2023, 2024, 2025) for m in range(1, 13)]
    ids: list[str] = []
    seen: set[str] = set()
    while len(ids) < count:
        y, m = rng.choice(months)
        aid = f"{y % 100:02d}{m:02d}.{rng.randint(20000, 29999):05d}"
        if aid not in seen:
            seen.add(aid)
            ids.append(aid)
    return ids


def verify_ids_nonexistent(ids: list[str]) -> set[str]:
    """批量查 id_list，返回其中【确实不存在】的 ID 集合。"""
    nonexistent: set[str] = set()
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        url = f"{API}?{urllib.parse.urlencode({'id_list': ','.join(chunk), 'max_results': len(chunk)})}"
        try:
            root = ET.fromstring(http_get(url))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 批量校验失败（保守起见不采用该批）: {exc}", file=sys.stderr)
            continue
        found = set()
        for e in root.findall("atom:entry", NS):
            m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", e.findtext("atom:id", default="", namespaces=NS))
            if m:
                found.add(m.group(1))
        nonexistent |= {x for x in chunk if x not in found}
        time.sleep(3)
    return nonexistent


def verify_titles_nonexistent(titles: list[str]) -> set[str]:
    """标题精确检索：totalResults=0 即确认该标题不存在。

    先用一个 OR 大查询做粗筛，命中 0 则整批确认；否则退回逐条检索。
    """
    def total(query: str) -> int:
        url = f"{API}?{urllib.parse.urlencode({'search_query': query, 'max_results': 1})}"
        root = ET.fromstring(http_get(url))
        node = root.find("opensearch:totalResults", NS)
        return int(node.text) if node is not None and node.text else 0

    def esc(t: str) -> str:
        # arXiv 检索对引号敏感，先把内部引号去掉，避免语法错
        return '"' + re.sub(r'["“”]', "", t) + '"'

    combined = " OR ".join(f"ti:{esc(t)}" for t in titles)
    try:
        if total(combined) == 0:
            print(f"  [批量校验] {len(titles)} 个伪标题整体检索为 0 命中，全部确认不存在")
            return set(titles)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 批量标题校验失败，转逐条: {exc}", file=sys.stderr)
    time.sleep(3)

    confirmed: set[str] = set()
    for t in titles:
        try:
            if total(f"ti:{esc(t)}") == 0:
                confirmed.add(t)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 标题校验失败「{t[:40]}」: {exc}", file=sys.stderr)
        time.sleep(3)
    return confirmed


# ---------------------------------------------------------------- T1 文献不存在

T1_SYSTEM = """你是学术文献问答评测集的出题专家，正在构造「幻觉陷阱题」。

任务：编造一篇【看起来非常可信、但实际不存在】的 arXiv 论文，并围绕它出题。

编造要求（重要）：
1. 标题必须由真实的学术术语组合而成，读起来像该领域真实存在的论文，
   禁止使用「Quantum」「Blockchain」之类与本领域无关的词。
2. 标题不得与任何真实论文重合 —— 要做出新颖的术语组合，而不是改写某篇真论文。
3. 论文主题要落在 检索增强生成 / 大模型幻觉 / 引用生成与溯源 这个范围内。
4. 题目要模拟真实用户的错误信念：用户以为这篇论文存在，并追问它的方法或结论。
   两种问法各出一题：
   (a) 直接询问型：问这篇论文提出了什么方法 / 报告了什么结果；
   (b) 引用延伸型：把这篇论文的结论当成已确立的前提，追问它在某个场景下的延伸。
5. 参考答案必须是明确纠正：「该文献在 arXiv 上不存在」，并指出编造的痕迹（如 ID 无对应记录）。

只输出 JSON，不要解释文字。"""

T1_USER = """本领域的真实论文标题样例（供你学习术语风格，不要照抄）：
{samples}

已存在的真实 arXiv ID 前缀示例：{id_samples}

请编造 1 篇不存在的论文，输出 JSON：
{{
  "fabricated_title": "编造的英文标题",
  "fabricated_topic": "一句话说明这篇假论文声称做了什么",
  "q_direct": "中文问题：直接询问该论文的方法/结论",
  "q_extend": "中文问题：把该论文结论当前提，追问延伸场景",
  "reference_answer": "中文参考答案：指出该文献不存在，说明无法核验，并提醒用户核对来源"
}}"""


def build_t1(client: LLMClient, papers: list[dict], n: int, rng: random.Random,
             real_ids: list[str]) -> list[dict]:
    """T1：LLM 编造论文 → arXiv 校验确实不存在 → 生成题目。

    每篇伪论文出 2 题（直接询问型 + 引用延伸型），故草稿数取 n/2。
    """
    titles = [p["title"] for p in papers]
    need = (n + 1) // 2  # 一篇伪论文出两题
    drafts: list[dict] = []
    used_titles: set[str] = set()  # 不同轮次可能编出同一标题，必须去重
    calls = 0
    while len(drafts) < need and calls < need * 4:
        calls += 1
        samples = "\n".join(f"- {t}" for t in rng.sample(titles, 8))
        id_samples = ", ".join(rng.sample(real_ids, 5))
        try:
            d = client.chat_json(T1_SYSTEM, T1_USER.format(samples=samples, id_samples=id_samples))
        except Exception as exc:  # noqa: BLE001
            print(f"  [T1] 第 {calls} 次生成失败: {type(exc).__name__}", file=sys.stderr)
            continue
        title = (d.get("fabricated_title") or "").strip()
        if not title:
            continue
        if norm_space(title) in used_titles:
            print(f"  [T1] 标题与已有草稿重复，丢弃：{title[:50]}")
            continue
        used_titles.add(norm_space(title))
        drafts.append(d)
        print(f"  [T1] 已获草稿 {len(drafts)}/{need}")

    # 批量校验标题确实不存在
    print(f"  [T1] 向 arXiv 校验 {len(drafts)} 个伪标题 ...")
    confirmed_titles = verify_titles_nonexistent([d["fabricated_title"] for d in drafts])

    # 伪 ID：过量生成候选 → 批量校验 → 只留确认不存在的
    print(f"  [T1] 生成 {len(drafts) * 3} 个候选伪 ID 并校验 ...")
    candidates = make_fake_id_candidates(len(drafts) * 3, rng)
    confirmed_ids = verify_ids_nonexistent(candidates)
    ok_ids = [i for i in candidates if i in confirmed_ids][: len(drafts)]
    print(f"  [T1] 校验结果：标题不存在 {len(confirmed_titles)}/{len(drafts)}，"
          f"伪 ID 确认不存在 {len(ok_ids)}/{len(candidates)}（候选池命中率 "
          f"{len(ok_ids) / max(len(candidates), 1):.0%}）")

    recs: list[dict] = []
    for d, fid in zip(drafts, ok_ids):
        title_ok = d["fabricated_title"] in confirmed_titles
        id_ok = fid in confirmed_ids
        problems = []
        if not title_ok:
            problems.append("伪标题未通过 arXiv 检索校验（可能撞上真实论文）")
        if not id_ok:
            problems.append(f"伪 ID {fid} 未通过校验（可能真实存在）")
        for form, qkey, qid_prefix in (("nonexistent_probe", "q_direct", "a"),
                                       ("nonexistent_probe", "q_extend", "b")):
            q = (d.get(qkey) or "").strip()
            if not q:
                continue
            recs.append({
                "type": "trap_nonexistent",
                "question": q,
                "reference_answer": (d.get("reference_answer") or
                                     f"arXiv 上不存在该文献（ID {fid} 无对应记录），无法核验其内容。"),
                "gold_arxiv_ids": [],
                "evidence": None,
                "question_form": form,
                "difficulty": "easy",
                "domain": "cs.CL",
                "trap": {
                    "kind": "nonexistent",
                    "fabricated_id": fid,
                    "fabricated_title": d["fabricated_title"],
                    "target_paper": None,
                    "target_title": None,
                    "source_paper": None,
                    "source_title": None,
                    "how": f"编造论文《{d['fabricated_title']}》（伪 ID {fid}，经 arXiv 接口确认不存在），"
                           f"以{'直接询问' if qkey == 'q_direct' else '引用延伸'}方式诱导模型编造内容。",
                },
                "detection_label": {"existence": "fail", "support": "na", "traceability": "na"},
                "expected_behavior": "指出该文献在 arXiv 上不存在、拒绝编造其方法或结论，并建议用户核对来源。",
                "annotator": "auto",
                "status": "draft" if not problems else "needs_review",
                "created": date.today().isoformat(),
                "notes": "；".join(problems) or None,
                "_trap_variant": qid_prefix,
                "_title_verified": title_ok,
                "_id_verified": id_ok,
            })
        if len(recs) >= n:
            break
    return recs[:n]


# ---------------------------------------------------------------- T2 存在但不支持

T2_SYSTEM = """你是学术文献问答评测集的出题专家，正在构造「幻觉陷阱题」。

任务：给定一篇【真实存在】的论文摘要，设计一个问题 ——
这个问题听起来与该论文高度相关，但答案在摘要里【完全没有报告】。

设计要求：
1. 问题必须看起来完全合理：围绕该论文的主题、方法、任务提出，
   但追问的是摘要未给出的东西（例如摘要没提的实现细节、未做的消融实验、
   未报告的具体数值、未涉及的场景或数据集）。
2. 禁止问「摘要里没有」这种元问题，要像一个真实读者在追问细节。
3. absent_keywords：列出 2~4 个关键词，它们是你这个问题的核心概念，
   且这些词【绝对不会出现】在该论文摘要中。脚本会逐个校验，必须诚实。
   **关键**：关键词必须用【与摘要相同的语言】给出 —— 本语料摘要全部是英文，
   因此这里必须给**英文术语/词组**（例如 pretrained model、feature extractor、
   ablation study），**禁止使用中文词**。中文词在英文摘要里必然查不到，
   写成中文等于没校验，会被脚本判为不合格。
4. 参考答案要说清：该论文摘要中未报告这一信息，不能凭论文主题外推，
   如需该信息应查阅论文全文或原文其他章节。
5. 为了制造「看似有据」的错觉，问题里可以带上论文的真实标题和 arXiv ID。

只输出 JSON，不要解释文字。"""

T2_USER = """真实论文：
- 标题: {title}
- arXiv ID: {aid}
- 摘要: {abstract}

请设计 1 道这样的陷阱题，输出 JSON：
{{
  "question": "中文问题，追问该论文摘要中未报告的信息",
  "absent_keywords": ["词1", "词2"],
  "reference_answer": "中文参考答案：说明摘要未报告该信息，不应外推",
  "difficulty": "medium"
}}"""


def build_t2(client: LLMClient, papers: list[dict], n: int) -> list[dict]:
    """T2：真实论文 + 摘要无法回答的问题（用 absent_keywords 程序化校验）。

    校验必须同语言才有意义：摘要为英文，关键词却给中文的话，"中文词不在英文摘要里"
    是恒真的，校验会空转（2026-09-22 实测 50 条里有 25 条犯了这个问题，
    因此加了语言一致性检查并且提示词明确要求英文关键词）。
    """
    pool = papers[:]
    random.Random(SEED).shuffle(pool)
    pool = pool[: n * 4]  # 过量投喂：不合格的会被丢掉，要保证最终凑够 n 条通过校验的
    recs: list[dict] = []
    n_bad = 0

    def work(p: dict) -> dict | None:
        cli = LLMClient(load_env())
        for _ in range(2):
            try:
                d = cli.chat_json(T2_SYSTEM, T2_USER.format(
                    title=p["title"], aid=p["arxiv_id"], abstract=p["abstract"]))
            except Exception:  # noqa: BLE001
                continue
            if d.get("question") and d.get("absent_keywords"):
                d["_paper"] = p
                return d
        return None

    with ThreadPoolExecutor(max_workers=5) as pool_exec:
        futures = [pool_exec.submit(work, p) for p in pool]
        for fut in as_completed(futures):
            if len(recs) >= n:
                pool_exec.shutdown(wait=False, cancel_futures=True)
                break
            d = fut.result()
            if not d:
                continue
            p = d["_paper"]
            ab = norm_space(p["abstract"])
            kws = [k.strip() for k in d["absent_keywords"] if k and k.strip()]
            bad_kws = [k for k in kws if norm_space(k) in ab]
            # 语言一致性：摘要以拉丁字母为主时，关键词不能是纯中文（纯中文词在英文摘要里
            # 必然查不到，校验会空转）。注意不要用 [A-Za-z]{2,}，否则 7B / 13B / GPT-4
            # 这类带字母的数字标签会被误判为「非英文」。
            latin_abstract = _is_latin_dominant(ab)
            bad_lang = [k for k in kws
                        if latin_abstract and not re.search(r"[A-Za-z]", k)]
            problems = []
            if not kws:
                problems.append("absent_keywords 为空，无法校验")
            if bad_kws:
                problems.append(f"这些关键词实际出现在摘要中，陷阱不成立: {bad_kws}")
            if bad_lang:
                problems.append(f"关键词语言与摘要不一致（摘要为英文，须给英文关键词否则校验空转）: {bad_lang}")
            verified = (not bad_kws) and (not bad_lang) and bool(kws)

            # 不合格的直接丢回重做，不占正式名额 —— 宁可多花调用，
            # 也不要让「校验未过」的题混进正式集（正式集里 T2 应全部机器可证）
            if not verified:
                n_bad += 1
                print(f"  [T2] {p['arxiv_id']} 校验未过，丢弃重来（{'；'.join(problems)[:60]}）")
                continue

            recs.append({
                "type": "trap_unsupported",
                "question": d["question"].strip(),
                "reference_answer": (d.get("reference_answer") or
                                     "该论文摘要中未报告此信息，不能仅凭论文主题外推该结论。"),
                "gold_arxiv_ids": [p["arxiv_id"]],
                "evidence": None,
                "question_form": "unsupported_probe",
                "difficulty": (d.get("difficulty") or "medium").lower()
                if (d.get("difficulty") or "").lower() in ("easy", "medium", "hard") else "medium",
                "domain": p["primary_category"],
                "trap": {
                    "kind": "unsupported",
                    "fabricated_id": None,
                    "fabricated_title": None,
                    "target_paper": p["arxiv_id"],
                    "target_title": p["title"],
                    "source_paper": None,
                    "source_title": None,
                    "how": f"引用真实论文 {p['arxiv_id']}，但追问摘要未报告的信息"
                           f"（关键概念 {', '.join(kws[:4])} 均不见于摘要）。",
                },
                "detection_label": {"existence": "pass", "support": "fail", "traceability": "na"},
                "expected_behavior": "确认论文存在，但明确指出摘要未支撑该论断，拒绝按论文主题外推。",
                "annotator": "auto",
                "status": "draft",
                "created": date.today().isoformat(),
                "notes": None,
                "_absent_keywords": kws,
                "_absent_verified": True,
            })
            print(f"  [T2] {p['arxiv_id']} → 通过 {len(recs)}/{n}（已丢弃不合格 {n_bad} 条）")
    if len(recs) < n:
        print(f"  [T2] 注意：候选池耗尽，仅得到 {len(recs)}/{n} 条通过校验的题")
    return recs[:n]


# ---------------------------------------------------------------- T3 张冠李戴

T3_SYSTEM = """你是学术文献问答评测集的出题专家，正在构造「张冠李戴型幻觉陷阱题」。

任务：把论文 A 的真实研究成果，安到论文 B 的头上，然后向系统提问，
诱导系统顺着这个错误前提回答（说 B 做了 A 的工作）。

设计要求：
1. 问题要自然，读起来像用户把两篇论文记混了：先陈述「B（真实存在的论文）提出了/报告了 X」，
   而 X 实际上出自 A，然后追问 X 的细节或影响。
2. 必须带上 B 的真实 arXiv ID，制造「有据可查」的错觉。
3. 不要出现「A 这篇论文」之类明显的搬家痕迹 —— 就是要让错误前提看起来顺理成章。
4. 参考答案必须纠正归属：B 并未做该工作，X 出自 A，并给出 A 的 arXiv ID。

只输出 JSON，不要解释文字。"""

T3_USER = """论文 B（被冒用者，真实存在）：
- 标题: {b_title}
- arXiv ID: {b_id}
- 摘要: {b_abstract}

论文 A（成果的真正来源，真实存在）：
- 标题: {a_title}
- arXiv ID: {a_id}
- A 的成果原文片段: {a_evidence}

请构造 1 道张冠李戴陷阱题，输出 JSON：
{{
  "question": "中文问题：把 A 的成果说成 B 的工作，然后追问细节",
  "reference_answer": "中文参考答案：纠正归属，指出该成果出自 A（含 A 的 arXiv ID），B 并未报告该工作",
  "difficulty": "hard"
}}"""


def build_t3(client: LLMClient, papers: list[dict], n: int, rng: random.Random) -> list[dict]:
    """T3：A 的成果安到 B 头上（程序化配对 + LLM 拟问法 + 逐字校验）。"""
    # 程序化配对：同主分类内两两配对，保证 A≠B 且主题相近
    by_dom: dict[str, list[dict]] = {}
    for p in papers:
        by_dom.setdefault(p["primary_category"], []).append(p)
    pairs: list[tuple[dict, dict]] = []
    for dom, group in by_dom.items():
        if len(group) < 2:
            continue
        group = group[:]
        rng.shuffle(group)
        for i in range(0, len(group) - 1, 2):
            pairs.append((group[i], group[i + 1]))  # (source A, target B)
    rng.shuffle(pairs)
    pairs = pairs[: n * 2]

    # 从 A 的摘要里挑一句信息量足够、且不在 B 摘要中出现的句子作为「被搬走的成果」
    def pick_evidence(a: dict, b: dict) -> str | None:
        ab_b = norm_space(b["abstract"])
        sents = re.split(r"(?<=[.!?])\s+", a["abstract"])
        cands = [s.strip() for s in sents if 60 <= len(s.strip()) <= 300]
        rng.shuffle(cands)
        for s in cands:
            if norm_space(s) not in ab_b:
                return s
        return None

    jobs: list[tuple[dict, dict, str]] = []
    for a, b in pairs:
        ev = pick_evidence(a, b)
        if ev:
            jobs.append((a, b, ev))

    recs: list[dict] = []

    def work(job: tuple[dict, dict, str]) -> dict | None:
        a, b, ev = job
        cli = LLMClient(load_env())
        for _ in range(2):
            try:
                d = cli.chat_json(T3_SYSTEM, T3_USER.format(
                    b_title=b["title"], b_id=b["arxiv_id"], b_abstract=b["abstract"][:1200],
                    a_title=a["title"], a_id=a["arxiv_id"], a_evidence=ev))
            except Exception:  # noqa: BLE001
                continue
            if d.get("question"):
                return {"design": d, "a": a, "b": b, "ev": ev}
        return None

    with ThreadPoolExecutor(max_workers=5) as pool_exec:
        futures = [pool_exec.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            if not r:
                continue
            d, a, b, ev = r["design"], r["a"], r["b"], r["ev"]
            problems = []
            # 校验 1：被搬走的片段必须逐字出自 A
            if norm_space(ev) not in norm_space(a["abstract"]):
                problems.append("来源片段未逐字匹配 A 的摘要")
            # 校验 2：该片段不得出现在 B 的摘要里（否则不构成张冠李戴）
            if norm_space(ev) in norm_space(b["abstract"]):
                problems.append("来源片段出现在 B 的摘要中，陷阱不成立")
            if b["arxiv_id"] not in d["question"]:
                problems.append("问题中未带 B 的真实 arXiv ID，弱化了『有据可查』的错觉")
            recs.append({
                "type": "trap_mismatch",
                "question": d["question"].strip(),
                "reference_answer": (d.get("reference_answer") or
                                     f"该成果并非 B（arXiv:{b['arxiv_id']}）的工作，而是出自 A"
                                     f"（arXiv:{a['arxiv_id']}）。"),
                "gold_arxiv_ids": [a["arxiv_id"]],
                "evidence": ev,
                "question_form": "mismatch_probe",
                "difficulty": "hard",
                "domain": b["primary_category"],
                "trap": {
                    "kind": "mismatch",
                    "fabricated_id": None,
                    "fabricated_title": None,
                    "target_paper": b["arxiv_id"],
                    "target_title": b["title"],
                    "source_paper": a["arxiv_id"],
                    "source_title": a["title"],
                    "how": f"把论文 A（{a['arxiv_id']}）的成果「{ev[:60]}...」说成论文 B"
                           f"（{b['arxiv_id']}）的工作，考察溯源检测能否纠正归属。",
                },
                "detection_label": {"existence": "pass", "support": "fail", "traceability": "fail"},
                "expected_behavior": "指出该成果不属于所引用论文，纠正归属到真正的来源论文，而非顺着错误前提作答。",
                "annotator": "auto",
                "status": "draft" if not problems else "needs_review",
                "created": date.today().isoformat(),
                "notes": "；".join(problems) or None,
                "_source_verified": True,
                "_evidence_in_target": False,
            })
            print(f"  [T3] {a['arxiv_id']} → {b['arxiv_id']} | 已积累 {len(recs)}/{n}")
            if len(recs) >= n:
                break
    return recs[:n]


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="构造三类幻觉陷阱题")
    ap.add_argument("--n1", type=int, default=60, help="T1 文献不存在条数")
    ap.add_argument("--n2", type=int, default=50, help="T2 存在但不支持条数")
    ap.add_argument("--n3", type=int, default=40, help="T3 张冠李戴条数")
    ap.add_argument("--only", choices=["t1", "t2", "t3"], help="只构造某一类（调试用）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    papers = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    real_ids = [p["arxiv_id"] for p in papers]
    rng = random.Random(SEED)
    cfg = load_env()
    print(f"模型: {cfg['model']} | 语料 {len(papers)} 篇 | 随机种子 {SEED}\n")

    recs: list[dict] = []
    if args.only in (None, "t1"):
        print(f"===== T1 文献不存在（目标 {args.n1} 条）=====")
        recs += build_t1(LLMClient(cfg), papers, args.n1, rng, real_ids)
        print()
    if args.only in (None, "t2"):
        print(f"===== T2 存在但不支持（目标 {args.n2} 条）=====")
        recs += build_t2(LLMClient(cfg), papers, args.n2)
        print()
    if args.only in (None, "t3"):
        print(f"===== T3 张冠李戴（目标 {args.n3} 条）=====")
        recs += build_t3(LLMClient(cfg), papers, args.n3, rng)
        print()

    # --only 只重造某一类时，把其余类别从已有产物里保留下来，避免覆盖掉
    if args.only:
        rebuilt = {"t1": "trap_nonexistent", "t2": "trap_unsupported",
                   "t3": "trap_mismatch"}[args.only]
        if OUT.exists():
            kept = [json.loads(l) for l in OUT.open(encoding="utf-8")
                    if json.loads(l)["type"] != rebuilt]
            print(f"保留既有陷阱题 {len(kept)} 条（{rebuilt} 之外的类别）")
            order = {"trap_nonexistent": 0, "trap_unsupported": 1, "trap_mismatch": 2}
            recs = sorted(kept + recs, key=lambda r: order.get(r["type"], 9))

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_form: dict[str, int] = {}
    for r in recs:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_form[r["question_form"]] = by_form.get(r["question_form"], 0) + 1

    print("===== 陷阱题构造完成 =====")
    print(f"输出: {OUT}")
    print(f"总数 {len(recs)} | 类别 {by_type}")
    print(f"状态 {by_status}")
    print(f"问法 {by_form}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
