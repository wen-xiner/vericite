#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""arXiv 语料拉取 —— VeriCite 毕设 data 链路第一步。

按预设的查询分组从 arXiv API 拉取真实论文元数据（ID / 标题 / 摘要 / 作者 / 日期 / 分类），
去重后输出 data/arxiv_raw.jsonl，原始 Atom XML 缓存到 data/raw/ 以便复现。

用法:
    python src/ingest/fetch_arxiv.py                # 默认目标 150 篇
    python src/ingest/fetch_arxiv.py --target 150
    python src/ingest/fetch_arxiv.py --no-cache     # 不写原始 XML

设计说明:
  * 只用标准库（urllib + xml.etree），避免为了拉数据装依赖。
  * 命题领域锁定在 RAG / LLM / 幻觉 / 引用，与毕设选题一致（cs.CL + cs.IR）。
  * 时间窗 2023-2025：既保证语料足够新，又避免拉到过老的历史论文。
  * arXiv 官方要求请求间隔 ≥3 秒，脚本内置 sleep，勿调小。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "arxiv_raw.jsonl"

API = "https://export.arxiv.org/api/query"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# 时间窗：2023-01-01 ~ 2025-12-31（arXiv submittedDate 语法）
DATE_RANGE = "submittedDate:[202301010000 TO 202512312359]"

# 查询分组：(slug, search_query, 期望条数)
QUERY_PLAN: list[tuple[str, str, int]] = [
    ("rag_cl", 'cat:cs.CL AND abs:"retrieval-augmented generation"', 30),
    ("hallucination_cl", 'cat:cs.CL AND abs:"hallucination"', 30),
    ("citation_gen", 'cat:cs.CL AND abs:"citation" AND abs:"generation"', 20),
    ("rag_ir", 'cat:cs.IR AND abs:"retrieval-augmented"', 20),
    ("factuality", 'cat:cs.CL AND abs:"factuality"', 20),
    ("faithfulness", 'cat:cs.CL AND abs:"faithfulness" AND abs:"large language model"', 15),
    ("longdoc_qa", 'cat:cs.CL AND abs:"long document" AND abs:"question answering"', 15),
]

# 摘要太短的论文信息量不足，出不了好题，直接丢弃
MIN_ABSTRACT_CHARS = 400

ID_RE = re.compile(r"abs/([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?")


def http_get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    """带重试的 GET（arXiv 偶发 503，直接重试比换代理省事）。"""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VeriCite/0.1 (research)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - 网络异常一律重试
            last = exc
            print(f"    [warn] 第 {attempt} 次请求失败: {type(exc).__name__}", file=sys.stderr)
            time.sleep(3 * attempt)
    raise RuntimeError(f"请求失败: {url}") from last


def clean(text: str | None) -> str:
    """arXiv 的标题/摘要里有多余换行和连续空格，压平。"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_entry(entry: ET.Element, slug: str) -> dict | None:
    raw_id = entry.findtext("atom:id", default="", namespaces=NS)
    m = ID_RE.search(raw_id)
    if not m:
        return None

    title = clean(entry.findtext("atom:title", default="", namespaces=NS))
    abstract = clean(entry.findtext("atom:summary", default="", namespaces=NS))
    if not title or len(abstract) < MIN_ABSTRACT_CHARS:
        return None

    # arXiv 会给出撤稿标记，这类论文不能进语料
    comment = clean(entry.findtext("arxiv:comment", default="", namespaces=NS))
    if "withdraw" in comment.lower():
        return None

    authors = [
        clean(a.findtext("atom:name", default="", namespaces=NS))
        for a in entry.findall("atom:author", NS)
    ]
    categories = [c.get("term", "") for c in entry.findall("atom:category", NS)]
    primary = entry.find("arxiv:primary_category", NS)

    aid = m.group(1)
    return {
        "arxiv_id": aid,
        "version": m.group(2) or "v1",
        "title": title,
        "abstract": abstract,
        "authors": [a for a in authors if a],
        "published": (entry.findtext("atom:published", default="", namespaces=NS) or "")[:10],
        "updated": (entry.findtext("atom:updated", default="", namespaces=NS) or "")[:10],
        "primary_category": (primary.get("term", "") if primary is not None else ""),
        "categories": sorted(set(c for c in categories if c)),
        "doi": clean(entry.findtext("arxiv:doi", default="", namespaces=NS)) or None,
        "journal_ref": clean(entry.findtext("arxiv:journal_ref", default="", namespaces=NS)) or None,
        "comment": comment or None,
        "abs_url": f"https://arxiv.org/abs/{aid}",
        "pdf_url": f"https://arxiv.org/pdf/{aid}",
        "abs_len": len(abstract),
        "source_query": slug,
    }


def fetch_query(slug: str, query: str, want: int, cache: bool) -> list[dict]:
    full_query = f"({query}) AND {DATE_RANGE}"
    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": want,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{API}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"
    print(f"[{slug}] 拉取 {want} 篇 ...")
    payload = http_get(url)

    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / f"{slug}.xml").write_bytes(payload)

    root = ET.fromstring(payload)
    out: list[dict] = []
    for entry in root.findall("atom:entry", NS):
        rec = parse_entry(entry, slug)
        if rec:
            out.append(rec)
    print(f"[{slug}] 有效 {len(out)} 篇")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="拉取 arXiv 语料（VeriCite）")
    ap.add_argument("--target", type=int, default=150, help="目标去重后论文数，默认 150")
    ap.add_argument("--no-cache", action="store_true", help="不写原始 XML 缓存")
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    for slug, query, want in QUERY_PLAN:
        try:
            records = fetch_query(slug, query, want, cache=not args.no_cache)
        except Exception as exc:  # noqa: BLE001 - 单个查询失败不该毁掉整轮
            print(f"[{slug}] 跳过：{exc}", file=sys.stderr)
            continue
        for rec in records:
            seen.setdefault(rec["arxiv_id"], rec)
        print(f"[累计] 去重后 {len(seen)} 篇")
        if len(seen) >= args.target:
            print("[停止] 已达目标数量")
            break
        time.sleep(3)  # arXiv 官方要求的请求间隔

    papers = list(seen.values())[: args.target]
    papers.sort(key=lambda r: r["arxiv_id"], reverse=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for rec in papers:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    by_year: dict[str, int] = {}
    for rec in papers:
        by_cat[rec["primary_category"]] = by_cat.get(rec["primary_category"], 0) + 1
        by_year[rec["published"][:4]] = by_year.get(rec["published"][:4], 0) + 1

    print("\n===== 拉取完成 =====")
    print(f"输出: {OUT_PATH}")
    print(f"论文总数: {len(papers)}")
    print(f"主分类分布: {dict(sorted(by_cat.items(), key=lambda x: -x[1]))}")
    print(f"年份分布: {dict(sorted(by_year.items()))}")
    print(f"平均摘要长度: {sum(r['abs_len'] for r in papers) // max(len(papers), 1)} 字符")
    print(f"抓取时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
