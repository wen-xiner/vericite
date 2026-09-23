#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""arXiv 接口封装 —— 只用来回答一个问题：这篇文献到底存不存在。

凭据「存在性检测」的第二层依据。第一层是本地语料库（src/common/corpus.py），
不在库里才来这里查。这是整套幻觉检测里唯一 100% 程序化、不依赖 LLM 的判定。

实测要点（2026-09-22 评测集构建时踩过）：
  * 直连 export.arxiv.org 可用，不需要代理。
  * 密集请求返回 HTTP 429，官方要求请求间隔 ≥3 秒。
  * id_list 批量查时，不存在的 ID **不会出现在**返回 entry 里（totalResults 只算真实存在的），
    所以「返回里没出现」＝确认不存在，可以批量查。
  * 标题存在性用 ti:"..." 精确检索，totalResults=0 即不存在。
    多个标题拼成 OR 大查询会因 URL 过长被限流，必须逐条查。

查询结果带本地缓存，同一个 ID / 标题只查一次。
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.textutil import (content_tokens, norm_arxiv_id, norm_space,  # noqa: E402
                             norm_title, token_f1)

API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
DEFAULT_CACHE = ROOT / "experiments" / "cache" / "arxiv_exists.json"


def http_get(url: str, retries: int = 4, timeout: int = 45) -> bytes:
    """带退避重试地取一个 URL。

    arXiv 对密集请求会返回 429，而且 429 时会先挂住十几秒才回响应，
    所以退避要拉长（6 秒那种短退避实测无效，会连续撞回 429）。
    """
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VeriCite/0.1 (research)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            wait = 60 if exc.code == 429 else 15 * attempt
            print(f"    [重试 {attempt}/{retries}] HTTP {exc.code}，{wait}s 后重试")
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 15 * attempt
            print(f"    [重试 {attempt}/{retries}] {type(exc).__name__}: {exc}，{wait}s 后重试")
        time.sleep(wait)
    raise RuntimeError(f"arXiv 请求失败: {url}") from last


class ArxivApi:
    """带缓存的 arXiv 存在性查询。"""

    def __init__(self, cache_path: str | Path | None = None,
                 id_batch: int = 50, id_interval: float = 8.0,
                 title_interval: float = 6.0, batch_pause: float = 30.0,
                 quiet: bool = False):
        self.cache_path = Path(cache_path or DEFAULT_CACHE)
        self.id_batch = id_batch
        self.id_interval = id_interval
        # 实测：3 秒间隔撑不住上百次连查，约 100 次后仍会被 429 打断（2026-09-22）。
        # 标题间隔放宽到 6 秒，并且每 20 条额外歇 batch_pause 秒。
        self.title_interval = title_interval
        self.batch_pause = batch_pause
        self.quiet = quiet
        self.ids: dict[str, bool] = {}
        self.titles: dict[str, bool] = {}
        # 标题精确检索落空后的宽松复核结果：norm_title → {"match": 最相似标题, "f1": 相似度}
        self.title_variant: dict[str, dict] = {}
        self.unverified: dict[str, list[str]] = {"ids": [], "titles": []}
        self.requests = 0
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.ids = dict(data.get("ids") or {})
                self.titles = dict(data.get("titles") or {})
                self.title_variant = dict(data.get("title_variant") or {})
            except Exception:  # noqa: BLE001 —— 缓存坏了就当没有，重新查
                self.ids, self.titles, self.title_variant = {}, {}, {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "updated": date.today().isoformat(),
            "note": "VeriCite 引用存在性核验缓存。ids/titles：true=arXiv 上存在，false=确认不存在。"
                    "title_variant：精确检索落空后，用标题核心词宽松检索得到的最相似真实标题及其词面相似度。",
            "ids": self.ids,
            "titles": self.titles,
            "title_variant": self.title_variant,
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def exists_ids(self, raw_ids: list[str]) -> dict[str, bool]:
        """批量核验 ID。返回 {归一化ID: 是否存在}，查询失败的 ID 不出现在结果里。"""
        todo = sorted({norm_arxiv_id(i) for i in raw_ids if i})
        pending = [i for i in todo if i not in self.ids]
        if pending:
            self._log(f"  arXiv 核验 {len(pending)} 个 ID（已缓存 {len(todo) - len(pending)} 个）")
            for start in range(0, len(pending), self.id_batch):
                chunk = pending[start:start + self.id_batch]
                qs = urllib.parse.urlencode({"id_list": ",".join(chunk),
                                             "max_results": len(chunk)})
                try:
                    root = ET.fromstring(http_get(f"{API}?{qs}"))
                except Exception as exc:  # noqa: BLE001
                    self.unverified["ids"].extend(chunk)
                    self._log(f"    [失败] 本批 {len(chunk)} 个 ID 未核验：{exc}")
                    continue
                found: set[str] = set()
                for e in root.findall("atom:entry", NS):
                    m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", e.findtext("atom:id", "", NS))
                    if m:
                        found.add(m.group(1))
                for i in chunk:
                    self.ids[i] = i in found
                self.requests += 1
                if start + self.id_batch < len(pending):
                    time.sleep(self.id_interval)
            self.save()
        return {i: self.ids[i] for i in todo if i in self.ids}

    def title_hits(self, title: str) -> int | None:
        """标题精确检索，返回命中条数；None 表示未核验（查询失败）。"""
        key = norm_title(title)
        if not key:
            return None
        if key in self.titles:
            return 1 if self.titles[key] else 0
        esc = '"' + re.sub(r'["“”]', "", norm_space(title)) + '"'
        qs = urllib.parse.urlencode({"search_query": f"ti:{esc}", "max_results": 3})
        try:
            root = ET.fromstring(http_get(f"{API}?{qs}", retries=2))
        except Exception as exc:  # noqa: BLE001
            self.unverified["titles"].append(title)
            self._log(f"    [失败] 标题未核验：{title[:60]}（{exc}）")
            return None
        node = root.find("opensearch:totalResults", NS)
        hits = int(node.text) if node is not None and node.text else 0
        self.titles[key] = hits > 0
        self.requests += 1
        self.save()
        time.sleep(self.title_interval)
        return hits

    def stats(self) -> dict:
        return {
            "cached_ids": len(self.ids),
            "cached_titles": len(self.titles),
            "cached_title_variants": len(self.title_variant),
            "requests_this_run": self.requests,
            "unverified_ids": self.unverified["ids"],
            "unverified_titles": self.unverified["titles"],
        }

    def search_titles(self, search_query: str, max_results: int = 5) -> list[str]:
        """按任意检索式取回若干条目标题（取不到就返回空列表，不抛异常）。"""
        qs = urllib.parse.urlencode({"search_query": search_query,
                                     "max_results": max_results})
        try:
            root = ET.fromstring(http_get(f"{API}?{qs}", retries=2))
        except Exception as exc:  # noqa: BLE001
            self._log(f"    [失败] 检索式未执行：{search_query[:60]}（{exc}）")
            return []
        self.requests += 1
        return [norm_space(e.findtext("atom:title", "", NS))
                for e in root.findall("atom:entry", NS)]

    def lookup_variant(self, title: str) -> dict | None:
        """只读缓存版：宽松复核还没做过就返回 None（调用方应据此判 unverified，不要现查）。

        网络请求一律集中在 ExistenceChecker.prefetch 里单线程做，
        判定阶段（可能多线程并发）只读缓存 —— 否则并发判定会一起打 arXiv 并抢写缓存文件。
        """
        return self.title_variant.get(norm_title(title))

    def lenient_title_match(self, title: str, threshold: float = 0.7) -> dict:
        """标题精确检索落空后，判断「是不是同一篇论文、只是标题写法有偏差」。

        动机（2026-09-22 实测）：模型凭记忆写标题时，副标题、连字符、缩写前缀常有出入，
        精确 `ti:"整串"` 会直接 0 命中。若就此判定「编造文献」，会把幻觉率高估一大截。
        这里改用核心词 AND 检索（放宽到"标题包含这几个词"），再用词面 F1 比对返回结果，
        相似度达标即认为「文献真实存在，只是引用标题不精确」。

        实测样例（真的对上了）：
            sam rag self adaptive multimodal retrieval augmented generation
              → Self-adaptive Multimodal Retrieval-Augmented Generation（F1 0.857）
        边界情况：0.70~0.75 区间会混进「同主题的另一篇论文」，如
            dsrag a diverse self retrieval... → Retrieval Augmented Generation Framework for the Nepali Legal...（0.700）
        所以这个区间需要在第 4.6 节逐条人工确认。

        返回 {"match": 最相似标题, "f1": 相似度, "checked": bool, "similar": bool}。
        """
        key = norm_title(title)
        if key in self.title_variant:
            return self.title_variant[key]
        toks = sorted(set(content_tokens(title)), key=len, reverse=True)[:5]
        if len(toks) < 2:
            res = {"match": None, "f1": 0.0, "checked": False}
        else:
            query = " AND ".join(f"ti:{t}" for t in toks)
            best, best_f1 = None, 0.0
            for cand in self.search_titles(query, 5):
                f = token_f1(content_tokens(title), content_tokens(cand))
                if f > best_f1:
                    best, best_f1 = cand, f
            res = {"match": best, "f1": round(best_f1, 3), "checked": True}
        res["similar"] = res["f1"] >= threshold
        self.title_variant[key] = res
        self.save()
        time.sleep(self.title_interval)
        return res


if __name__ == "__main__":
    api = ArxivApi()
    real = api.exists_ids(["2403.09040", "2411.00437"])
    fake = api.exists_ids(["2508.24271"])
    print("真实 ID:", real)
    print("伪 ID:", fake)
    print("统计:", api.stats())
