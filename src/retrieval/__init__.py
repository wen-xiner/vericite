#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索层 —— 文献知识库的向量化与召回（毕设大纲 3.2 / 3.3）。

模块分工:
    embedder.py     嵌入模型封装（默认 bge-m3，多语言）
    build_index.py  从 data/arxiv_raw.jsonl 建向量索引
    retriever.py    载入索引并做 top-k 召回

索引产物落在 corpus/index/<tag>/，不进版本库（见 .gitignore）。
"""
