#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""嵌入模型封装 —— VeriCite 检索层的编码器。

**为什么必须多语言**
-------------------
评测问句是**中文**的（`eval/v1/questions.jsonl` 400 条，例："HAD 模型将哪些功能
整合到单次推理过程中？"），而语料是**英文** arXiv 论文（143 篇，cs.CL 为主）。
这是典型的跨语言检索（CLIR）：
  * 纯英文模型（`BAAI/bge-small-en-v1.5`）→ 编码不了中文问句；
  * 纯中文模型（`BAAI/bge-small-zh-v1.5`）→ 中文向量与英文文档向量不在同一空间；
所以默认用 `BAAI/bge-m3`（多语言，1024 维，支持中英跨语检索）。

**网络**
-------
本机 `huggingface.co` 直连超时（2026-09-23 实测 HTTP 000），只有 `hf-mirror.com`
可通，故在 import 之前强制设置 `HF_ENDPOINT`。若用户已显式设置该环境变量，则不覆盖。

用法:
    from retrieval.embedder import Embedder
    emb = Embedder()                       # 默认 bge-m3
    vecs = emb.encode(["some text"])       # -> np.ndarray (n, dim)，已 L2 归一化
"""

from __future__ import annotations

import os
from pathlib import Path

# 必须在 import sentence_transformers / huggingface_hub 之前设置，否则会走直连超时。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[2]

# 本地模型目录：权重体积大（bge-m3 约 2.3GB），不入库（.gitignore 已忽略 corpus/*）。
# 下载见 README —— 直连 huggingface.co 不通，必须走 hf-mirror。
LOCAL_MODEL_DIR = ROOT / "corpus" / "models"

DEFAULT_MODEL = "BAAI/bge-m3"

# 摘要中位 1177 字符（约 250 token），512 足够；留出余量给以后的长切片。
DEFAULT_MAX_LEN = 512


class Embedder:
    """延迟加载的嵌入模型。建库与检索是两个进程，故模型只在首次 encode 时载入。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_seq_length: int = DEFAULT_MAX_LEN,
    ) -> None:
        self.model_name = model_name
        self.model_path = resolve_model(model_name)
        self.device = device
        self.max_seq_length = max_seq_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
            )
            self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> "object":
        """编码为 L2 归一化向量（归一化后内积 == 余弦相似度，配合 faiss IndexFlatIP）。"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def encode_query(self, text: str) -> "object":
        """单条查询编码。bge-m3 不强制 query instruction，中英混检直接编码即可。"""
        import numpy as np

        vec = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype="float32")


def model_slug(model_name: str) -> str:
    """`BAAI/bge-m3` -> `bge-m3`，用于索引目录命名。"""
    return model_name.rstrip("/").split("/")[-1]


def resolve_model(model_name: str) -> str:
    """本地 `corpus/models/<slug>/` 已有完整权重就用本地路径，否则回落到 HF 名称。

    回落到 HF 名称时由 sentence-transformers 自己下载（走 `HF_ENDPOINT` 镜像）。
    本机实测 `huggingface.co` 直连超时，所以镜像不是可选项而是必需项。
    """
    local = LOCAL_MODEL_DIR / model_slug(model_name)
    if (local / "config.json").exists():
        return str(local)
    return model_name


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
    e = Embedder(name, max_seq_length=512)
    v = e.encode_query("检索增强生成如何缓解大模型幻觉？")
    print(f"模型 {name} | 维度 {v.shape[1]} | 范数 {float((v ** 2).sum()) ** 0.5:.4f}")
