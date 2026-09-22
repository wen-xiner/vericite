# VeriCite

> **VeriCite**（/ˈverɪsaɪt/，Verify + Cite）—— 基于检索增强生成的学术文献研读问答系统与引用幻觉检测方法研究（本科毕业设计）

## 项目简介

针对大语言模型在学术文献问答中易编造引用、歪曲结论的问题，设计基于检索增强生成（RAG）的文献研读问答系统 **VeriCite**，提出可自动化的引用幻觉检测方法，并通过自建评测集验证其相对纯 LLM 基线在回答准确率与引用真实性上的提升。

**幻觉检测三层：**
1. **存在性**：引用文献是否真实存在（arXiv ID/DOI 核验）
2. **支持度**：引用文献内容是否支撑论断（摘要语义比对）
3. **溯源**：论断与出处是否对应（张冠李戴检测）

## 目录结构

```
vericite/
├── data/                # arXiv 原始数据、解析后的文本
├── corpus/              # 切分后的语料 + 向量库
├── eval/                # 评测集（正常题 + 幻觉陷阱题）
├── src/
│   ├── ingest/          # arXiv 拉取 + PDF 解析
│   ├── retrieval/       # 嵌入 + 检索
│   ├── generation/      # LLM 问答 + Prompt 模板
│   └── detection/       # 幻觉检测
└── experiments/         # 实验记录（配置+结果）
```

## 技术栈

- **语言**：Python
- **数据链路**：arXiv API → PDF 解析（GROBID / pdfplumber 回退）→ 文档切分 → SBERT 嵌入入库
- **问答链路**：向量召回（top-k）→ 重排序 → 引用约束 Prompt → LLM 生成 → 幻觉检测
- **LLM**：API 调用（DeepSeek 或同类，OpenAI 兼容接口）
- **嵌入**：sentence-transformers
- **向量库**：FAISS / Chroma / SQLite（待定）
- **Web 演示层**：Flask / FastAPI

## 状态

方向已定稿（2026-09-22），当前处于评测集构建 + 开题报告阶段。
