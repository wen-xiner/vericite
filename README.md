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
│   ├── arxiv_raw.jsonl      # 论文元数据（ID/标题/摘要/作者/分类）
│   └── raw/                 # arXiv API 原始 Atom XML 缓存
├── corpus/              # 切分后的语料 + 向量库
├── eval/                # 评测集
│   ├── schema.json          # 单条记录的结构定义（字段/枚举/标签期望）
│   ├── build/               # 评测集构建脚本（拉数据 → 出题 → 合并自检）
│   └── v1/                  # 评测集 v1 产物
├── src/
│   ├── common/          # 共用工具（LLM 客户端）
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

## 评测集构建

评测集是全篇实验的地基，结构定义见 `eval/schema.json`。设计要点：

**四类题目（共 400 条）**

| 类别 | 条数 | 考察点 | 期望检测标签（存在性/支持度/溯源） |
|---|---|---|---|
| `normal` 正常题 | 250 | 文献真实、内容支撑、出处对应 | pass / pass / pass |
| `trap_nonexistent` 文献不存在 | 60 | 编造标题 + 伪 arXiv ID，问其内容 | fail / na / na |
| `trap_unsupported` 存在但不支持 | 50 | 真实论文，追问摘要未报告的信息 | pass / fail / na |
| `trap_mismatch` 张冠李戴 | 40 | 把 A 论文的成果安到 B 论文头上 | pass / fail / fail |

**每道题的机器校验留痕**（记录在 `verification` 字段，论文中可据此说明评测集可信度）：

- 正常题：`evidence` 必须是所引论文摘要中**逐字出现**的连续片段，脚本按子串比对；同时检查问题上不含「这篇论文」这类含糊指代
- T1：伪标题经 arXiv 标题精确检索确认 `totalResults=0`；伪 ID 经 arXiv `id_list` 批量查询确认无对应记录
- T2：模型必须给出 `absent_keywords`，脚本逐个确认这些词**确实不在**摘要中。
  注意关键词**必须与摘要同语言（英文）**——用中文词去英文摘要里查「不存在」是恒真的，
  校验会空转（v1 首版曾犯此错，25/50 条无效，已改正并加了语言一致性检查）；
  校验不通过的题直接丢弃重做，不进正式集
- T3：被搬走的片段逐字出自 source 论文摘要，且**不得出现**在 target 论文摘要里

**构建流程（可复现）**

```bash
# 1. 拉取语料（143 篇 cs.CL/cs.IR，2023-2025，RAG/LLM/幻觉相关）
python src/ingest/fetch_arxiv.py --target 150

# 2. 生成正常题候选池（LLM 基于真实摘要出题，带 evidence 逐字校验）
python eval/build/normal.py --workers 6

# 3. 构造三类陷阱题（程序化设计 + 程序化校验）
python eval/build/traps.py --n1 60 --n2 50 --n3 40

# 4. 合并 + 结构自检（字段完整性、标签与类型一致性、引用 ID 是否在语料内等）
python eval/build/merge.py

# 5. 导出人工抽检单（分层抽样约 10%）
python eval/build/sample_for_review.py

# 6. 人工逐条判定后回填（判定必须由人填写，脚本只做回填与统计）
python eval/build/record_review.py
```

随机种子固定为 `20260922`，陷阱题配对与抽检抽样均可复现。LLM 密钥放在项目根目录 `.env`（已 gitignore，模板见 `.env.example`）。

### 人工抽检

自动构造的评测集必须交代人工核验过程，否则"质量可控"没有依据。当前做法：

- **范围**：按类别分层随机抽样 **40 条 / 400 条（10%）**（正常题 25、T1 6、T2 5、T3 4）
- **判定口径**：`通过`（可直接用）/ `修改`（小改后可用）/ `废弃`（不可用）。
  通过率 = 通过条数 / 抽检总条数，「修改」「废弃」均不计入通过
- **判定依据**：`review_sample.txt` 每条都附了论文摘要原文，依据所在行用 `▸` 标出，
  判定四步与不通过典型情形印在抬头
- **留痕**：结论写在每条记录的 `review` 字段，汇总在 `eval/v1/review_result.json`；
  `merge.py` 按题干匹配沿用它，重跑不会冲掉人工判定

> ⚠️ 表述纪律：人工抽检只覆盖抽到的那些题，**未抽到的题目仅通过程序化校验**。
> 论文中不得写成"全部 400 条经过人工复核"，只能写"全量经程序化校验 + 分层抽检 10%"。

## 状态

方向已定稿（2026-09-22）。

- [x] 项目骨架、语料拉取链路
- [x] 评测集 schema 定义
- [x] 评测集 v1 构建（400 条：正常题 250 + 三类陷阱题 150）
- [x] 评测集人工抽检（抽 40 条 / 10%）
- [ ] 开题报告
- [ ] 系统 baseline（纯 LLM → RAG）

