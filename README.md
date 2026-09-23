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
│   ├── common/          # 共用工具（LLM 客户端 / 语料访问 / 文本工具 / arXiv 接口）
│   ├── ingest/          # arXiv 拉取 + PDF 解析
│   ├── retrieval/       # 嵌入 + 检索
│   ├── generation/      # LLM 问答 + Prompt 模板 + baseline 答题器
│   ├── detection/       # 幻觉检测三层
│   └── experiments/     # 评测运行器（跑指标表）
└── experiments/         # 实验记录（配置+结果+核验缓存）
    ├── baseline_llm/            # 纯 LLM 基线，口径 A（中性 prompt，主线）
    ├── baseline_llm_strict/     # 纯 LLM 基线，口径 B（追加诚实性约束，消融对照）
    └── cache/                   # 引用存在性核验缓存（保证实验可复现）
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
- **本轮结果**：抽检 40 条全部判定「通过」（通过率 100%），分工种 25 / 6 / 5 / 4 全通过；
  判定后正式集状态为 `reviewed 40` + `draft 360`

> ⚠️ 表述纪律：人工抽检只覆盖抽到的那些题，**未抽到的题目仅通过程序化校验**。
> 论文中不得写成"全部 400 条经过人工复核"，只能写"全量经程序化校验 + 分层抽检 10%"。

## 幻觉检测与评测流程

三层检测的实现分工（这是第 3.4 节与第 4.5 节的主体）：

| 层 | 判什么 | 怎么判 | 是否依赖 LLM |
|---|---|---|---|
| 一 · 存在性 | 被引文献是否真实存在 | 本地语料库 → arXiv `id_list`/`ti:` 核验 | **否，纯程序化** |
| 二 · 支持度 | 文献内容是否支撑该论断 | LLM 判官（给论断 + 被引论文摘要） | 是 |
| 三 · 溯源 | 论断与出处是否对应 | LLM 判官（给论断 + 被引论文摘要 + 真正来源摘要） | 是 |

**第一层为什么是多态而不是两态**：只按"在不在本地 143 篇语料里"判，会把"模型引用了一篇真实
但语料外的论文"误判成幻觉。现有五态：
`in_corpus`（语料内）/ `exists_outside`（真实存在但超出本次知识库）/
`exists_variant`（标题写法有偏差，但宽松检索到高度相似的真实论文）/
`fabricated`（arXiv 也查不到＝编造）/ `unverified`（arXiv 不可用，不下结论）。
只有 `fabricated` 才算存在性幻觉。

**`exists_variant` 是为了不把幻觉率高估**：模型凭记忆写标题时，副标题、连字符、缩写前缀常
有出入，精确 `ti:"整串"` 检索会直接 0 命中，若就此判"编造"会冤枉一批真实文献。所以精确检
索落空后追加一次宽松复核（取 5 个最长实词做 `ti:词 AND ti:词` 检索，再算词面 F1 ≥ 0.7 认为
是同一篇）。实测 400 条里命中 10 条，例如
`sam rag self adaptive multimodal retrieval augmented generation`
→ `Self-adaptive Multimodal Retrieval-Augmented Generation`（F1 0.857）。

> ⚠️ 宽松阈值的代价：0.70~0.75 区间会混进"同主题的另一篇论文"（如 `dsrag a diverse self
> retrieval ...` 匹配到 `... for the Nepali Legal Domain`，F1 0.700，是假阳性）。
> 这一区间必须逐条人工确认后才能在论文里引用。
>
> ⚠️ 反向风险：宽松复核仍用"核心词 AND"，真实标题若不含其中某个词就会 0 结果 → F1=0
> → 被判 `fabricated`。抽样二次核验（改用 3 词 AND 与 `all:` 全字段检索）已在 11 条中
> 发现 1 条属误判（`measuring and reducing llm hallucination without gold standards`
> 实为 `Self-Known and Self-Unknown: Measuring and Reducing LLM Hallucination`，F1 0.800）。
> **所以 `fabricated` 的计数里含少量假阳性，论文里报数时必须注明这是软证据。**

**第二、三层为什么必须用 LLM 判官**：论断是中文、摘要是英文，词面重合度必然接近 0，
规则版要么恒判"不支持"要么恒判"支持"，等于没做（与 T2 关键词校验空转是同一类坑）。
词面版本只保留作第 4.5 节的对照基线。

**引用抽取要防正文噪声**：引号里的东西不都是论文标题。实测抓到过
`** 或 **arXiv:2304.09848,`（模型正文的普通引号 + markdown 星号）和
`**most** of the documents **mention** **risk X**`（模型在复述指令）被当成"书名号内容"。
这类串送去 arXiv 必然查不到，会被判成"编造"——**凭空造出幻觉**。
所以抽取层加了粗筛：含中文、含 markdown 标记、内部带 arXiv ID、
英文实词少于 4 个的候选一律丢弃。**宁可漏检，也不能把正文噪声当标题。**

**判官本身也要被检验**：判官是评估者不是被测系统，所以它能看真值材料；但它的准确率必须在
第 4.5 节用人工双盲标注检验，不能假定它一定对。跑 400 条之前先用 `judges.py --self-test`
做区分度自检（参考答案当"好答案"、故意编造的答案当"坏答案"）。

### 跑一次实验

```bash
# 1. 答题（纯 LLM 无检索；--strict 切到口径 B）
python src/generation/baseline_llm.py --workers 6

# 2. 评测：抽引用 → 第一层 → 判官 → 按题型判对 → 汇总指标
python src/experiments/run_eval.py --answers experiments/baseline_llm/answers.jsonl \
    --out experiments/baseline_llm --workers 6

# 只跑第一层不花钱 / arXiv 不通时先跳过远程核验
python src/experiments/run_eval.py --answers ... --layer1-only
python src/experiments/run_eval.py --answers ... --no-remote

# arXiv 恢复后只补核第一层，沿用已缓存的判官结论（不必重跑 400 次判官）
python src/experiments/run_eval.py --answers ... --refresh-layer1
```

**arXiv 限流怎么办**：arXiv 是按 IP 限流的，密集查询后会出现 HTTP 429、甚至一段时间完全超时，
冷却期实测可达一小时（换代理没用——出口 IP 不变）。所以第一层核验拆成两步、可反复跑：

```bash
# 1. 先预热核验缓存：增量保存，中断了直接再跑同一条命令，已查过的不会重复查
python src/detection/existence.py --prewarm experiments/baseline_llm/judged.jsonl \
    experiments/baseline_llm_strict/judged.jsonl

# 2. 缓存齐了再刷指标（这一步不调 LLM）
python src/experiments/run_eval.py --answers ... --refresh-layer1
```

核验结果落在 `experiments/cache/arxiv_exists.json`（入库），
所以**同一批实验的判定结果可复现**，不依赖 arXiv 当时的可用性。

实测的请求节奏（2026-09-22，改小会立刻撞回 429）：

| 参数 | 取值 | 说明 |
|---|---|---|
| `id_batch` | 50 | `id_list` 批量查，一批 50 个 ID 只要 1 次请求 |
| `id_interval` | 8s | 批次之间 |
| `title_interval` | 6s | 标题必须逐条查，3 秒实测撑不住上百次连查 |
| `batch_pause` | 30s | 每查 20 条标题额外歇一次 |

> ⚠️ 真正的限制是**单日累计请求量**，不是单次间隔。当天跑完 `--prewarm`（49 次）
> 加两轮 `--refresh-layer1` 后再做二次核验，即使间隔已经放到 6 秒仍被 429 打断，
> 且冷却要等一小时。所以：**补核要攒着一次做完，不要反复小批量地试。**
> 缓存是增量的、`--prewarm` 可反复跑，中断了直接重跑同一条命令即可，已查过的不重复查。

每次运行都会在输出目录留下 `config.json`（含 system prompt 原文与 sha256）、`answers.jsonl`、
`judged.jsonl`（逐题判定，断点续跑用）、`per_question.jsonl`、`metrics.json`。

### Prompt 口径（第 4.4 节消融维度之一）

纯 LLM 与 RAG 共用同一套 system prompt（`src/generation/prompts.py`），
只在"是否注入检索到的论文材料"上有差别——这是对比公平性的前提。

- **口径 A `neutral`（主线）**：只要求"回答具体、给出依据 ID"，**不叮嘱"不要编造"**。
  让模型自然表现，幻觉率才是真实水平。
- **口径 B `strict`**：追加诚实性约束，用来回答"光靠提示约束能压掉多少幻觉"。

> ⚠️ 踩坑记录：`strict` 的约束若写成「如果**给出的论文材料**里没有报告……」，纯 LLM 手里
> 没有材料，会把"我没有材料"读成"材料未报告"，正常题 93% 直接拒答——测出来的不是幻觉率，
> 而是"被 prompt 逼闭嘴的比例"。所以诚实性约束的措辞必须场景无关。

> 🚨 **最大的坑：system prompt 里不能出现任何具体 arXiv ID。**
> 初版把格式要求写成「给出该论文的 arXiv ID（**形如 2411.00437**）」，拿一个真实 ID 当示例。
> 结果模型在 250 道正常题里有 **107 题**直接照抄这个 ID，各自再配一个与题目主题贴合的
> **编造标题**（同一 ID 在"引用文本生成长度""幻觉分类法""同步机器翻译幻觉"三道无关题里
> 被配了三个完全不同的标题）。它一次污染了两条结论：
> ① "模型 24% 的引用都指向同一篇无关论文" —— 纯属示例泄漏，不是模型行为；
> ② "第一层 ID 核验 0 编造、说明模型不编 ID" —— 也不成立，模型抄的示例 ID 恰好是语料里的
> 真论文，当然查得到。**发现后已把示例改成纯文字描述，两套基线已重跑**
> （污染版归档在 `experiments/_archive_prompt_leak/`，不入库）。
>
> 教训：prompt 里当示例的字符串，一定会出现在输出里。凡是要统计模型行为的东西
> （ID、标题、数值、格式），示例都必须用明显不可能被当成真值的形式。

## 基线结果

纯 LLM 无检索基线，两套口径各 400 题（正常题 250 + 陷阱题 150），模型 `deepseek-chat`、
temperature 0.2、同一套 system prompt（仅"是否追加诚实性约束"不同）。
第一层 `unverified` 已清零，判定结果可由 `experiments/cache/arxiv_exists.json` 复现。

| 指标 | 口径 A `neutral`（主线） | 口径 B `strict` |
|---|---|---|
| system prompt sha256 | `a48c220b3cf123bc` | `ddf553297bea79f6` |
| 正常题回答准确率 | 18.8% | 2.4% |
| 正常题准确率（宽松，含部分一致） | 33.2% | 4.0% |
| 引用命中率 | 19.6% | 5.2% |
| 正常题幻觉率 | 77.6% | 27.2% |
| T1 伪文献 · 抗住率 | 43.3% | 53.3% |
| T2 无支撑 · 抗住率 | 24.0% | 96.0% |
| T3 归属错位 · 抗住率 | 52.5% | 92.5% |
| **整体幻觉率** | **71.2%** | **25.2%** |

### 结论一：模型不编 ID，但会编标题（第一层程序化判定，硬/软证据分开看）

| 引用类型 | 引用总次数 | 查无此文 | 编造率 |
|---|---|---|---|
| arXiv ID（硬证据） | 491 | **0** | **0%** |
| 论文标题（软证据） | 377 | 106 | 28% |

这是本项目最有价值的发现，也直接决定了检测器的设计：**如果只核验 arXiv ID 是否存在，
对模型实际产生的引用幻觉检出率是 0** —— 因为模型给出的 ID 全是真的，编造发生在标题上。
（另有 10 条标题属 `exists_variant`，即写法有偏差但文献真实，见上文宽松复核一节。）

### 结论二：prompt 约束是"拿准确率换幻觉率"

口径 B 把整体幻觉率从 71.2% 压到 25.2%，但正常题准确率同时从 18.8% 掉到 2.4%、
引用命中率从 19.6% 掉到 5.2%。而 T1（引用不存在的文献）抗住率只从 43.3% 升到 53.3%。
**初步解读（待人工核对机制）**：strict 的低幻觉率主要来自"大量拒答"而非"答得更准"，
且对"凭空引用一篇假文献"这类陷阱帮助有限。这部分要在第 4.4 节消融里结合逐题回答细看。

## 状态

方向已定稿（2026-09-22）。

- [x] 项目骨架、语料拉取链路
- [x] 评测集 schema 定义
- [x] 评测集 v1 构建（400 条：正常题 250 + 三类陷阱题 150）
- [x] 评测集人工抽检（抽 40 条 / 10%，40 条全部通过）
- [x] 三层幻觉检测模块（第一层程序化 + 二三层 LLM 判官；第一层含 ID/标题/宽松变体三段判定）
- [x] 引用存在性远程补核（ID + 标题全部核验，`unverified` 清零，判定可复现）
- [x] 修正 prompt 示例 ID 泄漏后重跑基线（口径 A/B 各 400 条 + 判官 + 第一层）
- [x] 纯 LLM 基线结果（见"基线结果"小节）
- [ ] 第一层 `fabricated` 的假阳性复核（`0.70~0.75` 松口径边界样本 + `F1=0` 二次核验）
- [ ] 知识库构建（PDF 拉取解析 / 切分 / 嵌入 / 向量库）
- [ ] 检索与生成链路（召回 / 重排序 / 引用约束）
- [ ] RAG 对比与消融实验
- [ ] Web 演示层
- [ ] 开题报告与论文各章

