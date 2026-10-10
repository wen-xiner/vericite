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
├── corpus/              # 向量库产物 + 本地模型权重（均不入库：体积大，可由脚本重建）
│   ├── index/               # faiss 索引 + 元数据（build_index.py 生成，CPU 约 90 秒可重建）
│   └── models/bge-m3/       # 本地嵌入模型权重（约 2.3GB，走 hf-mirror 下载）
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
- **数据链路**：arXiv API（143 篇，已完成）→ 文档切分 → 向量嵌入入库
  - 当前档位 **摘要级**：文档单元 = 一篇论文，向量取「标题 + 摘要」；全文 PDF 解析留作对照实验
  - PDF 解析器：本机无 Docker / 无 Java，GROBID 跑不起来，预留 pdfplumber 路线
- **问答链路**：向量召回（top-k）→ 重排序 → 引用约束 Prompt → LLM 生成 → 幻觉检测
- **LLM**：API 调用（DeepSeek 或同类，OpenAI 兼容接口）
- **嵌入**：sentence-transformers + `BAAI/bge-m3`（**多语言**——中文问句检索英文语料，见"知识库与检索"）
- **向量库**：FAISS（`faiss-cpu`，`IndexFlatIP`）
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
是同一篇）。两口径合计命中 19 条，例如
`sam rag self adaptive multimodal retrieval augmented generation`
→ `Self-adaptive Multimodal Retrieval-Augmented Generation`（F1 0.857）。

> ⚠️ **这个态是软证据，两个方向都会错，论文里只能当"疑似真实"用。**
>
> **偏松（假阳性）≈ 17%**：逐条人工复核 18 条（**必须看原始写法**，归一化后大小写全丢，
> 看不出缩写差异）后，**15 条站得住、3 条是同主题的另一篇**：
> `FenCE: A Framework for Evaluating the Factual Consistency of Abstractive Summarization`
> → 真实是 `Just ClozE! ...`（缩写 FenCE ≠ ClozE）；
> `QA-RAG: A Hybrid Retrieval-Augmented Generation Framework for Question Answering`
> → 真实是 `RAG-BioQA: ...`；
> `Federated Learning for Retrieval-Augmented Generation`
> → 真实是 `Privacy-Preserving Federated **Embedding** Learning for Localized RAG`（漏了 Embedding）。
> 另有已回退的一条 `etrag a general framework for relation extraction...` →
> `Retrieval-Augmented Generation-based Relation Extraction`（2026-09-23 实测 `ti:"ETRAG"`
> 命中 0 条 —— 世上没有以 ETRAG 命名的论文）。
> **病灶：F1 只算词集合重合，主题词（retrieval / augmented / generation / hallucination）
> 一重叠就能顶到 0.7 以上。**
>
> **偏紧（假阴性）**：真实标题若不含宽松检索所用的某个核心词，会 0 命中 → F1=0 →
> 被判 `fabricated`。二次核验（改用 3 词 AND 与 `all:` 全字段检索，见
> `src/experiments/verify_unmatched.py`）在 94 个"查无此文"标题里找回 13 个有对应（13.8%），
> 但逐条人工复核后**只有 5 条站得住**（自造词或长独特短语逐字对上，如 `HetaRAG`、`APEFT`、
> `data importance learning`），其余 8 条是同主题的另一篇或句中片段。
> 现存实例：`CommunityKG-RAG: A Retrieval-Augmented Generation Framework...`（模型写了省略号
> 截断）被 5 词 AND 判成 F1=0 → `fabricated`，但 `CommunityKG-RAG` 这个自造词在真实标题里
> 逐字存在，**属假阳性**。
>
> **根因：F1 单指标（以及"最长实词 AND 检索"这套装置）分不开"词序调换的同一篇"和
> "同主题的另一篇"，两者都落在 0.60~0.78，区间完全重叠。**
> 已实测三条更细的词面指标都不可分（示例：假阳性 `QA-RAG` 覆盖率 0.78 vs
> 真阳性 `Mitigating ... via Dynamic Trust Calibration` 覆盖率 0.80；
> 缩写否决规则又会误杀 `ALCE: ...` 这类"模型加了正确的系统缩写前缀"的真阳性）。
>
> **于是改用语义判官做佐证**（`src/experiments/judge_title_pairs.py`）：把「模型标题 +
> 候选标题」成对给 LLM，只给这两串、问是否指同一篇（不给 ID / 问题 / 摘要）。18 条 = 18 次调用。
> **结果：与人工逐条复核一致率 16/18 = 88.9%。**
> 2 条不一致，且**分歧是系统性的、属口径之争**：
> - 判官按「**这个标题能否唯一确定**这篇论文」判 → 尾部机制名/范围词不同 → 不是；
> - 人工按「**被引的这篇文献是否真实存在**」判 → 主干逐字一致 → 是。
>
> 不一致的两条正是"主干一致、尾部被写错"：
> `Mitigating Hallucination in LVLMs via Dynamic Trust Calibration`（真实是 `via Adaptive
> Attention Calibration`，前 9 词完全一致）、
> `Self-adaptive Retrieval-Augmented Generation`（漏了 `Multimodal`，同一回答另一处写对了）。
> 两者判为「同一篇」还是「另一篇」，取决于第 4.6 节怎么定义这个态 ——
> **这不是判官出错，而是 `exists_variant` 这一态本身夹在两个问题中间，模糊性不可消除。**
>
> 沿用人工口径（该态 18 条、精度 83%）作为**主线**，取判官的严口径只作对照：
> 严口径下该态 18 → 13、精度 72%，判官判「不是」的 5 条回流 `fabricated`（141 → ~146）。
> **本次采用"判官只做佐证、不改判定"**（判官结论见
> `experiments/cache/title_pair_judge.json`），所以下面所有数字与主指标均未受影响。
>
> ✅ **对主指标无影响**（2026-09-23 核算）：所有含 `exists_variant` 的题目都因其它失败模式
> （内容不支持 / 归属错误 / 未命中语料）已计为幻觉，唯一例外是口径 B 的 1 条，
> 折合幻觉率 0.25 个百分点。**即整体幻觉率不依赖这个态怎么判。**

**第二、三层为什么必须用 LLM 判官**：论断是中文、摘要是英文，词面重合度必然接近 0，
规则版要么恒判"不支持"要么恒判"支持"，等于没做（与 T2 关键词校验空转是同一类坑）。
词面版本只保留作第 4.5 节的对照基线。

**引用抽取要防正文噪声**：引号里的东西不都是论文标题。实测抓到过
`** 或 **arXiv:2304.09848,`（模型正文的普通引号 + markdown 星号）、
`**most** of the documents **mention** **risk X**`（模型在复述指令）、
`the company was founded in 1998 and is headquartered in Berlin.`（模型举例用的普通句子）、
`target context window size`（正文短语碎片）被当成"书名号内容"。
这类串送去 arXiv 必然查不到，会被判成"编造"——**凭空造出幻觉**。

所以抽取层加了粗筛（`src/detection/extract.py` 的 `looks_like_title`），
排除条件只收不放：含中文、含 markdown 标记、内部带 arXiv ID、英文词少于 4 个、
以**单个句号**收尾（整句而非标题；省略号 `...` / `…` 表示标题被截断，必须保留）、
以祈使动词开头（`review|using|rewrite|please|…`，模型在复述指令）、
全小写且无冒号且实词少于 6 个（正文碎片）。
**2026-09-23 收紧后实测：两口径合计多排除 12 条、新增 0 条**（纯收紧）。
**宁可漏检，也不能把正文噪声当标题。**

> ⚠️ **这条规则改错过一次，值得记下来**：初版写成 `raw.endswith(".")`，
> 把 `CommunityKG-RAG: A Retrieval-Augmented Generation Framework...`（省略号截断的**合法标题**）
> 一起删了。当时只验了"新增 0 条"就宣称"只收紧不放宽"——**只看新增、不看删除，等于没验。**
> 现在有专门工具兜底：`python src/experiments/check_extract_diff.py`
> 会同时列出**新增**（应为 0）和**删除**（逐条确认是否噪声），有新增就返回非 0 退出码。
> **改完抽取规则必跑它。**

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

**改判定规则 / 做复核时的三个配套工具**：

```bash
# 改了 looks_like_title 必跑：同时列「新增」（应为 0）和「删除」（逐条确认是否噪声）
python src/experiments/check_extract_diff.py

# fabricated 的二次核验：换更宽的检索式重查（3 词 AND + all: 全字段），找回"真实存在但标题写错"的
python src/experiments/verify_unmatched.py --summary     # 增量、可断点续跑
python backfill_second_pass.py --dry                     # 把人工确认成立的回灌进判定（改完要重跑 --refresh-layer1）

# exists_variant 的语义判官佐证：只给两条标题，不改判定与指标
python src/experiments/judge_title_pairs.py --dry        # 先列标题对，不调 LLM
python src/experiments/judge_title_pairs.py              # 跑判官（18 次调用）
python src/experiments/judge_title_pairs.py --summary    # 只看与人工的一致率
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

## 知识库与检索

### 当前档位：摘要级（document-level）

语料 143 篇（`data/arxiv_raw.jsonl`），**一个文档单元 = 一篇论文**，向量取自「标题 + 摘要」。
链路：`build_index.py` 编码入库 → `retriever.py` 向量召回。产物落 `corpus/index/<tag>/`
（`index.faiss` + `meta.jsonl` + `config.json`；config 里记了**语料 sha256**，保证指标能溯源到具体一版语料）。

**为什么先做摘要级、不先解析 PDF**：`eval/v1` 的 400 条题里，250 道正常题的证据字段带
`verification.evidence_verbatim_in_abstract = True`（2026-09-23 全量核实：250 条全为 True，
另抽 60 条做英文片段-摘要子串比对，**60/60 命中**）—— **出题粒度本来就是摘要**。
所以先把「召回 → 生成 → 三层检测」端到端跑通，全文级 PDF 解析留作对照实验。

### 技术选型（2026-09-23 定）

| 项 | 选型 | 理由 |
|---|---|---|
| 嵌入模型 | `BAAI/bge-m3`（1024 维） | **必须多语言**——评测问句是中文、语料是英文，纯 en / 纯 zh 模型都不适用 |
| 向量库 | `faiss-cpu`（`IndexFlatIP`） | 向量已 L2 归一化，内积即余弦；大纲点名的可选项之一 |
| 模型来源 | `corpus/models/bge-m3/`（本地目录） | `huggingface.co` 直连超时，权重走 `hf-mirror.com` 下到本地 |
| PDF 解析 | **暂未启用** | 本机无 Docker / 无 Java，GROBID 跑不起来；预留 pdfplumber 路线 |

### 召回结果（摘要级，top-k = 10）

| 指标 | 值 |
|---|---|
| Recall@1 | **93.2%** |
| Recall@3 | 97.4% |
| Recall@5 | 97.7% |
| Recall@10 | 98.2% |
| MRR | 0.9535 |
| 正常题 | **248/250 = 99.2%** |

（统计口径 = 340 条带 `gold_arxiv_ids` 的题；60 条伪文献题没有金标来源，不参与召回评估。
原始结果见 `experiments/retrieval/recall_abstract.json`。）

> ⚠️ **这个高召回不能用来论证「检索策略好」。** 出题证据的粒度是摘要，知识库的粒度也是摘要，
> 两者同源 → 检索天然偏易，Recall@1 93% 是**装置造成的天花板效应**，不是模型/检索器的功劳。
> 要论证切分粒度、top-k、重排序的优劣，必须上**全文级知识库**做对照。
> 这一条必须写进论文，否则就是拿装置的假象冒充结论。

未召回的 6 条（逐条留痕在结果 JSON 的 `misses` 里）：2 条正常题（`N-0097` / `N-0195`）
+ 4 条 `trap_mismatch`。「张冠李戴」题召回的不是 gold —— 这**符合该题型的设计**：
问题描述的是 A 的内容却指向 B，按内容召回自然会命中 A。

### 复现

```bash
# 1. 依赖（必须绕开沙箱代理，否则 PyPI 镜像源会返回空、pip 误报"找不到包"）
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  pip install faiss-cpu sentence-transformers

# 2. 模型权重下到 corpus/models/bge-m3/（约 2.3GB，huggingface.co 直连不通，走 hf-mirror）

# 3. 建索引（143 篇摘要，CPU 约 90 秒）
python src/retrieval/build_index.py

# 4. 召回评估（Recall@k / MRR，按题型分组）
python src/retrieval/retriever.py --eval --top-k 10 \
  --json-out experiments/retrieval/recall_abstract.json

# 单条查询
python src/retrieval/retriever.py --query "检索增强生成如何缓解大模型幻觉？" --top-k 5
```

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
| 正常题幻觉率 | 77.6% | 26.4% |
| T1 伪文献 · 抗住率 | 43.3% | 53.3% |
| T2 无支撑 · 抗住率 | 24.0% | 96.0% |
| T3 归属错位 · 抗住率 | 52.5% | 92.5% |
| **整体幻觉率** | **71.2%** | **24.8%** |

> 表中数字为 **2026-09-23 第一层复核后**的值（抽取层收紧 + 二次核验回灌后重跑，
> 判官结论沿用缓存、未花 LLM 费用）。复核前为：正常题幻觉率 B 27.2%，整体幻觉率 B 25.2%。
> **口径 A 的主指标未变（71.2% / 77.6% / 52.5%），只有口径 B 小幅改善。**
> 变化来自 **12 条正文噪声**被抽取层正确排除（原先被误判为"编造"，虚增了幻觉）。

### 结论一：模型几乎不编 ID，但会编标题（第一层程序化判定，硬/软证据分开看）

口径 A（主线）：

| 引用类型 | 引用总次数 | 查无此文 | 编造率 |
|---|---|---|---|
| arXiv ID（硬证据） | 491 | **0** | **0%** |
| 论文标题（软证据） | 368 | 94 | 25.5% |

两口径合计（A + B）：

| 引用类型 | 引用总次数 | 查无此文 | 编造率 |
|---|---|---|---|
| arXiv ID（硬证据） | 803 | **1** | **0.12%** |
| 论文标题（软证据） | 617 | 140 | 22.7% |

这是本项目最有价值的发现，也直接决定了检测器的设计：**如果只核验 arXiv ID 是否存在，
对模型实际产生的引用幻觉几乎检不出来** —— 因为模型给出的 ID 绝大多数是真的，编造发生在标题上。

那唯一 1 条 ID 编造（口径 B，T1-0002）：`arXiv:1809.12571`，配套标题
`HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering` **写对了**，
但 ID 记岔一位 —— 2026-09-23 实测 `1809.12571` 不存在、真实 ID 是 `1809.09600`。
**所以"模型不编 ID"应精确表述为"模型极少编 ID（1/803 = 0.12%），且错法多是记岔位数，
标题仍正确"。**

（另有 19 条标题属 `exists_variant`，即写法有偏差但疑似真实文献。该态是**软证据**、
人工复核 18 条后精度约 83%，详见上文宽松复核一节 —— 它不计入"查无此文"。）

### 结论二：prompt 约束是"拿准确率换幻觉率"

口径 B 把整体幻觉率从 71.2% 压到 24.8%，但正常题准确率同时从 18.8% 掉到 2.4%、
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
- [x] 第一层复核：抽取层收紧（多排除 12 条正文噪声、新增 0，差分核查工具
  `src/experiments/check_extract_diff.py` 兜底）+ 二次核验回灌（5 条，
  `src/experiments/verify_unmatched.py` + `backfill_second_pass.py`）后重跑两口径
- [x] `exists_variant` 的判据建设。词面路线已试尽（F1 阈值 / 5 词 AND 检索 / 覆盖率 /
  独特词对齐 / 缩写展开否决 —— 五条都验过，真/假区间完全重叠）；改上**语义判官**做佐证，
  与人工复核一致率 16/18 = 88.9%（`src/experiments/judge_title_pairs.py`）
- [ ] 第 4.6 节需要**定义"同一篇"的口径**：判官与人工的 2 条分歧全在"标题能否唯一确定"
  vs "被引文献是否存在"这一层。口径定了才能报该态的最终精度（人工口径 83% / 判官严口径 72%）
- [x] 知识库构建（**摘要级**）：143 篇语料向量化入库（`BAAI/bge-m3` + `faiss-cpu`）
- [x] 向量召回（top-k）：Recall@1 93.2% / Recall@5 97.7% / MRR 0.9535（摘要级，
  结果见 `experiments/retrieval/recall_abstract.json`；**含装置局限说明**，见"知识库与检索"）
- [ ] 知识库构建（**全文级**）：PDF 拉取解析 + 切分，与摘要级做对照
  —— 摘要级召回偏高是装置所致，**论证检索策略优劣必须靠全文级这一档**
- [ ] 重排序
- [ ] RAG 问答链路（检索材料注入 prompt + 生成）
- [ ] RAG 与纯 LLM 对比、消融实验（切分粒度 / top-k / 重排序）
- [ ] Web 演示层
- [ ] 开题报告与论文各章

