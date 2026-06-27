# CCL26 Task5: MGBIE — Minor-Grain Breeding Information Extraction

> 第二届中国计算语言学大会（CCL 2026）评测任务五：小杂粮育种信息抽取  
> **本方案取得第二名成绩**

## 方法概述

本方案采用 **检索增强 + 多Prompt集成 + 多次采样投票** 的 LLM-based 信息抽取方案，核心思路：

1. **BM25/Hybrid 检索增强**：对每个待抽取文档，用 BM25 字符 n-gram（或 BM25+Embedding RRF 混合）检索最相似的训练文档作为 few-shot 示例
2. **三 Prompt 变体集成**：设计 V6（严格类型约束）、V7（宽松语义）、V8（平衡）三种 System Prompt，覆盖不同抽取偏好
3. **多次采样 + 多数投票**：每个 Prompt 进行 N 次独立采样（每次随机选取 8 个示例），共 3×N 次抽取，通过投票阈值过滤噪声，保留高置信度结果

### 方法优势

- **无需训练**：纯 In-Context Learning，无需微调模型
- **高鲁棒性**：多采样投票有效消除 LLM 单次输出的随机误差
- **可扩展**：增加采样次数即可提升召回率，调高阈值提升精确率

## 项目结构

```
code/
├── predict.py              # 主入口：检索增强多采样投票预测
├── model.py                # LLM 客户端（OpenAI 兼容 API）
├── schema.py               # 12 种实体类型 + 6 种关系类型定义
├── tools.py                # JSON 解析、实体对齐、格式转换、投票、评估
├── retriever.py            # BM25 字符 n-gram 检索器
├── embedding_retriever.py  # Embedding 检索器 + Hybrid RRF 融合
├── prompts.py              # V6/V7/V8 三个 System Prompt
├── evaluate.py             # 独立评估脚本
├── distill_skills.py       # [可选] 从训练数据蒸馏抽取技能
├── deduplicate_skills.py   # [可选] 技能去重
├── skill_retriever.py      # [可选] 技能检索器
├── prompts_skill.py        # [可选] 技能增强 Prompt
├── predict_skill.py        # [可选] 技能增强预测入口
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── data/                   # 存放训练/测试数据 (需自行放入)
│   ├── train.json
│   ├── test.json
│   └── dev_split.json      # (可选) 开发集划分
└── results/                # 输出结果目录
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入实际配置：

```bash
cp .env.example .env
```

需配置：
- `MGBIE_BASE_URL`：OpenAI 兼容 API 地址（如 vLLM/sglang 部署的模型服务）
- `MGBIE_MODEL`：模型名称
- `EMBEDDING_MODEL_PATH`：Embedding 模型路径（默认 `sentence-transformers/all-mpnet-base-v2`）

### 3. 准备数据

将官方数据放入 `data/` 目录：
- `data/train.json`：训练集（作为检索 demo pool）
- `data/test.json`：测试集

## 运行方法

### 核心流程（BM25 检索 + 投票）

```bash
# 推理（BM25-only 检索，推荐首次运行使用）
python predict.py \
    --input data/test.json \
    --output results/submit.json \
    --pool data/train.json \
    --workers 16 \
    --retriever bm25 \
    --zip results/submit.zip

# 推理（Hybrid 检索：BM25 + Embedding RRF 融合）
python predict.py \
    --input data/test.json \
    --output results/submit.json \
    --pool data/train.json \
    --workers 16 \
    --retriever hybrid
```

### 开发集评估

```bash
python predict.py \
    --input data/train.json \
    --dev_split data/dev_split.json \
    --output results/dev_pred.json \
    --workers 16

# 或独立评估
python evaluate.py --pred results/dev_pred.json --gold data/train.json \
    --dev_split data/dev_split.json
```

### 预计算 Embedding 缓存（加速 Hybrid 检索）

```bash
python embedding_retriever.py --docs data/train.json --output data/train_embeddings.npy
```

### [可选] 技能增强流程

技能增强是一种补充方案，通过从训练数据中蒸馏细粒度抽取规则来提升效果：

```bash
# Step 1: 蒸馏技能
python distill_skills.py --input data/train.json --output data/skills_raw.jsonl --workers 32

# Step 2: 去重
python deduplicate_skills.py --input data/skills_raw.jsonl \
    --output data/skills_dedup.json --embeddings data/skills_embeddings.npy

# Step 3: 技能增强预测
python predict_skill.py \
    --input data/test.json \
    --output results/submit_skill.json \
    --pool data/train.json \
    --skills data/skills_dedup.json \
    --skill_embeddings data/skills_embeddings.npy \
    --workers 16
```

## 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_DRAWS` | 60 | 每个 Prompt 变体的采样次数 |
| `VOTE_THR` | 91 | 投票阈值（总 180 次中出现 ≥91 次保留） |
| `FEWSHOT_K` | 8 | 每次采样使用的 few-shot 示例数 |
| `RETRIEVAL_M` | 30 | BM25 检索的候选池大小 |
| `TEMPERATURE` | 0.0 | LLM 采样温度 |

> 投票阈值和采样次数需要根据实际情况调整。增大 N_DRAWS 提高稳定性但增加计算量，阈值约为总采样数的 50% 效果较好。

## 评估指标

采用官方评估方式：
- **NER Score** = 0.5 × F1 + 0.25 × P + 0.25 × R
- **RE Score** = 0.5 × F1 + 0.25 × P + 0.25 × R  
- **Total** = 0.4 × NER_Score + 0.6 × RE_Score

## 模型要求

本方案需要部署一个支持 OpenAI Chat Completions API 的大语言模型服务（如通过 vLLM 或 sglang 部署）。推荐使用支持 thinking/reasoning 模式的强推理模型（如 Qwen3-Thinking 系列）以获得最佳效果。

## License

本代码仅供学术研究使用。
