# EvoBench: 检索维护基准

> EvoBench 是一个评估向量检索索引维护策略的基准测试，包含真实代码演化数据、多模型嵌入对比、以及判别力分析工具。

## 目录结构

```
Evobench/
├── code/                    # 全部代码（均已修复已知 bug）
│   ├── 1_pipeline/          # 数据构造流水线（从 git 抽取到清洗）
│   ├── 2_experiments/       # 核心实验（判别力 sweep、多模型、全指标）
│   ├── 3_probes/            # 分析探针（δ 分布、漏检分类、GT 验证）
│   ├── 4_expansion/         # 扩仓库工具
│   └── run_v4_full.py       # 一键执行完整流程
├── data/                    # 基准数据集
│   ├── bench_all_clean/     # 10 库清洗后数据（四件套）
│   └── traceable_4libs/     # 4 库可溯源数据（带 SHA）
├── results/                 # 实验结果
│   ├── allmetrics/          # 全指标结果（Recall/MRR/nDCG/决策指标）
│   ├── sweep/               # 判别力 sweep 结果（6 模型）
│   ├── cross_model/         # 跨模型一致性
│   └── lowrecall/           # 低 recall 对照实验
├── logs/                    # 运行日志
│   ├── pipeline_v4/         # 流水线执行日志
│   ├── sweep_10libs/        # 10 库 sweep 日志（6 模型）
│   └── sweep_v4/            # v4 版 sweep 日志
├── .gitignore               # Git 忽略规则
├── LICENSE                  # MIT 许可证
├── requirements.txt         # Python 依赖
└── README.md                # 本文件
```

## 数据集

### bench_all_clean（10 库，2378 查询）

| 仓库 | 语言 | units | updates | 带GT查询 |
|------|------|-------|---------|---------|
| gin | Go | 434 | 1802 | 552 |
| echo | Go | 678 | 3351 | 474 |
| cobra | Go | 259 | 1079 | 335 |
| lodash | JS | 535 | 950 | 264 |
| pylint | Python | 162 | 720 | 223 |
| requests | Python | 43 | 69 | 151 |
| flask | Python | 131 | 334 | 128 |
| aiohttp | Python | 65 | 409 | 93 |
| click | Python | 23 | 86 | 84 |
| express | JS | 36 | 136 | 74 |

每个仓库目录包含标准四件套：
- `units.jsonl` — 函数级单元（基准版本）
- `updates.jsonl` — 真实 old→new 编辑演化
- `queries.txt` — 真实 GitHub issue 作为查询
- `ground_truth.json` — 检索 GT（查询→相关函数）

### traceable_4libs（4 库可溯源子集）

flask / requests / express / gin，每条数据带真实 commit SHA + issue 号，可在 GitHub 独立核对。

## 嵌入模型

| 模型 | 类型 | 维度 | 健康状态 |
|------|------|------|---------|
| all-MiniLM-L6-v2 | 通用句向量 | 384 | healthy |
| all-mpnet-base-v2 | 通用句向量 | 768 | healthy |
| BAAI/bge-small-en-v1.5 | 通用句向量 | 384 | healthy |
| microsoft/codebert-base | 代码专用 | 768 | broken（向量塌缩） |
| microsoft/graphcodebert-base | 代码专用 | 768 | weak |
| microsoft/unixcoder-base | 代码专用 | 768 | healthy |

> CodeBERT/GraphCodeBERT 存在向量塌缩问题（sanity check 差距 +0.020/+0.099），其检索指标偏低但数据已保留用于对照。
>
> **数据恢复说明**：CodeBERT/GraphCodeBERT 此前因模型加载问题多次 broken，部分实验数据仅打印到 stdout/日志而未落盘为 JSON。现已通过 `build_step_allmetrics_v3.py` 重新生成完整的 `allmetrics_codebert.json` 和 `allmetrics_graphcodebert.json`，所有论文 Table 2 中的数字均可溯源到对应 JSON 文件。

## 核心实验

### 1. 判别力 Sweep（Table 1）

脚本: `code/2_experiments/build_step4_multimodel.py`

对比三种重算策略的 Recall@10：
- **always-rebuild**: 每次更新都重算（上界）
- **never-rebuild**: 从不重算（下界）
- **audit(τ)**: δ>τ 才重算（本文方法）

结果: `results/sweep/{model}/baseline_sweep_{model}.json`

### 2. 全指标分析（Table 2）

脚本: `code/2_experiments/build_step_allmetrics_v3.py`

计算检索质量指标（Recall@k, MRR@k, nDCG@k, 翻转率）和决策指标（F1, P, R, FN 率, 省算%）。

结果: `results/allmetrics/allmetrics_{model}.json`

已有结果: minilm, mpnet, codebert, graphcodebert

各模型平均指标汇总（k=10）：

| 模型 | Avg Recall@10 | Avg MRR@10 | Avg nDCG@10 | Avg 翻转率 |
|------|-------------|-----------|------------|-----------|
| minilm | 0.5239 | 0.3582 | 0.4143 | 0.9302 |
| mpnet | 0.5407 | 0.3826 | 0.4369 | 0.8980 |
| codebert | 0.1254 | 0.0509 | 0.0783 | 0.9794 |
| graphcodebert | 0.1672 | 0.0668 | 0.1012 | 0.9928 |

> CodeBERT/GraphCodeBERT 指标偏低系向量塌缩所致（见 sanity check），数据保留用于跨模型一致性对照。

### 3. 跨模型一致性

基于 δ（嵌入漂移）排序的 Spearman 相关，验证不同模型对"哪些更新重要"的判断是否一致。

4 模型一致性矩阵（Spearman ρ）：

| | codebert | graphcodebert | minilm | mpnet |
|------|------|------|------|------|
| codebert | 1.000 | 0.934 | 0.814 | 0.797 |
| graphcodebert | 0.934 | 1.000 | 0.843 | 0.840 |
| minilm | 0.814 | 0.843 | 1.000 | 0.946 |
| mpnet | 0.797 | 0.840 | 0.946 | 1.000 |

结果: `results/allmetrics/cross_model_delta_consistency.json`

### 4. 低 Recall 对照实验

排除"always/never 差异小是因为 recall 绝对值太低"的替代解释。

结果: `results/lowrecall/lowrecall_control.json`

## 代码说明

### 1_pipeline/（数据构造）

| 脚本 | 功能 |
|------|------|
| build_step1_extract_evolution.py | 从 git 历史抽取函数级演化（old→new） |
| build_step2_queries_and_gt.py | 从 issue 经 "Fixes #N" 链接配检索 GT |
| build_step2p5_to_fourset.py | 转四件套格式 |
| build_step3_stats_and_package.py | 统计与打包 |
| clean_benchmark_data.py | 去重 + 剔离群 |

### 2_experiments/（核心实验）

| 脚本 | 功能 | 修复记录 |
|------|------|---------|
| build_step4_baseline_sweep_fixed.py | 判别力 sweep（修复版） | U0 基线用 old_text（非 units.text） |
| build_step4_multimodel.py | 多模型 sweep | CodeBERT 用 mean-pooling，BGE 加查询前缀 |
| build_step_allmetrics_v3.py | 全指标计算 | 扩展 I(u) 定义（覆盖非 GT 挤出情况） |
| exp_lowrecall_control.py | 低 recall 对照 | 三口径（全集/并集/GTleqk） |
| codemodel_sanity_check.py | 模型健康检查 | — |

### 3_probes/（分析探针）

| 脚本 | 功能 |
|------|------|
| probe_delta_v2.py | δ 全谱分布分析 |
| lowdelta_classify_v2.py | 低 δ 更新客观分类（已修：函数替换检测 + 数字字面量） |
| probe_lowdelta_inspect.py | 低 δ 更新人工核查 |
| validate_claims.py | 数据真实性验证 |

## 复现方法

```bash
# 环境
pip install -r requirements.txt

# 1. 判别力 sweep（任选模型）
cd code/2_experiments
python build_step4_multimodel.py --model minilm --bench-final ../../data/bench_all_clean --out ../../results/sweep/minilm

# 2. 全指标分析
python build_step_allmetrics_v3.py --model minilm --bench-final ../../data/bench_all_clean --out ../../results/allmetrics

# 3. 低 recall 对照
python exp_lowrecall_control.py --bench-final ../../data/bench_all_clean --out ../../results/lowrecall
```

## 数据溯源

每条数据带真实 commit SHA + issue 号，溯源链：
```
units/updates ← 真实 git 提交历史（同一函数多个真实版本）
queries/GT    ← 真实 issue #N ──"Fixes #N"──▶ 真实修复 commit ──▶ 改动的函数
```

## 关键修复记录

| 脚本 | 修复内容 |
|------|---------|
| build_step4_baseline_sweep_fixed | U0 基线用 old_text（非 units.text），修版本错位 |
| build_step4_multimodel | CodeBERT 用 mean-pooling，BGE 加查询前缀 |
| lowdelta_classify_v2 | 加函数替换检测 + 数字字面量计入结构 token |
| exp_lowrecall_control | 三口径（全集/并集/GTleqk），修 selection bias |
| build_step_allmetrics_v3 | 扩展 I(u) 定义，覆盖非 GT 单元挤出 GT 的情况 |
