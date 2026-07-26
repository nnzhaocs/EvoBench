# EvoBench 数据溯源包

> 全部真实数据，可追溯到 GitHub。经 validate_claims.py 验证：真实编辑演化✅、跨语言✅、GT可追溯✅。

## 内容

四个仓库目录（flask/requests/express/gin），每个含标准四件套：
- `units.jsonl` — 单元（函数基准版本），带真实 base_commit
- `updates.jsonl` — 真实 old→new 编辑，带 from_commit/to_commit（真实 SHA）
- `queries.txt` — 真实 GitHub issue 作查询
- `ground_truth.json` — 检索 GT（查询→相关函数），来自 issue 的真实修复 commit

## 数据来源（溯源链）

```
units/updates ← 真实 git 提交历史（同一函数多个真实版本）
queries/GT    ← 真实 issue #N ──"Fixes #N"──▶ 真实修复 commit ──▶ 改动的函数
```

## 规模

| 仓库 | 语言 | units | updates | 带GT查询 |
|------|------|-------|---------|---------|
| flask | Python | 131 | 334 | 63 |
| requests | Python | 43 | 69 | 18 |
| express | JS | 36 | 136 | 49 |
| gin | Go | 434 | 1802 | 499 |
| 合计 | 3语言 | 644 | 2341 | 629 |

## 来源说明（诚实标注）

- **units/updates**：学生用 step1 跑出的真实数据（step1 无 bug，数据正确）。
- **queries/GT**：用**修复版 step2** 重新生成（学生原跑的旧版有 diff 前缀 bug、GT 几乎全空，那批错误数据**未纳入本包**）。
- 本包只含真实、经验证的数据。错误的旧 GT 已排除。

## 可追溯性 = benchmark 的 provenance

每条数据带真实 commit SHA + issue 号，审稿人可在 GitHub 独立核对。
这就是 benchmark 需要的可追溯性，无需额外溯源系统。
