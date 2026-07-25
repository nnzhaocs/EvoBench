#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
δ 探针 v2：检查 MiniLM 对代码改动敏不敏感（修正版）
==================================================
v1 的问题：用"文本长度差"挑大/小改动，但长度差 ≠ 语义改动大小
（'>' 改 '>=' 长度没变但语义反了；加注释长度变大但语义没变）。
分组标准错了，会把"分组错"误判成"MiniLM钝"。

v2 改法：不靠长度分组，改成【全谱扫描】——对所有更新算 δ，看 δ 的
分布有没有区分度。MiniLM 若不钝，面对上千个各异的真实改动，δ 一定
有明显跨度；若钝，δ 会全挤在一个小值附近，不管改动大小。
辅以字符级编辑距离(实际改了多少)做参考，比长度差可靠。

用法：
    python3 probe_delta_v2.py --data clean_data_576 --repo gin
"""
import json, argparse, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="clean_data_576")
ap.add_argument("--repo", default="gin")
ap.add_argument("--sample", type=int, default=300, help="随机抽多少对算δ(全跑太慢)")
args = ap.parse_args()

upath = os.path.join(args.data, args.repo, "updates.jsonl")
if not os.path.exists(upath):
    print(f"❌ 找不到 {upath}，用 --data 指定数据目录")
    sys.exit(1)

print("加载 MiniLM...")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 先装: pip install --break-system-packages sentence-transformers")
    sys.exit(1)
m = SentenceTransformer("all-MiniLM-L6-v2")

def edit_distance(a, b):
    # 简单字符级编辑距离(Levenshtein),截断长文本防慢
    a, b = a[:500], b[:500]
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]

ups = [json.loads(l) for l in open(upath, encoding="utf-8") if l.strip()]
ups = [u for u in ups if (u.get("old_text") or "").strip() and (u.get("new_text") or "").strip()]

import random
random.seed(42)
if len(ups) > args.sample:
    ups = random.sample(ups, args.sample)

print(f"仓库={args.repo}，算 {len(ups)} 对更新的 δ...\n")

deltas = []
records = []
olds = [u["old_text"] for u in ups]
news = [u["new_text"] for u in ups]
# 批量编码更快
E_old = m.encode(olds, show_progress_bar=False)
E_new = m.encode(news, show_progress_bar=False)
for i, u in enumerate(ups):
    eo, en = E_old[i], E_new[i]
    cos = np.dot(eo, en) / (np.linalg.norm(eo) * np.linalg.norm(en) + 1e-9)
    d = 1 - cos
    ed = edit_distance(u["old_text"], u["new_text"])
    deltas.append(d)
    records.append((d, ed, len(u["old_text"]), len(u["new_text"])))

deltas = np.array(deltas)

# === 全谱分布统计 ===
print("=" * 60)
print("δ 全谱分布（这是核心，看有没有区分度）")
print("=" * 60)
for p in [0, 10, 25, 50, 75, 90, 100]:
    print(f"  第{p:3}百分位: δ = {np.percentile(deltas, p):.3f}")
print(f"  均值={deltas.mean():.3f}  标准差={deltas.std():.3f}")

# 区分度：跨度 + 有多少比例的更新 δ 明显不为0
span = np.percentile(deltas, 90) - np.percentile(deltas, 10)
frac_moving = (deltas > 0.15).mean()
print(f"\n  δ 跨度(p90-p10) = {span:.3f}")
print(f"  δ>0.15 的比例   = {frac_moving*100:.0f}%  (这些是MiniLM'看得出变化'的更新)")

# === 编辑距离 vs δ 相关性（辅助验证：真改得多的δ是否真更大）===
eds = np.array([r[1] for r in records])
if eds.std() > 0 and deltas.std() > 0:
    corr = np.corrcoef(eds, deltas)[0, 1]
    print(f"  编辑距离 vs δ 相关性 = {corr:.3f}  (>0.3说明'改得多→δ大',MiniLM在跟随真实改动)")
else:
    corr = 0

# === 判读 ===
print("\n" + "=" * 60)
print("判读")
print("=" * 60)
if np.percentile(deltas, 90) < 0.10:
    print("🔴 MiniLM 对代码钝：连改动最大的10%更新 δ 都<0.10。")
    print("   向量几乎不动，recall低/拉不开是度量塌了，不是真实规律。")
    print("   → 省算叙事是假象，考虑换 CodeBERT 类代码模型重做。")
elif span > 0.15 and frac_moving > 0.3 and corr > 0.2:
    print("✅ MiniLM 能感知代码变化：δ 有明显跨度、相当比例更新δ>0.15、")
    print("   且δ跟随真实编辑量(相关>0.2)。")
    print("   → 拉不开是真实的'检索对小改动鲁棒'，省算叙事成立。")
    print("   (recall绝对值低是另一问题，查GT质量/top-k，不影响省算主结论)")
else:
    print("⚠️ MiniLM 对代码有一定敏感度但不强(跨度或相关性中等)。")
    print("   省算大方向可用，但论文需诚实说明嵌入模型对代码的局限，")
    print("   或补一个代码模型(CodeBERT)做对照。把上面全部数字发给导师。")

print("\n把上面【δ全谱分布】和【判读】整段发给导师。")
