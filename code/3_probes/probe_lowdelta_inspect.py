#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
δ≈0 更新抽检：验证"MiniLM说没变"的更新到底真没变还是假没变
==========================================================
背景：δ探针发现中位δ=0.019、63%更新δ<0.083——即MiniLM认为过半更新
"没怎么变"。但MiniLM对代码偏钝(只13%更新δ>0.15)，所以这些低δ更新里
可能混着"实际改了逻辑、但MiniLM没察觉"的。

这批更新是省算叙事的命门：
  - 若低δ更新大多是格式/注释/重命名 → 真没变，省算成立
  - 若低δ里有 >改>=、+改-、加边界判断、改返回值 → MiniLM漏了真实语义
    变化，"δ≈0就不重算"会漏掉本该重算的，省算是拿正确性换算力

用法：
    python3 probe_lowdelta_inspect.py --data clean_data_576 --repo gin --n 20
输出：n对最低δ更新的 old→new 原文对照，人眼判断到底变没变。
"""
import json, argparse, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="clean_data_576")
ap.add_argument("--repo", default="gin")
ap.add_argument("--n", type=int, default=20, help="抽多少对最低δ更新看原文")
ap.add_argument("--sample", type=int, default=300)
args = ap.parse_args()

upath = os.path.join(args.data, args.repo, "updates.jsonl")
if not os.path.exists(upath):
    print(f"❌ 找不到 {upath}，用 --data 指定")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 先装: pip install --break-system-packages sentence-transformers")
    sys.exit(1)
m = SentenceTransformer("all-MiniLM-L6-v2")

ups = [json.loads(l) for l in open(upath, encoding="utf-8") if l.strip()]
ups = [u for u in ups if (u.get("old_text") or "").strip() and (u.get("new_text") or "").strip()]

import random
random.seed(42)
if len(ups) > args.sample:
    ups = random.sample(ups, args.sample)

olds = [u["old_text"] for u in ups]
news = [u["new_text"] for u in ups]
Eo = m.encode(olds, show_progress_bar=False)
En = m.encode(news, show_progress_bar=False)
for i, u in enumerate(ups):
    cos = np.dot(Eo[i], En[i]) / (np.linalg.norm(Eo[i]) * np.linalg.norm(En[i]) + 1e-9)
    u["_delta"] = 1 - cos

# 取 δ 最低的 n 对
low = sorted(ups, key=lambda u: u["_delta"])[:args.n]


def diff_lines(old, new):
    """简单逐行 diff，标出增删改的行"""
    import difflib
    ol, nl = old.splitlines(), new.splitlines()
    out = []
    for line in difflib.unified_diff(ol, nl, lineterm="", n=1):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            out.append("  [新增] " + line[1:].strip())
        elif line.startswith("-"):
            out.append("  [删除] " + line[1:].strip())
    return out


print("=" * 70)
print(f"抽检 δ 最低的 {args.n} 对更新（MiniLM认为'几乎没变'的）")
print("请人眼判断每一对：到底是'真没变(格式/注释/重命名)'还是")
print("'真变了但MiniLM没察觉(改逻辑/改运算符/改边界/改返回值)'")
print("=" * 70)

for idx, u in enumerate(low, 1):
    print(f"\n【{idx}】δ={u['_delta']:.4f}  {u['unit_id'][:60]}")
    dl = diff_lines(u["old_text"], u["new_text"])
    if not dl:
        print("  (逐行diff无差异——可能只差空白/末尾换行，基本是真没变)")
    else:
        for d in dl[:12]:  # 最多显示12行差异
            print(d)
        if len(dl) > 12:
            print(f"  ...(还有{len(dl)-12}行差异)")

print("\n" + "=" * 70)
print("判断指引（发给导师前先自己数一遍）：")
print(f"  在这 {args.n} 对里，数一下：")
print("  - 【真没变】格式/注释/空白/纯重命名 → 有几对？")
print("  - 【真变了但δ低】改了运算符(>/>=)、加减号、边界条件、")
print("     返回值、加/删了判断分支 → 有几对？")
print()
print("  若绝大多数是'真没变' → 省算成立，MiniLM的钝只让它更保守 ✅")
print("  若相当比例是'真变了但δ低' → 省算有漏洞：会漏重算本该重算的，")
print("     需在论文里量化这个漏检率，或换代码模型 🔴")
print("\n把这个数(真没变X对 / 真变了Y对)和上面几个典型例子发给导师。")
