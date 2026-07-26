#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低δ更新客观分类器（替代肉眼判"真变假变"）
==========================================
问题：肉眼看diff判"真变了没有"不可复现——导师和学生判同样20条，
数出的"真变了"具体是哪几对都对不上。因为"什么叫真变"没有客观标准。

本脚本用【客观规则】自动分类每个低δ更新，给出可复现的漏检率：

  类别A·纯形式改动（安全，低δ是对的）：
    去掉空白+统一大小写后 old==new → 只是重命名/大小写/空白/格式。
  类别B·仅标识符改动（较安全）：
    token序列结构相同，只有标识符名字变了（重命名）。
  类别C·实质改动（可能漏检，危险）：
    token结构变了——增删了运算符/关键字/字面量/控制流。

漏检率 = 类别C占低δ更新的比例。这个数客观、可复现、能写进论文。

用法：
    python3 lowdelta_classify.py --data 数据目录 --repo gin --topk 50
"""
import json, argparse, os, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="clean_data_576")
ap.add_argument("--repo", default="gin")
ap.add_argument("--topk", type=int, default=50, help="看δ最低的多少对")
ap.add_argument("--sample", type=int, default=300)
ap.add_argument("--delta-thresh", type=float, default=0.02, help="低δ阈值")
args = ap.parse_args()

upath = os.path.join(args.data, args.repo, "updates.jsonl")
if not os.path.exists(upath):
    print(f"❌ 找不到 {upath}"); sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 先装 sentence-transformers"); sys.exit(1)
m = SentenceTransformer("all-MiniLM-L6-v2")

# ---- token化：把代码拆成 标识符/运算符/关键字/字面量 ----
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|==|!=|<=|>=|&&|\|\||[+\-*/%<>=(){}\[\];,.:&|!]")
# 结构性token（非标识符）：运算符、括号、关键字
KEYWORDS = {"if","else","for","while","return","func","def","switch","case",
            "break","continue","&&","||","==","!=","<=",">=","<",">","+","-","*","/","%"}

def tokens(code):
    return TOKEN_RE.findall(code)

def norm_form(code):
    # 去所有空白 + 统一小写
    return re.sub(r"\s+", "", code).lower()

def structural_tokens(code):
    # 只保留非标识符的结构token（运算符/括号/关键字/数字字面量）
    out = []
    for t in tokens(code):
        if t in KEYWORDS or not re.match(r"^[A-Za-z_]\w*$", t):
            out.append(t)
        elif t in ("true","false","nil","null","None"):
            out.append(t)
        elif re.match(r"^\d", t):
            out.append(t)  # 数字字面量:边界值/缓冲区/超时(0→1,<1→<2)语义重要,必须计入
    return out

def called_identifiers(code):
    # 提取"后面跟着 ( 的标识符" = 函数调用/API名
    return set(re.findall(r"([A-Za-z_][\w\.]*)\s*\(", code))

def classify(old, new):
    if norm_form(old) == norm_form(new):
        return "A_纯形式"  # 去空白+小写后相同 = 重命名/大小写/空白
    so, sn = structural_tokens(old), structural_tokens(new)
    if so == sn:
        # 结构token相同=只动了标识符。但要区分:改的是"变量名"还是"函数调用"?
        # 若被调用的函数集合变了 → 是换实现/换API,语义可能变 → 归为可疑(C')
        if called_identifiers(old) != called_identifiers(new):
            return "C2_疑似换实现"  # 只动标识符但换了被调函数,如 []byte→StringToBytes
        return "B_仅变量名"  # 结构+调用都没变,只是变量重命名,较安全
    return "C1_结构改动"  # 结构token变了:增删运算符/关键字/控制流

# ---- 算δ,取最低topk ----
ups = [json.loads(l) for l in open(upath, encoding="utf-8") if l.strip()]
ups = [u for u in ups if (u.get("old_text") or "").strip() and (u.get("new_text") or "").strip()]
import random; random.seed(42)
if len(ups) > args.sample: ups = random.sample(ups, args.sample)

Eo = m.encode([u["old_text"] for u in ups], show_progress_bar=False)
En = m.encode([u["new_text"] for u in ups], show_progress_bar=False)
for i, u in enumerate(ups):
    cos = np.dot(Eo[i], En[i]) / (np.linalg.norm(Eo[i]) * np.linalg.norm(En[i]) + 1e-9)
    u["_delta"] = 1 - cos

low = sorted(ups, key=lambda u: u["_delta"])[:args.topk]

# ---- 分类统计 ----
CATS = ["A_纯形式", "B_仅变量名", "C2_疑似换实现", "C1_结构改动"]
cats = {c: [] for c in CATS}
for u in low:
    cats[classify(u["old_text"], u["new_text"])].append(u)

print("=" * 60)
print(f"{args.repo}: δ最低的 {len(low)} 对更新，客观规则分类")
print("=" * 60)
n = len(low)
for c in CATS:
    print(f"  {c}: {len(cats[c])} 对 ({len(cats[c])/n*100:.0f}%)")

c_struct = len(cats["C1_结构改动"])
c_impl = len(cats["C2_疑似换实现"])
print(f"\n★ 漏检率(下界) = C1结构改动 {c_struct/n*100:.0f}%")
print(f"★ 漏检率(更真实) = C1 + C2疑似换实现 = {(c_struct+c_impl)/n*100:.0f}%")
print("  C1=结构token变了(增删运算符/控制流); C2=只动标识符但换了被调函数(如[]byte→StringToBytes)")
print("  论文用'至少X%'——因为C2的实现替换是否真变语义,规则也不能100%确定。")

print("\n=== C1(结构改动)例子 ===")
import difflib
def show(u):
    return [l for l in difflib.unified_diff(u["old_text"].splitlines(),
            u["new_text"].splitlines(), lineterm="", n=0)
            if l and l[0] in "+-" and not l.startswith(("+++","---"))][:3]
for u in cats["C1_结构改动"][:5]:
    print(f"  δ={u['_delta']:.3f} {u['unit_id'][:42]}")
    for d in show(u): print(f"     {d}")

print("\n=== C2(疑似换实现,B类里挑出来的)例子——这些最容易被漏 ===")
for u in cats["C2_疑似换实现"][:5]:
    print(f"  δ={u['_delta']:.3f} {u['unit_id'][:42]}")
    for d in show(u): print(f"     {d}")

print("\n" + "=" * 60)
print("怎么用这个数：")
print("  漏检率 = C类占比。这是客观定义(结构token变了)，不靠肉眼判。")
print("  三个语言(gin/flask/express)都跑一遍，比较C类占比：")
print("  - Python若明显低于Go → Go的高C类多是大小写重命名(会被A类吸收)")
print("  - 论文里写：省算判据对δ<{:.2f}的更新中约X%的结构改动敏感度不足，".format(args.delta_thresh))
print("    故报告的省算比例为保守下界。")
