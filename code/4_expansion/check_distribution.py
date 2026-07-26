#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
总体分布检查：扩完仓库后，看是否达到"benchmark做完"的标准
============================================================
标准（四条全绿才算数据部分完成）：
  1. 总带GT查询数 1500-2500
  2. 没有单仓库超过总量30%（治gin独大）
  3. 规模谱覆盖小/中/大档
  4. ≥3种语言

用法：
    python3 check_distribution.py --dir bench_all_expanded
（先把已有的 flask/requests/express/gin 也拷进这个目录）
"""
import json, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="bench_all_expanded")
args = ap.parse_args()

LANG_HINT = {  # 简名→语言，用于语言维度统计
    "flask":"Python","requests":"Python","pytest":"Python","xarray":"Python",
    "pylint":"Python","sphinx":"Python","aiohttp":"Python","scrapy":"Python",
    "pydantic":"Python","click":"Python","gin":"Go","express":"JavaScript",
    "axios":"JavaScript","koa":"JavaScript","fastify":"JavaScript",
    "cobra":"Go","echo":"Go","fiber":"Go","chi":"Go",
}

repos = [d for d in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, d))]
rows = []
total = 0
for r in repos:
    base = os.path.join(args.dir, r)
    try:
        q = len([l for l in open(f"{base}/queries.txt") if l.strip()])
        u = len([l for l in open(f"{base}/units.jsonl") if l.strip()])
        rows.append((r, u, q, LANG_HINT.get(r, "?")))
        total += q
    except Exception as e:
        print(f"⚠️ {r} 读取失败: {e}")

rows.sort(key=lambda x: -x[2])
print(f"{'仓库':12}{'units':>7}{'查询':>7}{'占比':>7}  语言")
print("-" * 45)
for r, u, q, lang in rows:
    pct = q / total * 100 if total else 0
    print(f"{r:12}{u:>7}{q:>7}{pct:>6.1f}%  {lang}")
print("-" * 45)
print(f"{'总计':12}{'':>7}{total:>7}")

# ==== 四条标准检查 ====
print("\n" + "=" * 45)
print("达标检查")
print("=" * 45)

# 1. 总量
c1 = 1500 <= total <= 2500
print(f"1. 总查询 1500-2500: {total} {'✅' if c1 else ('⬆还需加库' if total<1500 else '(超了,可截断大库)')}")

# 2. 单仓库占比
if rows:
    maxpct = rows[0][2] / total if total else 1
    c2 = maxpct < 0.30
    print(f"2. 单仓库<30%: 最大是{rows[0][0]}占{maxpct*100:.0f}% {'✅' if c2 else '🔴还需加库稀释'}")
else:
    c2 = False

# 3. 规模谱
units_list = sorted([u for _, u, _, _ in rows])
small = sum(1 for u in units_list if u < 100)
mid = sum(1 for u in units_list if 100 <= u < 400)
large = sum(1 for u in units_list if u >= 400)
c3 = small >= 1 and mid >= 2 and large >= 1
print(f"3. 规模谱(小/中/大档各有点): 小{small}个 中{mid}个 大{large}个 {'✅' if c3 else '⚠️某档偏空'}")

# 4. 语言
langs = set(lang for _, _, _, lang in rows if lang != "?")
c4 = len(langs) >= 3
print(f"4. ≥3种语言: {sorted(langs)} ({len(langs)}种) {'✅' if c4 else '⚠️语言偏少'}")

print("\n" + "=" * 45)
if c1 and c2 and c3 and c4:
    print("🎉 四条全绿！数据部分完成，可以跑最终 step4 出省算曲线了。")
else:
    print("还差几条，看上面🔴/⚠️项：不够就按候选清单再加中段库。")
    print("（Rust/Java仓库需先给step1加语言支持，暂时先靠Python凑量）")
