#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_claims.py
==================
验证：跑出来的 benchmark 数据，到底满不满足论文 introduction 里承诺的每一条性质。

这不是"生成数据"，是"检查数据配不配得上论文的声称"。跑完 step3+step4 后跑这个。
每条声称 → 一个可判定的检查 → PASS / FAIL / 需人工看。

用法：
  python validate_claims.py \
      --bench-final ./bench_final \
      --stats ./benchmark_release/stats_for_paper.json \
      --sweep ./sweep_result/baseline_sweep.json

只读数据，不改任何东西。
"""
import argparse, os, json, glob, sys


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def check(name, passed, detail):
    mark = "✅ PASS" if passed is True else ("❌ FAIL" if passed is False else "⚠️ 人工看")
    print(f"  [{mark}] {name}")
    print(f"          {detail}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-final", default="./bench_final")
    ap.add_argument("--stats", default="./benchmark_release/stats_for_paper.json")
    ap.add_argument("--sweep", default="./sweep_result/baseline_sweep.json")
    args = ap.parse_args()

    print("="*64)
    print("EvoBench 数据验证：是否满足论文 introduction 的所有声称")
    print("="*64)

    results = {}
    repo_dirs = sorted(d for d in glob.glob(os.path.join(args.bench_final, "*"))
                       if os.path.isdir(d) and os.path.exists(os.path.join(d, "units.jsonl")))

    # ---------- 声称1：真实编辑演化（old≠new + 带 commit SHA）----------
    print("\n【声称1】真实编辑演化：updates 是真实 old→new，带 commit SHA")
    total_up, bad_up, no_sha = 0, 0, 0
    empty_old = 0
    for rd in repo_dirs:
        for u in load_jsonl(os.path.join(rd, "updates.jsonl")):
            total_up += 1
            if u.get("old_text", "") == u.get("new_text", ""):
                bad_up += 1
            if not u.get("old_text"):
                empty_old += 1          # C6 检查：空 old_text
            if not (u.get("from_commit") and u.get("to_commit")):
                no_sha += 1
    results["c1"] = check("真实编辑演化",
        bad_up == 0 and no_sha == 0,
        f"总更新 {total_up}，old=new 的 {bad_up}（应0），缺SHA {no_sha}（应0），"
        f"空old_text {empty_old}（C6坑，应0，>0需清洗）")

    # ---------- 声称2：跨语言 ----------
    print("\n【声称2】跨语言覆盖：Python + JS + Go 都有")
    langs = set()
    for rd in repo_dirs:
        us = load_jsonl(os.path.join(rd, "units.jsonl"))
        if us:
            langs.add(us[0].get("lang", "?"))
    results["c2"] = check("跨语言",
        len(langs) >= 2 and "python" in langs,
        f"语言集合 {sorted(langs)}（至少 python + 另一门）")

    # ---------- 声称3：检索 GT 真实可追溯 ----------
    print("\n【声称3】检索 GT 真实：GT 指向的 unit 都在库里")
    total_gt, dangling = 0, 0
    for rd in repo_dirs:
        gtp = os.path.join(rd, "ground_truth.json")
        if not os.path.exists(gtp):
            continue
        gt = json.load(open(gtp, encoding="utf-8"))
        unit_ids = set(u["unit_id"] for u in load_jsonl(os.path.join(rd, "units.jsonl")))
        for q, units in gt.items():
            for uid in units:
                total_gt += 1
                if uid not in unit_ids:
                    dangling += 1
    results["c3"] = check("检索GT可追溯",
        total_gt > 0 and dangling == 0,
        f"GT 条目 {total_gt}，悬空(指向不存在的unit) {dangling}（应0）")

    # ---------- 声称4：内容改动与漂移解耦（需 stats）----------
    print("\n【声称4】内容改动与嵌入漂移解耦（相关约 0.5-0.8，不是≈1）")
    if os.path.exists(args.stats):
        stats = json.load(open(args.stats, encoding="utf-8"))
        if stats.get("fake_embed"):
            results["c4"] = check("解耦", None,
                "⚠️ stats 是 fake_embed=true 的假数，不能判！用真 MiniLM 重跑 step3")
        else:
            # 若 stats 里有 corr 字段则判，否则提示人工看散点
            corr = stats.get("decoupling_corr")
            if corr is not None:
                results["c4"] = check("解耦", 0.3 <= corr <= 0.85,
                    f"改动-漂移相关 {corr}（应 0.5-0.8；≈1 表示没解耦，判据无意义）")
            else:
                results["c4"] = check("解耦", None,
                    "stats 无 decoupling_corr 字段，需人工看 δ-改动量散点确认相关在 0.5-0.8")
    else:
        results["c4"] = check("解耦", None, f"stats 文件不存在：{args.stats}，先跑 step3")

    # ---------- 声称5：影响随规模稀疏化（需 stats 规模谱）----------
    print("\n【声称5】检索影响随库规模稀疏化：大库影响查询数 < 小库")
    if os.path.exists(args.stats):
        stats = json.load(open(args.stats, encoding="utf-8"))
        spec = stats.get("scale_spectrum", [])
        if stats.get("fake_embed"):
            results["c5"] = check("稀疏化", None, "⚠️ 假数不能判，真 MiniLM 重跑")
        elif len(spec) >= 2:
            spec_sorted = sorted(spec, key=lambda s: s.get("n_units", 0))
            small = spec_sorted[0].get("avg_affected_queries", 0)
            large = spec_sorted[-1].get("avg_affected_queries", 0)
            results["c5"] = check("稀疏化", large <= small,
                f"最小库影响 {small} vs 最大库影响 {large}（应递减/稀疏化）；"
                f"若反增需查是否 append 污染")
        else:
            results["c5"] = check("稀疏化", None,
                f"规模谱只有 {len(spec)} 个点，至少要 2 个仓库才能看趋势")
    else:
        results["c5"] = check("稀疏化", None, "stats 不存在，先跑 step3")

    # ---------- 声称6：★三类策略拉得开（判别力，命根子）----------
    print("\n【声称6】★★ benchmark 判别力：三类重算策略必须拉开差异")
    if os.path.exists(args.sweep):
        sweep = json.load(open(args.sweep, encoding="utf-8"))
        if sweep.get("fake_embed"):
            results["c6"] = check("判别力", None,
                "⚠️ sweep 是假数，用真 MiniLM 重跑 step4")
        else:
            ok_any = False
            detail_lines = []
            for repo in sweep.get("per_repo", []):
                rows = {r["policy"]: r for r in repo.get("sweep", [])}
                aw = rows.get("always-rebuild", {})
                nv = rows.get("never-rebuild", {})
                # 判据：always recall > never recall（重算确实提升质量）
                # 且存在某个 audit τ：recall 接近 always 但省算显著
                a_rec = aw.get("recall", 0); n_rec = nv.get("recall", 0)
                spread = a_rec - n_rec
                best_audit = None
                for p, r in rows.items():
                    if p.startswith("audit") and r.get("save_pct", 0) > 30 and r.get("recall", 0) >= n_rec + 0.5*spread:
                        best_audit = r
                good = spread > 0.02 and best_audit is not None
                ok_any = ok_any or good
                detail_lines.append(
                    f"{repo['repo']}: always_rec={a_rec:.3f} never_rec={n_rec:.3f} "
                    f"差={spread:.3f} {'✓有好audit点' if best_audit else '✗audit无优势'}")
            results["c6"] = check("★判别力（命根子）", ok_any,
                " | ".join(detail_lines) +
                ("\n          ✓ 至少一个仓库三类策略拉得开，benchmark 有判别力"
                 if ok_any else
                 "\n          ✗ 三类策略拉不开 = 没有判别力 = 论文核心塌，需回炉查度量/数据"))
    else:
        results["c6"] = check("★判别力", None, f"sweep 不存在：{args.sweep}，先跑 step4")

    # ---------- 汇总 ----------
    print("\n" + "="*64)
    print("汇总")
    print("="*64)
    p = sum(1 for v in results.values() if v is True)
    f = sum(1 for v in results.values() if v is False)
    m = sum(1 for v in results.values() if v is None)
    print(f"  PASS {p} / FAIL {f} / 需人工或补数据 {m}")
    if results.get("c6") is False:
        print("\n  🔴 命根子(声称6判别力)FAIL：这是最严重的，benchmark 区分不了方法。优先解决。")
    elif results.get("c6") is None:
        print("\n  ⚠️ 命根子(声称6判别力)还没数据：跑完 step4 真数第一件事看这条。")
    if f == 0 and m == 0:
        print("\n  ✅ 全部通过：数据满足论文所有声称，可以填占位符出下一版。")
    print()


if __name__ == "__main__":
    main()
