#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_benchmark_data.py
=======================
清洗四件套数据，解决审查发现的两个必修问题：
  问题1：queries.txt 有重复行 → 去重（真实唯一查询数才准）
  问题2：GT 离群值（一个查询指向几十个函数，来自大改动 commit）→ 按阈值过滤

不改原始数据，输出到新目录。同时产出清洗报告 + GT 精度抽检样本。

用法：
  python clean_benchmark_data.py --in ./bench_final --out ./bench_final_clean --gt-max 10

参数：
  --gt-max N   GT 相关函数数 > N 的查询视为离群，剔除（默认10，对应top-k=10）
               设 0 表示不过滤离群、只去重。

铁律：只删不造。去重和过滤都是删除操作，不新增/编造任何数据。
"""
import argparse, os, json, glob, shutil, random


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def clean_repo(rd, out_rd, gt_max):
    name = os.path.basename(rd.rstrip("/"))
    os.makedirs(out_rd, exist_ok=True)

    # units / updates 原样复制（这两个没问题，审查已验证真实）
    shutil.copy(os.path.join(rd, "units.jsonl"), os.path.join(out_rd, "units.jsonl"))
    shutil.copy(os.path.join(rd, "updates.jsonl"), os.path.join(out_rd, "updates.jsonl"))

    # 读 queries + GT
    q_raw = [l.rstrip("\n") for l in open(os.path.join(rd, "queries.txt"), encoding="utf-8") if l.strip()]
    gt = json.load(open(os.path.join(rd, "ground_truth.json"), encoding="utf-8"))

    # --- 问题1：queries 去重（保序）---
    seen = set(); q_dedup = []
    for q in q_raw:
        if q not in seen:
            seen.add(q); q_dedup.append(q)
    n_dup = len(q_raw) - len(q_dedup)

    # --- 问题2：GT 离群过滤 ---
    outliers = {}
    gt_clean = {}
    for q, units in gt.items():
        if gt_max > 0 and len(units) > gt_max:
            outliers[q] = len(units)      # 记录被剔的离群查询
        else:
            gt_clean[q] = units

    # queries 只保留仍有 GT 的（去重 + 非离群）
    q_final = [q for q in q_dedup if q in gt_clean]

    # 写出核心两件
    with open(os.path.join(out_rd, "queries.txt"), "w", encoding="utf-8") as f:
        for q in q_final:
            f.write(q + "\n")
    json.dump(gt_clean, open(os.path.join(out_rd, "ground_truth.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ★ 修复 bug1：附属文件必须跟着过滤，不能原样 copy（否则脏数原路带回，清洗白做）
    q_final_set = set(q_final)

    # query_index.jsonl：每行 {line_no, query_id, issue_number, query}，按 query 文本过滤
    qi_path = os.path.join(rd, "query_index.jsonl")
    qid_keep = set()          # 记录保留下来的 query_id，供 gt_by_qid 过滤
    text2qid = {}
    if os.path.exists(qi_path):
        kept_rows = []
        seen_qtext = set()          # ★ 按 query 文本去重，每个文本只留第一行（与 queries.txt 一致）
        for row in load_jsonl(qi_path):
            qt = row.get("query")
            if qt in q_final_set and qt not in seen_qtext:
                seen_qtext.add(qt)
                kept_rows.append(row)
                qid_keep.add(row.get("query_id"))
                text2qid[qt] = row.get("query_id")
        # 重新编 line_no，保持与清洗后 queries.txt 行序一致
        text2lineno = {q: i for i, q in enumerate(q_final)}
        kept_rows.sort(key=lambda r: text2lineno.get(r.get("query"), 1e9))
        for i, r in enumerate(kept_rows):
            r["line_no"] = i
        with open(os.path.join(out_rd, "query_index.jsonl"), "w", encoding="utf-8") as f:
            for r in kept_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # gt_by_qid.json：key 是 query_id，按保留的 qid 过滤
    qid_path = os.path.join(rd, "gt_by_qid.json")
    if os.path.exists(qid_path):
        gt_qid_raw = json.load(open(qid_path, encoding="utf-8"))
        if qid_keep:
            gt_qid_clean = {k: v for k, v in gt_qid_raw.items() if k in qid_keep}
        else:
            # 没有 query_index 兜底：无法映射 qid，安全起见按 GT 单元数同样剔离群
            gt_qid_clean = {k: v for k, v in gt_qid_raw.items()
                            if not (gt_max > 0 and len(v) > gt_max)}
        json.dump(gt_qid_clean, open(os.path.join(out_rd, "gt_by_qid.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    return {
        "repo": name,
        "queries_raw": len(q_raw),
        "queries_dedup": len(q_dedup),
        "duplicates_removed": n_dup,
        "gt_outliers_removed": len(outliers),
        "outlier_sizes": sorted(outliers.values(), reverse=True)[:5],
        "queries_final": len(q_final),
        "gt_final": len(gt_clean),
    }


def sample_gt_for_audit(out_root, n=20):
    """从清洗后数据抽 n 条 GT 供人工精度抽检。
    ★ 修复 bug2：带上 issue 号 + commit SHA + GitHub 链接，让人能去核对，不再是摆设。"""
    samples = []
    for rd in sorted(glob.glob(os.path.join(out_root, "*"))):
        if not os.path.isdir(rd):
            continue
        gtp = os.path.join(rd, "ground_truth.json")
        if not os.path.exists(gtp):
            continue
        repo = os.path.basename(rd)
        gt = json.load(open(gtp, encoding="utf-8"))
        # 建 query文本 -> issue号 映射（从 query_index）
        text2issue = {}
        qi = os.path.join(rd, "query_index.jsonl")
        if os.path.exists(qi):
            for row in load_jsonl(qi):
                text2issue[row.get("query")] = row.get("issue_number")
        # 建 unit_id -> to_commit 映射（从 updates，供核对函数改动）
        uid2commit = {}
        up = os.path.join(rd, "updates.jsonl")
        if os.path.exists(up):
            for u in load_jsonl(up):
                uid2commit[u.get("unit_id")] = u.get("to_commit")
        # 仓库全名（供拼 GitHub URL）
        owner = {"flask": "pallets/flask", "requests": "psf/requests",
                 "express": "expressjs/express", "gin": "gin-gonic/gin"}.get(repo, repo)
        items = list(gt.items())
        random.seed(42)
        for q, units in random.sample(items, min(5, len(items))):
            issue = text2issue.get(q)
            samples.append({
                "repo": repo,
                "query": q[:80],
                "issue_number": issue,
                "issue_url": f"https://github.com/{owner}/issues/{issue}" if issue else None,
                "gt_units": units[:5],
                "gt_count": len(units),
                "gt_unit_commits": {u: uid2commit.get(u) for u in units[:3]},
                "如何核对": "打开 issue_url 看问题，再看 gt_units 里的函数是不是这个 issue 的修复真改的"
            })
    return samples[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="./bench_final")
    ap.add_argument("--out", default="./bench_final_clean")
    ap.add_argument("--gt-max", type=int, default=10,
                    help="GT 函数数 > 此值的查询视为离群剔除（默认10=top-k）；0=只去重不过滤")
    args = ap.parse_args()

    repos = sorted(d for d in glob.glob(os.path.join(args.inp, "*"))
                   if os.path.isdir(d) and os.path.exists(os.path.join(d, "units.jsonl")))
    if not repos:
        print(f"❌ {args.inp} 下无四件套"); return

    os.makedirs(args.out, exist_ok=True)
    print("="*66)
    print(f"数据清洗：去重 queries + 过滤 GT 离群(>{args.gt_max}个函数)")
    print("="*66)
    reports = []
    for rd in repos:
        r = clean_repo(rd, os.path.join(args.out, os.path.basename(rd)), args.gt_max)
        reports.append(r)
        print(f"\n{r['repo']}:")
        print(f"  queries: {r['queries_raw']} → 去重 {r['queries_dedup']}"
              f"（删重复 {r['duplicates_removed']}）→ 过滤离群后 {r['queries_final']}")
        print(f"  GT: 剔除离群查询 {r['gt_outliers_removed']} 个"
              f"（离群大小 {r['outlier_sizes']}）→ 最终 {r['gt_final']}")

    # 汇总规模表
    print("\n" + "="*66)
    print("清洗后规模表（用这个数填论文，别用含重复的行数）")
    print("="*66)
    print(f"  {'仓库':10} {'原queries行':10} {'去重':6} {'剔离群':7} {'最终查询':8}")
    tot_final = 0
    for r in reports:
        print(f"  {r['repo']:10} {r['queries_raw']:10} {r['queries_dedup']:6} "
              f"{r['gt_outliers_removed']:7} {r['queries_final']:8}")
        tot_final += r['queries_final']
    print(f"  {'合计':10} {'':10} {'':6} {'':7} {tot_final:8}")

    # GT 精度抽检样本
    samples = sample_gt_for_audit(args.out)
    json.dump({"reports": reports, "gt_max": args.gt_max,
               "total_final_queries": tot_final,
               "重要说明": {
                   "离群过滤≠噪声提纯": "本脚本剔除的是'一次改几十个函数的大commit'(离群)。"
                       "它解决不了'一个commit改7个函数、但只有部分与issue真相关'的commit内噪声"
                       "(如flask template_filter条混进is_prime)。GT精度问题需人工抽检下面的样本，"
                       "别以为跑了清洗GT就100%干净。",
                   "去重口径": "精确文本匹配。标题高度相似但不完全相同不会去重。",
               },
               "audit_samples": samples},
              open(os.path.join(args.out, "cleaning_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n清洗后数据 → {args.out}")
    print(f"清洗报告 + GT精度抽检样本 → {args.out}/cleaning_report.json")
    print(f"\n★ 用清洗后的数据跑 step3/step4，避免离群值污染判别力表。")
    print(f"★ 论文里带GT查询数用【{tot_final}】(清洗后)，不是 629。")


if __name__ == "__main__":
    main()
