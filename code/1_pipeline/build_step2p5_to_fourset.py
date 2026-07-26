#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_step2p5_to_fourset.py
===========================
Benchmark 构造 · 第 2.5 步（衔接）：把 step2 产出转成下游统一「四件套」

为什么需要这一步：
  - step2 产出 queries.jsonl（带 query_id/issue_number 的结构化查询）+ ground_truth.json（key=query_id）。
  - 下游 负载体检.py 要的是 queries.txt（每行一条纯文本查询），且按行序处理。
  - 未来的成本/Recall 测量脚本要按「查询文本」或「行序」查 GT。
  两边格式对不上，直接喂会读空/报错。这一步把接缝补上。

铁律（延续）：
  - 只做格式转换，绝不新增/编造任何查询或 GT。转换前后条数、内容一一对应，可核对。
  - 转换是纯函数式：queries.jsonl 的每一条，原样落一行到 queries.txt，行序 = GT 对齐序。

产出（下游能直接吃）：
  <out>/units.jsonl        （从 step1 复制，原样）
  <out>/updates.jsonl      （从 step1 复制，原样）
  <out>/queries.txt        （每行一条查询文本，行序严格对齐下面两个对照）
  <out>/ground_truth.json  （★ key 改成"查询文本"，供按文本查 GT 的下游用）
  <out>/gt_by_qid.json     （保留 key=query_id 的原始 GT，供按 id 查的下游用）
  <out>/query_index.jsonl  （每行 {line_no, query_id, issue_number, query}，行序对照表）

用法：
  python build_step2p5_to_fourset.py \
      --step1-dir ./bench_raw/click \
      --step2-dir ./bench_raw/click/step2 \
      --out ./bench_final/click

  # 预置多仓库
  python build_step2p5_to_fourset.py --preset
"""
import argparse, os, json, sys, shutil

REPOS = ["flask", "requests", "express", "gin"]


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def process(step1_dir, step2_dir, out_dir):
    name = os.path.basename(out_dir.rstrip("/"))
    print(f"\n{'='*56}\n衔接: {name}\n{'='*56}")

    # --- 校验输入齐备 ---
    u_path = os.path.join(step1_dir, "units.jsonl")
    up_path = os.path.join(step1_dir, "updates.jsonl")
    q_path = os.path.join(step2_dir, "queries.jsonl")
    gt_path = os.path.join(step2_dir, "ground_truth.json")
    for p in (u_path, up_path, q_path, gt_path):
        if not os.path.exists(p):
            print(f"❌ 缺输入 {p}。确认 step1/step2 都跑完。"); return None

    os.makedirs(out_dir, exist_ok=True)

    # --- units / updates 原样复制（不动一个字节）---
    shutil.copy(u_path, os.path.join(out_dir, "units.jsonl"))
    shutil.copy(up_path, os.path.join(out_dir, "updates.jsonl"))
    n_units = sum(1 for _ in open(os.path.join(out_dir, "units.jsonl"), encoding="utf-8"))
    n_updates = sum(1 for _ in open(os.path.join(out_dir, "updates.jsonl"), encoding="utf-8"))

    # --- 读 step2 查询 + GT ---
    queries = load_jsonl(q_path)                        # [{query_id, query, issue_number}, ...]
    gt_by_qid = json.load(open(gt_path, encoding="utf-8"))   # {query_id: [unit_id,...]}

    # --- 生成 queries.txt（每行一条，行序 = 对齐序）+ 对照 + 文本键GT ---
    #     查询里可能含换行，写 txt 时压成单行（下游按行读）。
    def flatten(t):
        return " ".join(t.split())

    lines, index_rows, gt_by_text = [], [], {}
    seen_text = {}
    for i, q in enumerate(queries):
        qid = q["query_id"]
        text = flatten(q["query"])
        lines.append(text)
        index_rows.append({"line_no": i, "query_id": qid,
                           "issue_number": q.get("issue_number"),
                           "link_type": q.get("link_type"),      # ★v3 透传：strong/weak
                           "source_type": q.get("source_type"),  # ★v3 透传：issue/pr
                           "query": text})
        # 文本键 GT：若两条查询文本恰好相同（罕见），合并 GT，避免覆盖丢数据
        gt = gt_by_qid.get(qid, [])
        if text in gt_by_text:
            gt_by_text[text] = sorted(set(gt_by_text[text]) | set(gt))
        else:
            gt_by_text[text] = gt

    with open(os.path.join(out_dir, "queries.txt"), "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    json.dump(gt_by_text, open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    shutil.copy(gt_path, os.path.join(out_dir, "gt_by_qid.json"))
    with open(os.path.join(out_dir, "query_index.jsonl"), "w", encoding="utf-8") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 转换后自检：条数一致 + GT 键都能对上 ---
    ok = True
    if len(lines) != len(queries):
        print(f"  ❌ queries.txt 行数 {len(lines)} != step2 查询数 {len(queries)}"); ok = False
    # 每个 query_id 的 GT，在文本键版本里应能查到且内容一致（集合）
    for q in queries:
        t = flatten(q["query"])
        if set(gt_by_text.get(t, [])) < set(gt_by_qid.get(q["query_id"], [])):
            print(f"  ❌ GT 对齐丢失: {q['query_id']}"); ok = False; break
    # GT 指向的 unit_id 必须都在 units 里（可复现）
    unit_ids = set(json.loads(l)["unit_id"]
                   for l in open(os.path.join(out_dir, "units.jsonl"), encoding="utf-8"))
    dangling = set()
    for gts in gt_by_qid.values():
        for uid in gts:
            if uid not in unit_ids:
                dangling.add(uid)
    if dangling:
        print(f"  ⚠️ 有 {len(dangling)} 个 GT unit_id 不在 units 里（step2 应已过滤，检查）")

    meta = {"repo": name, "units": n_units, "updates": n_updates,
            "queries": len(lines), "gt_queries": len(gt_by_text),
            "dangling_gt": len(dangling),
            "selfcheck_pass": ok and not dangling,
            "synthetic_data": False,
            "note": "格式转换 only；未新增/编造任何查询或 GT"}
    json.dump(meta, open(os.path.join(out_dir, "meta_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  units {n_units} | updates {n_updates} | queries {len(lines)} | GT {len(gt_by_text)}")
    print(f"  自检 {'✅ 通过' if meta['selfcheck_pass'] else '❌ 未过'} -> {out_dir}")
    print(f"  下游可直接：python 负载体检.py --units {out_dir}/units.jsonl "
          f"--updates {out_dir}/updates.jsonl --queries {out_dir}/queries.txt "
          f"--repo-name {name} --purpose measurement")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-dir")
    ap.add_argument("--step2-dir")
    ap.add_argument("--out")
    ap.add_argument("--preset", action="store_true")
    ap.add_argument("--bench-root", default="./bench_raw")
    ap.add_argument("--final-root", default="./bench_final")
    args = ap.parse_args()

    metas = []
    if args.preset:
        for name in REPOS:
            s1 = os.path.join(args.bench_root, name)
            s2 = os.path.join(s1, "step2")
            metas.append(process(s1, s2, os.path.join(args.final_root, name)))
    else:
        if not (args.step1_dir and args.step2_dir and args.out):
            print("❌ 需 --step1-dir --step2-dir --out，或用 --preset"); sys.exit(1)
        metas.append(process(args.step1_dir, args.step2_dir, args.out))

    metas = [m for m in metas if m]
    passed = sum(m["selfcheck_pass"] for m in metas)
    print(f"\n汇总：{passed}/{len(metas)} 仓库转换自检通过。")
    print("现在四件套对齐下游，可跑 负载体检.py 和未来的成本/Recall 测量。")


if __name__ == "__main__":
    main()
