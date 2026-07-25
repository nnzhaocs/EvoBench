#!/usr/bin/env python3
"""合并 3 台实例的 bench_final + 已有 4 库 → clean → 验 3 绿 → 漏检 → dist → sweep"""
import subprocess, sys, os, shutil, json, glob

PKG = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# 各实例 bench_final 拷贝到此目录
MERGED_BENCH = os.path.join(PKG, 'bench_final_merged')
CLEAN = os.path.join(PKG, 'bench_all_clean')
EXPANDED = os.path.join(PKG, 'bench_all_expanded')
SWEEP = os.path.join(PKG, 'sweep_final')
LOGS = os.path.join(PKG, 'logs_merge')
for d in [MERGED_BENCH, CLEAN, EXPANDED, SWEEP, LOGS]:
    os.makedirs(d, exist_ok=True)

SCRIPTS = {
    'clean': os.path.join(PKG, '1_pipeline', 'clean_benchmark_data.py'),
    'dist': os.path.join(PKG, '4_expansion', 'check_distribution.py'),
    'sweep4': os.path.join(PKG, '2_experiments', 'build_step4_multimodel.py'),
    'lowdelta': os.path.join(PKG, '3_probes', 'lowdelta_classify_v2.py'),
}

# 已有 4 库
EXISTING_CLEAN = os.path.join(PKG, '5_data_clean576')

# 合并步骤
print("=" * 60)
print("合并各实例 bench_final 到 bench_final_merged/")
print("=" * 60)

# 已有 4 库
for r in ['flask', 'requests', 'express', 'gin']:
    src = os.path.join(EXISTING_CLEAN, r)
    dst = os.path.join(MERGED_BENCH, r)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
        print(f"  并入已有 {r}")

# 实例产出（从 bench_final_A/ bench_final_B/ bench_final_C/ 拷贝）
for inst in ['A', 'B', 'C']:
    src_base = os.path.join(PKG, f'bench_final_{inst}')
    if not os.path.exists(src_base):
        print(f"  ⚠️ 找不到 bench_final_{inst}/，跳过")
        continue
    for name in os.listdir(src_base):
        src = os.path.join(src_base, name)
        dst = os.path.join(MERGED_BENCH, name)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"  并入实例 {inst}: {name}")

# clean
print("\n统一清洗...")
if os.path.exists(CLEAN):
    shutil.rmtree(CLEAN)
subprocess.run([PYTHON, '-u', SCRIPTS['clean'], '--in', MERGED_BENCH, '--out', CLEAN, '--gt-max', '10'])

# 验 3 绿 + 收录
print("\n验 3 绿 + 收录...")
if os.path.exists(EXPANDED):
    shutil.rmtree(EXPANDED)
os.makedirs(EXPANDED, exist_ok=True)
total_q = 0; ok_repos = []
for d in sorted(glob.glob(f"{CLEAN}/*")):
    if not os.path.isdir(d): continue
    name = os.path.basename(d)
    try:
        q = sum(1 for l in open(os.path.join(d, 'queries.txt')) if l.strip())
        gt = json.load(open(os.path.join(d, 'ground_truth.json')))
        qi = sum(1 for l in open(os.path.join(d, 'query_index.jsonl')) if l.strip())
        aligned = (q == len(gt) == qi)
        dangling = len(set(x for v in gt.values() for x in v) -
                       set(json.loads(l)['unit_id'] for l in open(os.path.join(d, 'units.jsonl')) if l.strip()))
        over = sum(1 for v in gt.values() if len(v) > 10)
        ok = aligned and dangling == 0 and over == 0 and q > 0
        print(f"  {name:12} q={q:4} 对齐={'OK' if aligned else 'BAD'} 悬空={dangling} 离群={over} {'->收录' if ok else '->跳过'}")
        if ok:
            shutil.copytree(d, os.path.join(EXPANDED, name), dirs_exist_ok=True)
            total_q += q; ok_repos.append(name)
    except Exception as e:
        print(f"  {name}: 错误 {e}")
print(f"\n收录 {len(ok_repos)} 个仓库，共 {total_q} 条查询")

# 漏检
print("\n每库漏检率...")
for r in ok_repos:
    print(f"--- {r} ---")
    subprocess.run([PYTHON, '-u', SCRIPTS['lowdelta'], '--data', EXPANDED, '--repo', r,
                    '--sample', '300', '--topk', '50', '--delta-thresh', '0.02'])

# distribution
print("\n分布检查...")
subprocess.run([PYTHON, '-u', SCRIPTS['dist'], '--dir', EXPANDED])

# sweep
print("\n6 模型 sweep...")
for m in ['minilm', 'mpnet', 'bge', 'codebert', 'graphcodebert', 'unixcoder']:
    out_dir = os.path.join(SWEEP, m)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- {m} ---")
    subprocess.run([PYTHON, '-u', SCRIPTS['sweep4'], '--bench-final', EXPANDED, '--out', out_dir, '--model', m])

print("\n" + "=" * 60)
print("全部完成")
print(f"bench_all_expanded: {EXPANDED}")
print(f"sweep_final: {SWEEP}")
