#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格按 README + 学生执行文档 执行完整流程（v4 修复版）：
  闸门 → 扩仓库(7库) → 合并4库 → clean → 验3绿 → 漏检 → distribution → 6模型sweep
所有输出保存到 logs/ 和对应结果目录。
"""
import subprocess, sys, os, shutil, json, time, glob as globmod

# 环境变量
env = os.environ.copy()
env['GITHUB_TOKEN'] = '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'
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','NO_PROXY','no_proxy']:
    env.pop(k, None)
env['HF_ENDPOINT'] = 'https://hf-mirror.com'
env['HF_HUB_OFFLINE'] = '0'

# 目录
PKG = r'c:\Users\26240\Desktop\SemView\EvoBench_final_v4\EvoBench_final'
CLEAN_576 = os.path.join(PKG, '5_data_clean576')
PYTHON = sys.executable
LOGS = os.path.join(PKG, 'logs')
os.makedirs(LOGS, exist_ok=True)

# 脚本路径
SCRIPTS = {
    'probe':  os.path.join(PKG, '3_probes', 'probe_lowdelta_inspect.py'),
    'step1':  os.path.join(PKG, '1_pipeline', 'build_step1_extract_evolution.py'),
    'step2':  os.path.join(PKG, '1_pipeline', 'build_step2_queries_and_gt.py'),
    'step25': os.path.join(PKG, '1_pipeline', 'build_step2p5_to_fourset.py'),
    'clean':  os.path.join(PKG, '1_pipeline', 'clean_benchmark_data.py'),
    'dist':   os.path.join(PKG, '4_expansion', 'check_distribution.py'),
    'sweep4': os.path.join(PKG, '2_experiments', 'build_step4_multimodel.py'),
    'lowdelta': os.path.join(PKG, '3_probes', 'lowdelta_classify_v2.py'),
}

REPOS = [
    ('pytest', 'https://github.com/pytest-dev/pytest.git', 'python'),
    ('click',  'https://github.com/pallets/click.git',    'python'),
    ('pylint', 'https://github.com/pylint-dev/pylint.git', 'python'),
    ('axios',  'https://github.com/axios/axios.git',         'js'),
    ('koa',    'https://github.com/koajs/koa.git',           'js'),
    ('cobra',  'https://github.com/spf13/cobra.git',         'go'),
    ('echo',   'https://github.com/labstack/echo.git',       'go'),
]

def run(name, cmd, log_name, timeout=1800):
    log_path = os.path.join(LOGS, log_name)
    print(f'\n{"="*60}\n[开始] {name}\n{"="*60}')
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'=== {name} ===\n')
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  text=True, env=env, timeout=timeout, cwd=PKG)
        except subprocess.TimeoutExpired:
            f.write('\n[超时]\n')
            proc = type('obj', (object,), {'returncode': -1})
        elapsed = time.time() - t0
        f.write(f'\n[耗时] {elapsed:.1f}s\n[退出码] {proc.returncode}\n')
    status = 'OK' if proc.returncode == 0 else 'FAIL'
    print(f'[完成] {name} ({status}, {elapsed:.1f}s)')
    return proc.returncode

# ========================================================================
# 第0.5步：闸门 — 跨语言抽检（gin/flask/express）
# ========================================================================
print('\n' + '#'*60)
print('# 第0.5步：闸门 — 跨语言抽检 δ≈0 的更新')
print('# 学生执行文档: 先花半天验证，别花一天返工')
print('#'*60)

for repo in ['gin', 'flask', 'express']:
    run(f'闸门: {repo}',
        [PYTHON, '-u', SCRIPTS['probe'],
         '--data', CLEAN_576, '--repo', repo, '--n', '20'],
        f'gate_{repo}.txt', timeout=600)

print('\n★ 闸门日志已保存，请人眼判断后决定是否继续。')
print('   如需继续扩库，按 Enter 继续，或 Ctrl+C 终止。')

# ========================================================================
# 第2步：扩仓库（7新库 step1→2→2.5 + 合并4库 + clean + 验3绿 + 漏检）
# ========================================================================
BENCH_RAW = os.path.join(PKG, 'bench_raw')
BENCH_FINAL = os.path.join(PKG, 'bench_final')
BENCH_ALL_CLEAN = os.path.join(PKG, 'bench_all_clean')
BENCH_ALL_EXPANDED = os.path.join(PKG, 'bench_all_expanded')
SWEEP_FINAL = os.path.join(PKG, 'sweep_final')

# 清理旧数据（保留 logs 和已有 bench_raw/step1 产出）
for d in [BENCH_FINAL, BENCH_ALL_CLEAN, BENCH_ALL_EXPANDED, SWEEP_FINAL]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
os.makedirs(BENCH_RAW, exist_ok=True)

# 2a: 7个新仓库 step1→2→2.5
print('\n--- 2a: 7个新仓库 step1→2→2.5 (v4 修复版，日志应有 [step2 v4] 横幅) ---')
for name, url, lang in REPOS:
    raw_dir = os.path.join(BENCH_RAW, name)
    clone_dir = os.path.join(BENCH_RAW, f'{name}_clone')
    final_dir = os.path.join(BENCH_FINAL, name)

    # skip if already has bench_final with queries
    if os.path.exists(final_dir) and os.path.exists(os.path.join(final_dir, 'queries.txt')):
        qc = sum(1 for l in open(os.path.join(final_dir, 'queries.txt'), encoding='utf-8') if l.strip())
        if qc > 0:
            print(f'  {name}: 已有 bench_final (q={qc})，跳过')
            continue

    # step1
    rc1 = run(f'Step1: {name}',
              [PYTHON, '-u', SCRIPTS['step1'], '--repo', url, '--lang', lang,
               '--out', raw_dir, '--workdir', clone_dir, '--max-commits', '2000'],
              f'{name}_step1.txt', timeout=3600)
    if rc1 != 0:
        print(f'  {name}: step1 失败，跳过')
        continue

    # step2 v4 (clone_dir/name)
    actual_clone = os.path.join(clone_dir, name)
    rc2 = run(f'Step2: {name}',
              [PYTHON, '-u', SCRIPTS['step2'], '--repo-dir', raw_dir, '--repo-url', url,
               '--clone-dir', actual_clone, '--gt-max', '8'],
              f'{name}_step2.txt', timeout=1800)
    if rc2 != 0:
        print(f'  {name}: step2 失败，跳过')
        continue

    # step2.5 (step2-dir = raw_dir/step2)
    rc25 = run(f'Step2.5: {name}',
               [PYTHON, '-u', SCRIPTS['step25'], '--step1-dir', raw_dir,
                '--step2-dir', os.path.join(raw_dir, 'step2'), '--out', final_dir],
               f'{name}_step2p5.txt', timeout=300)
    if rc25 != 0:
        print(f'  {name}: step2.5 失败')
    else:
        qf = os.path.join(final_dir, 'queries.txt')
        qc = sum(1 for l in open(qf, encoding='utf-8') if l.strip()) if os.path.exists(qf) else 0
        print(f'  {name}: 四件套已生成 (queries={qc})')

# 2b: 合并已有4库
print('\n--- 2b: 并入已有4库 ---')
for r in ['flask', 'requests', 'express', 'gin']:
    src = os.path.join(CLEAN_576, r)
    dst = os.path.join(BENCH_FINAL, r)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
        print(f'  并入 {r}')
    elif os.path.exists(dst):
        print(f'  {r}: 已存在')

# 2c: 统一 clean（★ 对父目录跑一次，非每仓库单独）
print('\n--- 2c: 统一清洗（去重+剔离群）---')
if os.path.exists(BENCH_ALL_CLEAN):
    shutil.rmtree(BENCH_ALL_CLEAN)
run('Clean', [PYTHON, '-u', SCRIPTS['clean'],
     '--in', BENCH_FINAL, '--out', BENCH_ALL_CLEAN, '--gt-max', '10'],
    'clean_all.txt', timeout=300)

# 2d: 验3绿 + 收录
print('\n--- 2d: 验3绿 + 收录 ---')
if os.path.exists(BENCH_ALL_EXPANDED):
    shutil.rmtree(BENCH_ALL_EXPANDED)
os.makedirs(BENCH_ALL_EXPANDED, exist_ok=True)

total_q = 0; ok_repos = []
for d in sorted(globmod.glob(os.path.join(BENCH_ALL_CLEAN, '*'))):
    if not os.path.isdir(d): continue
    name = os.path.basename(d)
    try:
        q = sum(1 for l in open(os.path.join(d, 'queries.txt'), encoding='utf-8') if l.strip())
        gt = json.load(open(os.path.join(d, 'ground_truth.json'), encoding='utf-8'))
        gbq = len(json.load(open(os.path.join(d, 'gt_by_qid.json'), encoding='utf-8')))
        qi = sum(1 for l in open(os.path.join(d, 'query_index.jsonl'), encoding='utf-8') if l.strip())
        units = set(json.loads(l)['unit_id'] for l in open(os.path.join(d, 'units.jsonl'), encoding='utf-8') if l.strip())
        tgt = set(x for v in gt.values() for x in v)
        dangling = len(tgt - units)
        over = sum(1 for v in gt.values() if len(v) > 10)
        aligned = (q == len(gt) == gbq == qi)
        ok = aligned and dangling == 0 and over == 0 and q > 0
        flag = '->收录' if ok else '->跳过'
        print(f'  {name:12} q={q:4} 对齐={"OK" if aligned else "BAD"} 悬空={dangling} 离群={over} {flag}')
        if ok:
            shutil.copytree(d, os.path.join(BENCH_ALL_EXPANDED, name), dirs_exist_ok=True)
            total_q += q
            ok_repos.append(name)
    except Exception as e:
        print(f'  {name}: 错误 {e}')
print(f'\n收录 {len(ok_repos)} 个仓库，共 {total_q} 条带GT查询')

# 2e: 每库漏检率
print('\n--- 2e: 每库漏检分类器 ---')
for r in ok_repos:
    run(f'漏检: {r}',
        [PYTHON, '-u', SCRIPTS['lowdelta'], '--data', BENCH_ALL_EXPANDED,
         '--repo', r, '--sample', '300', '--topk', '50', '--delta-thresh', '0.02'],
        f'lowdelta_{r}.txt', timeout=600)

# ========================================================================
# 第3步：check_distribution
# ========================================================================
print('\n' + '#'*60)
print('# 第3步：分布检查（四条达标）')
print('#'*60)

run('check_distribution',
    [PYTHON, '-u', SCRIPTS['dist'], '--dir', BENCH_ALL_EXPANDED],
    'check_distribution.txt', timeout=300)

# ========================================================================
# 第4步：6模型 sweep（README: 不跑9个，砍jina/starencoder/codet5p省1/3算力）
# ========================================================================
print('\n' + '#'*60)
print('# 第4步：6模型 sweep')
print('#'*60)

MODELS = ['minilm', 'mpnet', 'bge', 'codebert', 'graphcodebert', 'unixcoder']
for m in MODELS:
    out_dir = os.path.join(SWEEP_FINAL, m)
    os.makedirs(out_dir, exist_ok=True)
    run(f'Sweep: {m}',
        [PYTHON, '-u', SCRIPTS['sweep4'], '--bench-final', BENCH_ALL_EXPANDED,
         '--out', out_dir, '--model', m],
        f'sweep_{m}.txt', timeout=3600)

# ========================================================================
# 汇总
# ========================================================================
print('\n' + '='*60)
print('全部完成')
print('='*60)
print(f'收录: {ok_repos}')
print(f'总查询: {total_q}')
print(f'bench_all_expanded: {BENCH_ALL_EXPANDED}')
print(f'sweep_final: {SWEEP_FINAL}')
print(f'日志: {LOGS}')