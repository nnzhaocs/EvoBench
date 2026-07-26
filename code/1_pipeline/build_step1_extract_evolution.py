#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_step1_extract_evolution.py
================================
Benchmark 构造 · 第一步：从真实开源仓库抽取「函数级演化流」

铁律（与老师要求一致，代码内强制执行）：
  1. 只用真实 git 历史，绝不合成、绝不仿真。
  2. 只用真实 embedding（MiniLM），绝不 --fake-embed。
  3. 规模要够：默认要求每个仓库至少 MIN_CHAINS 条真实演化链
     （同一函数被真实修改 >=2 次），达不到就报警，不凑数。

这一步产出什么：
  对每个目标仓库，遍历真实 commit 历史，抽出「同一个函数在时间线上的多个版本」，
  形成 units（基准版本函数）+ updates（相邻版本的真实 old->new 编辑）。
  这是 benchmark 的「编辑演化」骨架——审计表证明这是现有负载最缺的东西。

  检索查询 + GT 在第二步（build_step2）做；本步只负责演化流。

用法：
  python build_step1_extract_evolution.py \
      --repo https://github.com/pallets/flask.git \
      --lang python \
      --max-commits 800 \
      --out ./bench_raw/flask

  # 一次多个仓库（推荐，配置在 REPOS 里）
  python build_step1_extract_evolution.py --preset

环境：
  pip install GitPython
  需要能 git clone（网络访问 github）。不需要 GPU、不需要 HF——这一步不算 embedding。
"""
import argparse, os, re, json, sys, hashlib, tempfile, shutil
from collections import defaultdict

try:
    import git  # GitPython
except ImportError:
    print("❌ 需要 GitPython：pip install GitPython"); sys.exit(1)

# ---------- 铁律阈值：达不到就报警，不凑数 ----------
MIN_CHAINS_PER_REPO = 30      # 每个仓库至少要有这么多条真实演化链，否则这个仓库不合格
MIN_VERSIONS_PER_CHAIN = 2    # 一条链至少 2 个版本（即函数至少被真实改过 1 次）
MAX_FUNC_LINES = 400          # 跳过超长函数（多半是数据表/生成代码）
SEED = 42                     # 仅用于可复现的抽样，不用于任何"造数据"

# ---------- 预置目标仓库（真实、活跃、许可干净、有长演化史）----------
# 覆盖多语言 + 多规模，直接对着审计表的空缺去补。
REPOS = [
    # (url, lang, license, 说明)
    ("https://github.com/pallets/flask.git",        "python", "BSD-3",   "中型 web 框架，演化史长"),
    ("https://github.com/psf/requests.git",         "python", "Apache-2","HTTP 库，超长演化史"),
    ("https://github.com/expressjs/express.git",    "js",     "MIT",     "跨语言：Node web 框架"),
    ("https://github.com/gin-gonic/gin.git",        "go",     "MIT",     "跨语言：Go web 框架"),
]

# ---------- 各语言的函数边界正则（够用即可，抽的是真实代码）----------
FUNC_PATTERNS = {
    "python": re.compile(r'^(\s*)def\s+([A-Za-z_]\w*)\s*\(', re.M),
    "js":     re.compile(r'^(\s*)(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(', re.M),
    "go":     re.compile(r'^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(', re.M),
}
FILE_EXT = {"python": ".py", "js": ".js", "go": ".go"}


def is_test_or_noise(path):
    """剔除测试文件/生成文件污染（审计里踩过：~31% 是 test）。"""
    p = path.lower()
    return ("test" in p or "/tests/" in p or p.endswith("_test.go")
            or "spec" in p or "vendor/" in p or "node_modules/" in p
            or ".min.js" in p or "generated" in p)


def extract_functions(text, lang):
    """从一个文件内容里抽出 {func_name: func_body}。真实代码，无合成。"""
    pat = FUNC_PATTERNS.get(lang)
    if not pat:
        return {}
    funcs = {}
    lines = text.split("\n")
    matches = list(pat.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(2) if lang != "go" else m.group(1)
        start_pos = m.start()
        start_line = text[:start_pos].count("\n")
        # 函数体：从这一行到下一个同级函数/文件尾（简单但对真实代码够用）
        if lang == "python":
            indent = len(m.group(1))
            body_lines = [lines[start_line]]
            for ln in lines[start_line + 1:]:
                if ln.strip() == "":
                    body_lines.append(ln); continue
                cur_indent = len(ln) - len(ln.lstrip())
                if cur_indent <= indent and ln.strip():
                    break
                body_lines.append(ln)
            body = "\n".join(body_lines)
        else:
            # 花括号语言：从 { 计数到配平
            end_line = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start_pos:end_line]
            depth = 0; cut = len(body)
            for j, ch in enumerate(body):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cut = j + 1; break
            body = body[:cut]
        if body.count("\n") <= MAX_FUNC_LINES and len(body.strip()) > 0:
            funcs[name] = body
    return funcs


def process_repo(url, lang, lic, note, workdir, max_commits, out_dir):
    print(f"\n{'='*60}\n仓库: {url}\n语言: {lang} | 许可: {lic} | {note}\n{'='*60}")
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_path = os.path.join(workdir, name)
    if not os.path.exists(clone_path):
        print(f"[clone] 真实克隆 {url} ...（这是真仓库，非仿真）")
        git.Repo.clone_from(url, clone_path)  # 真实克隆
    repo = git.Repo(clone_path)

    ext = FILE_EXT[lang]
    # 时间正序遍历真实 commit
    commits = list(repo.iter_commits('HEAD', max_count=max_commits))
    commits.reverse()
    print(f"[history] 真实 commit 数（截取）: {len(commits)}")

    # func_key -> [(commit_idx, commit_sha, date, body)]  记录同一函数的真实版本序列
    chains = defaultdict(list)
    prev_blobs = {}  # path -> text at previous commit (to detect real changes)

    for ci, commit in enumerate(commits):
        if ci % 100 == 0:
            print(f"  扫描 commit {ci}/{len(commits)} ...")
        try:
            tree = commit.tree
        except Exception:
            continue
        # 只看这次 commit 真实改动的文件（diff），避免全树扫描的开销与噪声
        if not commit.parents:
            changed = [b.path for b in tree.traverse()
                       if getattr(b, 'type', '') == 'blob' and b.path.endswith(ext)]
        else:
            diffs = commit.parents[0].diff(commit)
            changed = [d.b_path for d in diffs
                       if d.b_path and d.b_path.endswith(ext)]
        for path in changed:
            if is_test_or_noise(path):
                continue
            try:
                blob = commit.tree / path
                text = blob.data_stream.read().decode("utf-8", errors="ignore")
            except Exception:
                continue
            funcs = extract_functions(text, lang)
            for fname, body in funcs.items():
                key = f"{path}::{fname}"
                # 只在函数体真实变化时记一个新版本（去掉纯空白差异）
                norm = re.sub(r'\s+', ' ', body).strip()
                if chains[key] and re.sub(r'\s+', ' ', chains[key][-1][3]).strip() == norm:
                    continue  # 没真变，不记
                chains[key].append((ci, commit.hexsha[:12],
                                    commit.committed_datetime.isoformat(), body))

    # 只保留真实演化链（版本数 >= MIN_VERSIONS_PER_CHAIN）
    real_chains = {k: v for k, v in chains.items() if len(v) >= MIN_VERSIONS_PER_CHAIN}
    print(f"[chains] 函数总数 {len(chains)} | 真实演化链(>= {MIN_VERSIONS_PER_CHAIN} 版) {len(real_chains)}")

    # ---------- 铁律校验：链数不够就报警，不凑数 ----------
    if len(real_chains) < MIN_CHAINS_PER_REPO:
        print(f"  ⚠️ 真实演化链只有 {len(real_chains)} 条 < 要求 {MIN_CHAINS_PER_REPO}。")
        print(f"     → 这个仓库演化不够，要么加大 --max-commits，要么换更活跃的仓库。")
        print(f"     → 绝不允许通过合成/复制来凑链数。")

    # ---------- 产出 units + updates（真实 old->new）----------
    os.makedirs(out_dir, exist_ok=True)
    units, updates = [], []
    for key, versions in real_chains.items():
        # units = 每条链的第一个真实版本（基准）
        base_ci, base_sha, base_date, base_body = versions[0]
        uid = f"{name}::{key}"
        units.append({"unit_id": uid, "text": base_body, "repo": name,
                      "lang": lang, "license": lic,
                      "base_commit": base_sha, "base_date": base_date,
                      "n_versions": len(versions)})
        # updates = 每对相邻真实版本的编辑
        for (c0, s0, d0, b0), (c1, s1, d1, b1) in zip(versions, versions[1:]):
            updates.append({"unit_id": uid, "old_text": b0, "new_text": b1,
                            "update_type": "edit", "repo": name, "lang": lang,
                            "from_commit": s0, "to_commit": s1,
                            "from_date": d0, "to_date": d1,
                            "commit_idx": c1})

    with open(os.path.join(out_dir, "units.jsonl"), "w", encoding="utf-8") as f:
        for u in units: f.write(json.dumps(u, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "updates.jsonl"), "w", encoding="utf-8") as f:
        for u in updates: f.write(json.dumps(u, ensure_ascii=False) + "\n")

    meta = {"repo": name, "url": url, "lang": lang, "license": lic,
            "commits_scanned": len(commits),
            "total_funcs": len(chains), "real_evolution_chains": len(real_chains),
            "units": len(units), "updates": len(updates),
            "min_chains_required": MIN_CHAINS_PER_REPO,
            "chains_ok": len(real_chains) >= MIN_CHAINS_PER_REPO,
            "synthetic_data": False, "fake_embed": False}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[out] units {len(units)} | updates(真实编辑对) {len(updates)} -> {out_dir}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="单个仓库 git url")
    ap.add_argument("--lang", choices=list(FUNC_PATTERNS.keys()))
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--max-commits", type=int, default=800,
                    help="截取多少条真实 commit（越大演化链越多，别设太小凑不够链）")
    ap.add_argument("--out", default="./bench_raw/repo")
    ap.add_argument("--preset", action="store_true", help="跑预置的多仓库清单")
    ap.add_argument("--workdir", default=None, help="克隆目录（默认临时目录）")
    args = ap.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="benchclone_")
    print(f"[workdir] 真实克隆目录: {workdir}")

    metas = []
    if args.preset:
        for url, lang, lic, note in REPOS:
            name = url.rstrip("/").split("/")[-1].replace(".git", "")
            metas.append(process_repo(url, lang, lic, note, workdir,
                                      args.max_commits, f"./bench_raw/{name}"))
    else:
        if not (args.repo and args.lang):
            print("❌ 单仓库模式需 --repo 和 --lang，或用 --preset"); sys.exit(1)
        metas.append(process_repo(args.repo, args.lang, args.license, "",
                                  workdir, args.max_commits, args.out))

    print(f"\n{'='*60}\n汇总\n{'='*60}")
    total_ok = 0
    for m in metas:
        flag = "✅" if m["chains_ok"] else "⚠️ 链数不足"
        print(f"  {m['repo']:12} {m['lang']:7} 演化链 {m['real_evolution_chains']:4}  "
              f"units {m['units']:4} updates {m['updates']:5}  {flag}")
        total_ok += m["chains_ok"]
    print(f"\n合格仓库 {total_ok}/{len(metas)}。")
    if total_ok < len(metas):
        print("⚠️ 有仓库演化链不够。加大 --max-commits 或换仓库，绝不合成凑数。")
    print("下一步：build_step2 给这些 units 配真实检索查询 + GT（用仓库 issue/SO 问题）。")


if __name__ == "__main__":
    main()
