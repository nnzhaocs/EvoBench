#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_step2_queries_and_gt.py
=============================
Benchmark 构造 · 第二步：给第一步的函数演化流配「真实检索查询 + GT」

铁律（与老师要求一致，代码内强制执行）：
  1. 查询和 GT 都来自真实数据，绝不合成、绝不仿真。
     - 查询 = 真实 GitHub issue 标题+正文（真实用户提的问题）。
     - GT = 修复该 issue 的真实 commit 所改动的真实函数（issue "Fixes #N" 链路）。
       这是真实的「问题→相关代码」映射，不是人为编造。
  2. 规模要够：每个仓库要求 >=MIN_QUERIES_WITH_GT 条「查询带非空 GT」，
     达不到就报警，绝不用编造的查询/GT 凑数。
  3. 检索 GT 只取「该 commit 真实改动、且在第一步 units 里的函数」——
     保证 GT 指向的单元确实在语料库里，可复现。

链路原理（全程真实，无一步合成）：
  真实 issue #N  ──"Fixes #N"──▶  真实修复 commit  ──diff──▶  真实改动的函数
     (查询)                                                      (检索 GT)

用法：
  # 依赖第一步产出的 bench_raw/<repo>/{units.jsonl, meta.json}
  python build_step2_queries_and_gt.py \
      --repo-dir ./bench_raw/click \
      --repo-url https://github.com/pallets/click.git \
      --clone-dir /tmp/benchclone_xxx/click \
      --max-issues 500

  # 预置多仓库（和第一步 preset 对应）
  python build_step2_queries_and_gt.py --preset

环境：
  pip install GitPython requests
  需要能访问 github.com（git 历史）+ api.github.com（issue 文本）。
  不需要 GPU、不需要 HuggingFace（这一步不算 embedding）。
  ★ GitHub API 有限流：匿名 60 次/时。设 GITHUB_TOKEN 环境变量可提到 5000 次/时。
"""
import argparse, os, re, json, sys, time
from collections import defaultdict

try:
    import git
except ImportError:
    print("❌ 需要 GitPython：pip install GitPython"); sys.exit(1)
try:
    import requests
except ImportError:
    print("❌ 需要 requests：pip install requests"); sys.exit(1)

MIN_QUERIES_WITH_GT = 30   # 每仓库至少这么多条「查询带真实非空 GT」，否则报警
MIN_QUERY_LEN = 15         # 太短的 issue 标题（如 "bug"）跳过，保证查询有信息量

REPOS = [
    ("https://github.com/pallets/flask.git",     "flask"),
    ("https://github.com/psf/requests.git",      "requests"),
    ("https://github.com/expressjs/express.git", "express"),
    ("https://github.com/gin-gonic/gin.git",     "gin"),
]

GH_API = "https://api.github.com"


def gh_headers():
    h = {"User-Agent": "benchmark-builder", "Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def owner_repo_from_url(url):
    m = re.search(r'github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$', url)
    return (m.group(1), m.group(2)) if m else (None, None)


def fetch_issues_by_numbers(owner, repo, numbers, cache_path=None, max_fetch=2000):
    """★v3 根治号窗口错位：不再倒序拉'最新 issue'，而是拿 commit 引用到的号，
    逐个拉详情。这样 issue 端与 commit 端永远对齐，交集不可能为空
    （旧版拉最新 issue、号偏新，与 commit 引用的旧号错开 → GT=0 的根因）。

    同时区分 issue vs PR：GitHub 的 /issues/{n} 对 PR 号也返回，用返回体里
    是否含 'pull_request' 字段判定，打 source_type 标记（论文里可再筛）。

    冻结数据源：拉到的详情写本地 cache_path（JSON），多机/复现读同一份。
    """
    # 优先读缓存（冻结源，保证多机一致、可复现）
    if cache_path and os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        print(f"  [cache] 命中 {cache_path}，读 {len(cached)} 条冻结 issue/PR 详情")
        return {int(k): v for k, v in cached.items()}

    issues = {}
    nums = sorted(numbers)[:max_fetch]
    print(f"  [fetch] 按 commit 引用的 {len(nums)} 个号逐个拉详情（对齐拉取，非倒序最新）...")
    for i, n in enumerate(nums):
        if i % 100 == 0 and i:
            print(f"    进度 {i}/{len(nums)}，已配到 {len(issues)} 条有效")
        url = f"{GH_API}/repos/{owner}/{repo}/issues/{n}"
        try:
            r = requests.get(url, headers=gh_headers(), timeout=20)
        except Exception:
            continue
        if r.status_code == 403:
            print(f"  ⚠️ GitHub API 限流（已配 {len(issues)} 条）。")
            print(f"     x-ratelimit-remaining = {r.headers.get('x-ratelimit-remaining')}")
            print(f"     → 换 token 或稍后重跑（已拉的会进缓存，不重复）。")
            break
        if r.status_code == 404:      # 号不存在（被删/转移）
            continue
        if r.status_code != 200:
            continue
        it = r.json()
        title = (it.get("title") or "").strip()
        body = (it.get("body") or "").strip()
        if len(title) < MIN_QUERY_LEN:
            continue
        # ★ 关键标记：issue 还是 PR（PR 也走 /issues/{n} 端点）
        source_type = "pr" if "pull_request" in it else "issue"
        issues[n] = {"number": n, "title": title, "body": body[:1000],
                     "source_type": source_type}
        time.sleep(0.1)
    # 写缓存冻结
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump({str(k): v for k, v in issues.items()},
                  open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  [cache] 冻结 {len(issues)} 条详情 -> {cache_path}")
    n_issue = sum(1 for v in issues.values() if v["source_type"] == "issue")
    n_pr = sum(1 for v in issues.values() if v["source_type"] == "pr")
    print(f"  拿到 {len(issues)} 条有效（真实 issue {n_issue} / PR {n_pr}）")
    return issues


def build_issue_to_commit(repo, clone_dir, max_commits=8000):
    """从真实 commit message 抽引用的 issue/PR 号 -> 该 commit 的真实改动。
    返回 {number: {"link": "strong"/"weak", "commits": [(sha, [paths]), ...]}}。
    ★v3：区分强链接(Fixes/Closes/Resolves #N，语义明确)与弱链接(尾部 (#N)，
    多为 PR 合并号)，落到查询的 link_type，供论文按需筛。"""
    issue2commit = {}
    strong_pat = re.compile(r'(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)', re.I)
    tail_pat = re.compile(r'#(\d+)')
    for commit in repo.iter_commits('HEAD', max_count=max_commits):
        msg = commit.message
        strong = set(int(n) for n in strong_pat.findall(msg))
        weak = set(int(n) for n in tail_pat.findall(msg.split("\n")[0])) - strong
        if not (strong or weak) or not commit.parents:
            continue
        try:
            diffs = commit.parents[0].diff(commit)
            paths = [d.b_path for d in diffs if d.b_path]
        except Exception:
            continue
        for n, link in [(x, "strong") for x in strong] + [(x, "weak") for x in weak]:
            e = issue2commit.setdefault(n, {"link": link, "commits": []})
            # 强链接优先：若已有弱、现遇强，升级为强
            if link == "strong":
                e["link"] = "strong"
            e["commits"].append((commit.hexsha[:12], paths))
    return issue2commit


FUNC_PATTERNS = {
    ".py": re.compile(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', re.M),
    ".js": re.compile(r'^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(', re.M),
    ".go": re.compile(r'^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(', re.M),
}
FUNC_MATCH = {ext: re.compile(pat.pattern) for ext, pat in FUNC_PATTERNS.items()}  # 行首 match 版

# ★ per-commit GT 上限：单个修复 commit 若配出超过这么多函数，判定为
#   大重构/批量改动而非"精准定位修复"，整体丢弃（不作 GT）。
#   实测行号法在大 commit 上可配出 36 个函数——那不是"这个 issue 相关的代码"，是噪声。
#   丢弃而非截断：截断会留下随机子集，同样污染；整体丢弃保证留下的 GT 都是精准定位的。
GT_MAX_PER_COMMIT = 8


def _func_ranges(text, ext):
    """解析文件全文，返回 [(func_name, start_line, end_line)]（1-based，含端点）。
    ★按语言用正确的定界方式：
      - Python：缩进定界（下一个缩进<=当前的 def 行前结束）。
      - go/js：花括号配平定界（从 def 行的 { 计数到配平），不能用缩进
        （花括号语言嵌套/闭包不靠缩进，缩进法会把区间算错、制造 false GT）。"""
    lines = text.split("\n")
    mpat = FUNC_MATCH[ext]
    if ext == ".py":
        defs = []  # (line_no, name, indent)
        for i, ln in enumerate(lines):
            m = mpat.match(ln)
            if m:
                defs.append((i + 1, m.group(1), len(ln) - len(ln.lstrip())))
        ranges = []
        for idx, (lno, name, indent) in enumerate(defs):
            end = len(lines)
            for lno2, _, indent2 in defs[idx + 1:]:
                if indent2 <= indent:
                    end = lno2 - 1
                    break
            ranges.append((name, lno, end))
        return ranges
    else:
        # 花括号配平：对每个 def 行，从该行起数 { }，到深度归零的行为区间尾。
        ranges = []
        for i, ln in enumerate(lines):
            m = mpat.match(ln)
            if not m:
                continue
            name = m.group(1)
            depth = 0
            started = False
            end = len(lines)
            for j in range(i, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if "{" in lines[j]:
                    started = True
                if started and depth <= 0:
                    end = j + 1
                    break
            ranges.append((name, i + 1, end))
        return ranges


def _changed_newlines(patch_txt):
    """从 unified diff 解析：新文件中被改动/新增的行号集合（1-based）。"""
    changed = set()
    cur_new = None
    for ln in patch_txt.split("\n"):
        if ln.startswith("@@"):
            m = re.search(r'\+(\d+)(?:,(\d+))?', ln)
            if m:
                cur_new = int(m.group(1))
        elif cur_new is not None:
            if ln.startswith("+") and not ln.startswith("+++"):
                changed.add(cur_new); cur_new += 1
            elif ln.startswith("-") and not ln.startswith("---"):
                pass  # 删除行不占新文件行号
            elif ln.startswith("\\"):
                pass  # "\ No newline at end of file"
            else:  # 上下文行
                cur_new += 1
    return changed


def funcs_changed_in_commit(repo, sha, paths, unit_index, gt_max=None):
    """★v4 行号范围法（主）+ 文本法（补）取并集，最大化召回；改函数体内部也能定位。
    再套 per-commit GT 上限过滤大 commit 噪声。返回 unit_id 列表。
    gt_max=None 时用全局 GT_MAX_PER_COMMIT。超上限返回 []（并可由调用方感知为"丢弃"）。"""
    cap = GT_MAX_PER_COMMIT if gt_max is None else gt_max
    gt = set()
    try:
        commit = repo.commit(sha)
        if not commit.parents:
            return []
        diffs = commit.parents[0].diff(commit, create_patch=True)
    except Exception:
        return []
    for d in diffs:
        path = d.b_path
        if not path:
            continue
        ext = os.path.splitext(path)[1]
        pat = FUNC_PATTERNS.get(ext)
        if not pat:
            continue
        try:
            patch_txt = d.diff.decode("utf-8", errors="ignore") if isinstance(d.diff, bytes) else str(d.diff)
        except Exception:
            patch_txt = ""

        # ---- 路径1（主）：行号范围法。改动行号落在哪个函数区间 → 该函数 ----
        try:
            newtext = (commit.tree / path).data_stream.read().decode("utf-8", errors="ignore")
        except Exception:
            newtext = ""
        if newtext:
            changed_lines = _changed_newlines(patch_txt)
            for name, s, e in _func_ranges(newtext, ext):
                if any(s <= cl <= e for cl in changed_lines):
                    for uid in unit_index.get(f"{path}::{name}", []):
                        gt.add(uid)

        # ---- 路径2（补）：文本法。def 行直接出现在改动行里 ----
        changed_code_lines = []
        for ln in patch_txt.split("\n"):
            if ln.startswith("+") and not ln.startswith("+++"):
                changed_code_lines.append(ln[1:])
            elif ln.startswith("-") and not ln.startswith("---"):
                changed_code_lines.append(ln[1:])
        for fname in set(pat.findall("\n".join(changed_code_lines))):
            for uid in unit_index.get(f"{path}::{fname}", []):
                gt.add(uid)

        # ---- 路径3（补）：hunk 头上下文里的函数名 ----
        for hdr in re.findall(r'@@.*@@\s*(.*)', patch_txt):
            for fname in pat.findall(hdr):
                for uid in unit_index.get(f"{path}::{fname}", []):
                    gt.add(uid)

    # ★ per-commit GT 上限：大 commit 配出过多函数 = 批量重构，不是精准修复，整体丢弃
    if len(gt) > cap:
        return None   # None = 被上限丢弃（区别于 [] = 没配到任何函数），调用方可统计
    return sorted(gt)


STEP2_VERSION = "v4 (行号范围法+花括号定界+GT上限; commit-first号对齐+冻结+issue/PR标记)"


def process_repo(repo_dir, repo_url, clone_dir, max_issues, out_dir, gt_max=None):
    repo_name = os.path.basename(repo_dir.rstrip("/"))
    cap = GT_MAX_PER_COMMIT if gt_max is None else gt_max
    print(f"\n{'='*60}\n仓库: {repo_name}  [step2 {STEP2_VERSION}]  GT上限={cap}\n{'='*60}")

    units_path = os.path.join(repo_dir, "units.jsonl")
    if not os.path.exists(units_path):
        print(f"❌ 找不到第一步产出 {units_path}，先跑 build_step1。"); return None
    units = [json.loads(l) for l in open(units_path, encoding="utf-8")]
    # 建索引：path::func -> [unit_id...]（第一步 unit_id 形如 repo::path::func）
    unit_index = defaultdict(list)
    for u in units:
        uid = u["unit_id"]
        parts = uid.split("::", 1)
        if len(parts) == 2:
            unit_index[parts[1]].append(uid)  # key = path::func
    print(f"[units] 第一步单元 {len(units)}")

    owner, repo_short = owner_repo_from_url(repo_url)
    if not owner:
        print(f"❌ 无法解析 {repo_url}"); return None

    if not os.path.exists(clone_dir):
        print(f"[clone] 需要第一步的克隆目录；重新克隆 {repo_url} ...")
        git.Repo.clone_from(repo_url, clone_dir)
    repo = git.Repo(clone_dir)
    head_sha = repo.head.commit.hexsha[:12]   # ★冻结：记录本次 HEAD，写进 meta
    print(f"[freeze] 本次 clone HEAD = {head_sha}（复现需 checkout 此 SHA）")

    # ★v3 关键改动：先扫 commit 得到"被引用的号集合"，再按这些号拉 issue 详情。
    #   旧版倒过来（先拉最新 issue 再匹配 commit）导致号窗口错位 → GT=0。
    print(f"[link] 先扫 commit，建 号→修复commit→改动函数 的真实链路...")
    issue2commit = build_issue_to_commit(repo, clone_dir)
    n_strong = sum(1 for e in issue2commit.values() if e["link"] == "strong")
    print(f"  commit 引用了 {len(issue2commit)} 个号（强链接 {n_strong} / 弱链接 {len(issue2commit)-n_strong}）")

    print(f"[issues] 按引用号拉 issue/PR 详情（对齐拉取，冻结缓存）...")
    cache_path = os.path.join(out_dir, "issue_cache.json")
    os.makedirs(out_dir, exist_ok=True)
    issues = fetch_issues_by_numbers(owner, repo_short, set(issue2commit.keys()),
                                     cache_path=cache_path, max_fetch=max_issues)

    # 组装 queries + GT（全部真实，带 link_type / source_type 标记）
    queries, gt = [], {}
    dropped_commits = 0   # 被 GT 上限丢弃的大 commit 数（写进 meta，供调阈值）
    for num, iss in issues.items():
        if num not in issue2commit:
            continue
        gt_units = set()
        for sha, paths in issue2commit[num]["commits"]:
            fc = funcs_changed_in_commit(repo, sha, paths, unit_index, gt_max=cap)
            if fc is None:       # 被上限丢弃
                dropped_commits += 1
                continue
            gt_units.update(fc)
        if not gt_units:
            continue  # 修复没落在语料函数上，跳过（不硬塞）
        q = iss["title"] + (("\n" + iss["body"]) if iss["body"] else "")
        qid = f"{repo_name}#{iss['source_type']}-{num}"
        queries.append({"query_id": qid, "query": q, "issue_number": num,
                        "link_type": issue2commit[num]["link"],      # strong / weak
                        "source_type": iss["source_type"]})           # issue / pr
        gt[qid] = sorted(gt_units)

    # 分层统计（论文按需筛：最严 = strong+issue）
    def cnt(lt=None, st=None):
        return sum(1 for q in queries
                   if (lt is None or q["link_type"] == lt)
                   and (st is None or q["source_type"] == st))
    print(f"[out] 带非空GT查询 {len(queries)}  |  "
          f"strong={cnt('strong')} weak={cnt('weak')}  |  "
          f"issue={cnt(st='issue')} pr={cnt(st='pr')}  |  "
          f"最严(strong+issue)={cnt('strong','issue')}")
    print(f"[gt-cap] GT上限={cap}，被丢弃的大commit数={dropped_commits}"
          f"（丢弃多→该仓库批量重构多、可定位修复少；可调 --gt-max 观察）")

    # ---------- 铁律校验 ----------
    if len(queries) < MIN_QUERIES_WITH_GT:
        print(f"  ⚠️ 带真实 GT 的查询只有 {len(queries)} < 要求 {MIN_QUERIES_WITH_GT}。")
        print(f"     → 加大 --max-issues，或选 issue 更多的活跃仓库。")
        print(f"     → 绝不允许编造查询或硬塞 GT 凑数。")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    json.dump(gt, open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta2 = {"repo": repo_name, "head_sha_frozen": head_sha,
             "referenced_numbers": len(issue2commit),
             "issues_fetched": len(issues),
             "queries_with_gt": len(queries),
             "by_link_type": {"strong": cnt("strong"), "weak": cnt("weak")},
             "by_source_type": {"issue": cnt(st="issue"), "pr": cnt(st="pr")},
             "strictest_strong_issue": cnt("strong", "issue"),
             "gt_max_per_commit": cap,
             "dropped_large_commits": dropped_commits,
             "min_required": MIN_QUERIES_WITH_GT,
             "ok": len(queries) >= MIN_QUERIES_WITH_GT,
             "synthetic_data": False, "fake_embed": False,
             "gt_source": "commit-first linkage: ref#N in commit -> fetch issue/PR by number -> changed funcs",
             "reproduce_note": f"checkout {head_sha} + read issue_cache.json for identical result"}
    json.dump(meta2, open(os.path.join(out_dir, "meta_step2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  写出 queries.jsonl + ground_truth.json -> {out_dir}")
    return meta2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", help="第一步某仓库产出目录，如 ./bench_raw/click")
    ap.add_argument("--repo-url", help="该仓库 git url")
    ap.add_argument("--clone-dir", help="第一步的克隆目录（复用，避免重克隆）")
    ap.add_argument("--max-issues", type=int, default=500)
    ap.add_argument("--gt-max", type=int, default=GT_MAX_PER_COMMIT,
                    help=f"单commit配出超过此数的函数则整体丢弃（默认{GT_MAX_PER_COMMIT}）。"
                         "调小=更严格只留精准修复；调大=召回高但引入大commit噪声。")
    ap.add_argument("--preset", action="store_true")
    ap.add_argument("--bench-root", default="./bench_raw")
    args = ap.parse_args()

    if not os.environ.get("GITHUB_TOKEN"):
        print("⚠️ 未设 GITHUB_TOKEN，GitHub API 限流 60 次/时，可能拉不满。")
        print("   建议：export GITHUB_TOKEN=你的token（5000 次/时）。\n")

    metas = []
    if args.preset:
        for url, name in REPOS:
            rd = os.path.join(args.bench_root, name)
            cd = f"/tmp/benchclone_{name}"
            metas.append(process_repo(rd, url, cd, args.max_issues,
                                      os.path.join(rd, "step2"), gt_max=args.gt_max))
    else:
        if not (args.repo_dir and args.repo_url):
            print("❌ 需 --repo-dir 和 --repo-url，或用 --preset"); sys.exit(1)
        cd = args.clone_dir or f"/tmp/benchclone_{os.path.basename(args.repo_dir)}"
        metas.append(process_repo(args.repo_dir, args.repo_url, cd,
                                  args.max_issues, os.path.join(args.repo_dir, "step2"),
                                  gt_max=args.gt_max))

    print(f"\n{'='*60}\n汇总\n{'='*60}")
    ok = 0
    for m in metas:
        if not m: continue
        flag = "✅" if m["ok"] else "⚠️ 查询不足"
        print(f"  {m['repo']:12} 真实查询(带GT) {m['queries_with_gt']:4}  {flag}")
        ok += m["ok"]
    print(f"\n合格仓库 {ok}/{len([m for m in metas if m])}。")
    print("下一步：build_step3 用真 MiniLM 跑体检/统计，把 units+updates+queries+GT 打包发布。")


if __name__ == "__main__":
    main()
