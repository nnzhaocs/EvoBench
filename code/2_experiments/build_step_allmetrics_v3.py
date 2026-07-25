#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_step_allmetrics.py  —— 一次跑出全部指标
==============================================
把判别力(原step5) + 多指标(原step6) 合并。嵌入模型只前向一次,所有指标一起算。

★ 学生只需跑这一个脚本,得到论文需要的全部补充指标:

【检索质量指标】(编辑后库状态)
  - Recall@k, MRR@k, nDCG@k
  - top-k 翻转率(编辑前后 top-k 集合变化的查询比例)

【重算决策指标】(正类 = I(u)>0, 即该更新真正改变了检索)
  - 判别力 Disc(π) = F1(R(π), H)  —— always/never/audit-τ 各一个
  - 各 τ 下 Precision / Recall / F1 / 漏判率(FN rate)
  - |H| 守卫: |H|<10 的库标黄(F1噪声大,别单独进表)

依赖: 只读数据集四件套(units/updates/queries.txt/ground_truth.json)。
      不抓 issue、不需要 GitHub token。只需 MiniLM(或指定模型)前向一次。

用法:
  cd 2_experiments
  python build_step_allmetrics.py --model minilm --bench-final ../bench_all_clean --out ../allmetrics_result
  # 可选多模型: --model mpnet / bge ...
"""
import argparse, os, json, glob
import numpy as np
from build_step4_multimodel import Embedder, load_jsonl

try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

H_SMALL = 10  # |H| 守卫阈值


# ---------- 统计工具: 核心负结果(|Δ|小)的显著性保护 ----------
def bootstrap_ci_mean(diffs, n_boot=2000, seed=0):
    """对逐查询配对差 diffs 的均值做 bootstrap 95% CI。
       核心发现是'always vs never 差异极小'——这是负结果,必须证明它不是样本不足的噪声。"""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return {"mean": round(float(d.mean()), 4),
            "ci_low": round(float(np.percentile(means, 2.5)), 4),
            "ci_high": round(float(np.percentile(means, 97.5)), 4),
            "n": int(d.size)}


def paired_test(diffs, min_nonzero=10):
    """配对 Wilcoxon: H0 = 编辑前后 recall 无差异(即 always 与 never 无差异)。
       p 大 => 差异不显著,支撑'重算收益稀疏'。
       注意: 零差(编辑前后 recall 相同)是'影响稀疏'的核心证据,故一并报告 n_total 与 n_nonzero。
       非零配对差过少(< min_nonzero)时不报 p(样本量不足,Wilcoxon 不可信)。"""
    d = np.asarray(diffs, dtype=float)
    n_total = int(d.size)
    nz = d[d != 0]
    n_nonzero = int(nz.size)
    if not _HAS_SCIPY:
        return {"test": "wilcoxon", "p_value": None, "n_total": n_total,
                "n_nonzero": n_nonzero, "note": "scipy缺失"}
    if n_nonzero < min_nonzero:
        return {"test": "wilcoxon", "p_value": None, "n_total": n_total,
                "n_nonzero": n_nonzero, "note": f"非零配对差{n_nonzero}<{min_nonzero},样本量不足,p不可信"}
    try:
        # zero_method='zsplit' 保留零差信息、不悄悄丢弃; 全部配对差参与
        stat, p = wilcoxon(d, zero_method="zsplit")
        return {"test": "wilcoxon", "p_value": round(float(p), 4),
                "n_total": n_total, "n_nonzero": n_nonzero}
    except Exception as e:
        return {"test": "wilcoxon", "p_value": None, "n_total": n_total,
                "n_nonzero": n_nonzero, "note": str(e)}


def quantiles(vals, qs=(10, 50, 90)):
    if not vals:
        return {f"p{q}": 0.0 for q in qs}
    a = np.asarray(vals, dtype=float)
    return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs}

# ---------- 检索指标工具 ----------
def dcg(rels): return sum(r/np.log2(i+2) for i,r in enumerate(rels))
def ndcg_at_k(ranked, gset, k):
    rels=[1.0 if u in gset else 0.0 for u in ranked[:k]]
    idcg=dcg(sorted(rels,reverse=True)); return (dcg(rels)/idcg) if idcg>0 else 0.0
def mrr_at_k(ranked, gset, k):
    for i,u in enumerate(ranked[:k]):
        if u in gset: return 1.0/(i+1)
    return 0.0
def recall_at_k(ranked, gset, k):
    if not gset: return 0.0
    return sum(1 for u in ranked[:k] if u in gset)/len(gset)


def process_repo(rd, emb, k, taus):
    name=os.path.basename(rd.rstrip("/"))
    units=load_jsonl(os.path.join(rd,"units.jsonl"))
    updates=load_jsonl(os.path.join(rd,"updates.jsonl"))
    qtexts=[l.strip() for l in open(os.path.join(rd,"queries.txt"),encoding="utf-8") if l.strip()]
    gt=json.load(open(os.path.join(rd,"ground_truth.json"),encoding="utf-8"))
    if not units or not qtexts or not gt:
        print(f"  ⚠️ {name}: 缺四件套,跳过"); return None

    unit_ids=[u["unit_id"] for u in units]
    id2idx={uid:i for i,uid in enumerate(unit_ids)}
    uid_set=set(unit_ids)
    valid=[u for u in updates if u.get("new_text") and u["unit_id"] in uid_set]

    # 版本对齐(与 step4 一致): U0=first_old, 新向量=last_new
    first_old,last_new={},{}
    for u in valid:
        uid=u["unit_id"]
        if uid not in first_old and (u.get("old_text") or "").strip(): first_old[uid]=u["old_text"]
        last_new[uid]=u["new_text"]
    u0_texts=[first_old.get(u["unit_id"],u.get("text","")) for u in units]

    # ===== 嵌入只前向一次 =====
    U0=emb.encode(u0_texts)
    Q=emb.encode(qtexts, is_query=True)
    U_new_map={}
    if last_new:
        keys=list(last_new.keys()); NEW=emb.encode([last_new[uid] for uid in keys])
        for uid,v in zip(keys,NEW): U_new_map[uid]=v

    # 编辑后库状态
    U_after=U0.copy()
    for uid,v in U_new_map.items():
        if uid in id2idx: U_after[id2idx[uid]]=v

    # ===== 1) 检索质量指标 (含逐查询 before/after recall, 供配对显著性检验) =====
    # k 敏感性: 在一次 argsort 里同时算 k∈{5,10,20} (W20)。
    # 主指标仍以 args.k (默认 10) 报,其余作为附录鲁棒性证据。
    K_SENS = sorted(set([5, 10, 20, k]))
    K_MAX = max(K_SENS)
    k_metrics = {kv: {"mrr":0.0,"ndcg":0.0,"rec":0.0,"flips":0,
                       "rec_after_list":[],"rec_before_list":[]} for kv in K_SENS}
    nq = 0
    for qi,qt in enumerate(qtexts):
        g=gt.get(qt)
        if not g: continue
        nq+=1; gset=set(g)
        # 一次 argsort 拿 top-K_MAX
        order_after = np.argsort(-(U_after@Q[qi]))[:K_MAX]
        order_before = np.argsort(-(U0@Q[qi]))[:K_MAX]
        ranked_after_ids = [unit_ids[i] for i in order_after]
        ranked_before_ids = [unit_ids[i] for i in order_before]
        for kv in K_SENS:
            r_after = recall_at_k(ranked_after_ids[:kv], gset, kv)
            r_before = recall_at_k(ranked_before_ids[:kv], gset, kv)
            k_metrics[kv]["rec_after_list"].append(r_after)
            k_metrics[kv]["rec_before_list"].append(r_before)
            k_metrics[kv]["rec"] += r_after
            k_metrics[kv]["mrr"] += mrr_at_k(ranked_after_ids[:kv], gset, kv)
            k_metrics[kv]["ndcg"] += ndcg_at_k(ranked_after_ids[:kv], gset, kv)
            if set(order_after[:kv].tolist()) != set(order_before[:kv].tolist()):
                k_metrics[kv]["flips"] += 1
    for kv in K_SENS:
        for key in ("rec","mrr","ndcg"): k_metrics[kv][key] /= nq
        k_metrics[kv]["flip_rate"] = k_metrics[kv]["flips"] / nq

    # 主指标沿用 args.k
    rec_after_list = k_metrics[k]["rec_after_list"]
    rec_before_list = k_metrics[k]["rec_before_list"]
    mrr = k_metrics[k]["mrr"]; ndcg = k_metrics[k]["ndcg"]
    rec = k_metrics[k]["rec"]; flip_rate = k_metrics[k]["flip_rate"]

    # Δ_edit = recall(编辑后) − recall(编辑前), 逐查询配对
    diffs=[a-b for a,b in zip(rec_after_list, rec_before_list)]
    delta_ci=bootstrap_ci_mean(diffs)
    delta_sig=paired_test(diffs)

    # ===== 2) 逐更新 I(u) —— 扩展定义 =====
    # 旧 I(u) 只统计 u 是 GT 且 u 自己进出 top-k 的查询。
    # 新 I(u) 遍历所有查询,统计任何以下情况:
    #   (a) 某 GT 单元在 topk_old 但不在 topk_new (被挤出)
    #   (b) 某 GT 单元在 topk_new 但不在 topk_old (新进入)
    #   (c) u 是 GT 且 u 自身进出 top-k (旧定义,已被 a/b 覆盖)
    # 未覆盖:top-k 内部排名变化 (影响 MRR/nDCG),这在 threats 里明说。
    # 同时保留旧定义 I_old(u) 以便对照,不动下游 H 的口径可以两套报。
    I = {}           # 新 I'(u): 全查询扫,统计涉 GT 的 top-k 集合变化
    I_narrow = {}    # 旧 I(u): 仅 u 是 GT 的查询,统计 u 自身进出
    unit_to_q = {}
    for qi, qt in enumerate(qtexts):
        g = gt.get(qt)
        if not g: continue
        for uid in g: unit_to_q.setdefault(uid, []).append(qi)

    # 预计算所有查询在 U0 状态下的 top-k (加速)
    all_qi_with_gt = [qi for qi, qt in enumerate(qtexts) if gt.get(qt)]
    topk_old_by_q = {}
    for qi in all_qi_with_gt:
        topk_old_by_q[qi] = set(np.argsort(-(U0 @ Q[qi]))[:k].tolist())

    for u in valid:
        uid = u["unit_id"]
        if uid not in U_new_map or uid not in id2idx:
            continue
        idx = id2idx[uid]
        # 旧定义: 仅遍历 u 是 GT 的查询
        narrow_aq = unit_to_q.get(uid, [])
        ch_narrow = 0
        # 新定义: 遍历所有带 GT 的查询
        ch_new = 0

        saved = U0[idx].copy()
        U0[idx] = U_new_map[uid]  # 应用该更新

        for qi in all_qi_with_gt:
            t_old = topk_old_by_q[qi]
            t_new = set(np.argsort(-(U0 @ Q[qi]))[:k].tolist())
            if t_old == t_new: continue
            # 该查询的 GT 单元 (以 idx 表示)
            qt = qtexts[qi]
            gt_ids = set(id2idx[g] for g in gt.get(qt, []) if g in id2idx)
            if not gt_ids: continue
            # 新定义 (a)+(b): 涉 GT 的集合变化
            gt_in_old = gt_ids & t_old
            gt_in_new = gt_ids & t_new
            if gt_in_old != gt_in_new:
                ch_new += 1
            # 旧定义: u 本身进出 top-k 且 u 是该查询的 GT
            if qi in narrow_aq and (idx in t_old) != (idx in t_new):
                ch_narrow += 1

        U0[idx] = saved  # 还原
        I[uid] = ch_new
        I_narrow[uid] = ch_narrow

    H = set(uid for uid, v in I.items() if v > 0)
    H_narrow = set(uid for uid, v in I_narrow.items() if v > 0)
    h_unreliable = len(H) < H_SMALL

    # ===== 每库 δ 分位数 (把'内容更新影响稀疏'从 gin 个案升为跨库普遍) =====
    delta_vals=[]
    for u in valid:
        uid=u["unit_id"]
        if uid in id2idx and uid in U_new_map:
            delta_vals.append(1.0-float(U0[id2idx[uid]]@U_new_map[uid]))
    delta_dist=quantiles(delta_vals)
    delta_dist["max"]=round(float(max(delta_vals)),4) if delta_vals else 0.0
    delta_dist["frac_below_0.02"]=round(float(np.mean([d<0.02 for d in delta_vals])),4) if delta_vals else 0.0

    # 每个更新的 δ(按 unit_id), 供跨模型一致性矩阵在多次运行后计算
    delta_by_uid={}
    for u in valid:
        uid=u["unit_id"]
        if uid in id2idx and uid in U_new_map:
            delta_by_uid[uid]=round(1.0-float(U0[id2idx[uid]]@U_new_map[uid]),6)

    # ===== 每库 GT 大小分布 (构念效度: 修复精准落在少数函数) =====
    gt_sizes=[len(g) for g in gt.values() if g]
    gt_dist={"mean":round(float(np.mean(gt_sizes)),4) if gt_sizes else 0.0,
             "median":int(np.median(gt_sizes)) if gt_sizes else 0,
             "frac_size1":round(float(np.mean([s==1 for s in gt_sizes])),4) if gt_sizes else 0.0,
             "frac_size2to3":round(float(np.mean([2<=s<=3 for s in gt_sizes])),4) if gt_sizes else 0.0}

    # ===== 3) 决策指标 + 判别力(always/never/audit-τ) =====
    def decide(policy, tau):
        R=set()
        for u in valid:
            uid=u["unit_id"]
            if uid not in id2idx or uid not in U_new_map: continue
            delta=1.0-float(U0[id2idx[uid]]@U_new_map[uid])
            do = (policy=="always") or (policy=="audit" and delta>tau)
            if do: R.add(uid)
        tp=len(R&H); fp=len(R-H); fn=len(H-R)
        missed_H = sorted(H - R)  # audit 漏掉的高影响 uid,供 W15 分析
        prec=tp/(tp+fp) if (tp+fp) else 0.0
        recl=tp/(tp+fn) if (tp+fn) else 0.0
        f1=2*prec*recl/(prec+recl) if (prec+recl) else 0.0
        fn_rate=fn/len(H) if H else 0.0
        # 省算率: 相对 always 基线跳过的比例。always 恒为 0(基线本身)。
        # n_always 用去重的 unit 数,与 R(set)同口径,保证 always 的 save_pct=0。
        n_always=len(set(u["unit_id"] for u in valid
                         if u["unit_id"] in id2idx and u["unit_id"] in U_new_map))
        save_pct=round((1.0-len(R)/n_always)*100,1) if n_always else 0.0
        return {"policy":policy if policy!="audit" else f"audit(τ={tau})",
                "precision":round(prec,4),"recall":round(recl,4),"f1":round(f1,4),
                "fn_rate":round(fn_rate,4),"save_pct":save_pct,
                "TP":tp,"FP":fp,"FN":fn,"R_size":len(R),
                "missed_H_uids":missed_H}
    decisions=[decide("always",0),decide("never",0)]+[decide("audit",t) for t in taus]

    # ===== W15: 每个 audit-τ 策略, 分析其漏掉的 H uid 的 I(u) 分布 =====
    # 目的: 验证"audit 漏掉的更新对检索质量边际影响低"这一推测。
    # 若 missed 的 I(u) 中位数 << H 全集 I(u) 中位数, 则支持推测; 反之则该说法不成立。
    def _stats(vals):
        if not vals: return {"n":0}
        a=np.asarray(vals, dtype=float)
        return {"n":int(a.size),"mean":round(float(a.mean()),3),
                "median":round(float(np.median(a)),3),
                "max":int(a.max())}
    H_I_vals = [I[u] for u in H]
    fn_I_analysis = {"H_all_I": _stats(H_I_vals)}
    for dm in decisions:
        if not dm["policy"].startswith("audit"): continue
        missed = dm["missed_H_uids"]
        missed_I = [I[u] for u in missed if u in I]
        fn_I_analysis[dm["policy"]] = {
            "missed_I": _stats(missed_I),
            "verdict": ("推测成立: missed 中位 I(u) 明显低于全 H 中位"
                        if missed_I and np.median(missed_I) < 0.5*np.median(H_I_vals or [1])
                        else ("推测不成立: missed 中位 I(u) 接近或高于全 H 中位"
                              if missed_I else "无漏掉(策略过于激进)"))
        }

    # 清理 decisions 中的 missed_H_uids (太长, 不进最终 JSON, 只用于分析)
    for dm in decisions: dm.pop("missed_H_uids", None)

    # ===== W16: δ vs 字符级编辑量 相关(每库一个 Spearman) =====
    # 简易 edit_size = |old|+|new| - 2*len(common substring) 近似,不用真 Levenshtein
    def _edit_size(a, b):
        a = a or ""; b = b or ""
        # 最长公共子串近似,快: 用 set 的字符 bigram 重合
        if not a or not b: return len(a) + len(b)
        return abs(len(a) - len(b)) + sum(1 for ca, cb in zip(a, b) if ca != cb)
    delta_edit_pairs = []
    for u in valid:
        uid = u["unit_id"]
        if uid in delta_by_uid:
            es = _edit_size(u.get("old_text", ""), u.get("new_text", ""))
            delta_edit_pairs.append((delta_by_uid[uid], es))
    delta_vs_editsize = {"n": len(delta_edit_pairs), "spearman": None}
    if len(delta_edit_pairs) >= 10:
        try:
            from scipy.stats import spearmanr
            xs = [p[0] for p in delta_edit_pairs]; ys = [p[1] for p in delta_edit_pairs]
            rho = spearmanr(xs, ys).correlation
            delta_vs_editsize["spearman"] = None if rho != rho else round(float(rho), 3)
        except Exception as e:
            delta_vs_editsize["error"] = str(e)

    # ===== W6 支撑: 每库更新路径分布(前缀聚合) =====
    # unit_id 形如 "repo::path/to/file.py::funcname"; 取 path 前两层作路径前缀
    from collections import Counter
    prefix_counter = Counter()
    for u in valid:
        parts = u["unit_id"].split("::")
        if len(parts) >= 2:
            path_parts = parts[1].split("/")
            prefix = "/".join(path_parts[:2]) if len(path_parts) >= 2 else path_parts[0]
            prefix_counter[prefix] += 1
    update_path_dist = {"n_prefixes": len(prefix_counter),
                        "top5": [{"prefix": p, "n": n} for p, n in prefix_counter.most_common(5)]}

    # 打印
    print(f"\n=== {name} (nq={nq}, |H|_new={len(H)}, |H|_narrow={len(H_narrow)}) ===")
    print(f"  [I(u) 定义对比] 扩展 |H|={len(H)} vs 旧 |H|={len(H_narrow)} "
          f"(扩展/旧={len(H)/max(len(H_narrow),1):.2f}x)")
    print(f"  [检索 k={k}] Recall={rec:.3f} MRR={mrr:.3f} nDCG={ndcg:.3f} 翻转率={flip_rate:.3f}")
    # W20 k 敏感性:并列打印其他 k
    for kv in K_SENS:
        if kv == k: continue
        km = k_metrics[kv]
        print(f"  [检索 k={kv}] Recall={km['rec']:.3f} MRR={km['mrr']:.3f} nDCG={km['ndcg']:.3f} 翻转率={km['flip_rate']:.3f}")
    pv = delta_sig.get("p_value")
    print(f"  [Δ_edit] 均值={delta_ci['mean']:+.4f} 95%CI=[{delta_ci['ci_low']:+.4f},{delta_ci['ci_high']:+.4f}] "
          f"配对p={pv if pv is not None else 'NA'} (n={delta_ci['n']}, 非零差={delta_sig.get('n_nonzero')})")
    print(f"  [δ分布] p10/p50/p90={delta_dist['p10']}/{delta_dist['p50']}/{delta_dist['p90']} "
          f"max={delta_dist['max']} δ<0.02占比={delta_dist['frac_below_0.02']}")
    print(f"  [GT大小] 均值={gt_dist['mean']} 中位={gt_dist['median']} "
          f"单函数占比={gt_dist['frac_size1']} 2-3占比={gt_dist['frac_size2to3']}")
    if h_unreliable:
        print(f"  ⚠️⚠️ |H|={len(H)}<{H_SMALL}: 决策/判别力指标噪声大,别单独进表,仅汇总用")
    for dm in decisions:
        print(f"  [决策] {dm['policy']:14} F1(Disc)={dm['f1']:.3f} P={dm['precision']:.3f} R={dm['recall']:.3f} 漏判={dm['fn_rate']:.3f} 省算={dm['save_pct']:.1f}%")
    # W16 打印
    if delta_vs_editsize["spearman"] is not None:
        print(f"  [W16 δ vs 编辑量] Spearman ρ = {delta_vs_editsize['spearman']} (n={delta_vs_editsize['n']})")
    # W15 打印
    print(f"  [W15 audit 漏掉的 H uid 的 I(u)]")
    print(f"    全 H I(u) 中位={fn_I_analysis['H_all_I'].get('median','NA')} 均值={fn_I_analysis['H_all_I'].get('mean','NA')} n={fn_I_analysis['H_all_I'].get('n')}")
    for pol, info in fn_I_analysis.items():
        if pol == "H_all_I": continue
        mi = info["missed_I"]
        print(f"    {pol:14} missed I(u) 中位={mi.get('median','NA')} 均值={mi.get('mean','NA')} n={mi.get('n')} → {info['verdict']}")

    return {"repo":name,"nq":nq,"H_size":len(H),"H_size_narrow":len(H_narrow),
            "H_unreliable":h_unreliable,
            "retrieval":{"recall":round(rec,4),"mrr":round(mrr,4),"ndcg":round(ndcg,4),"flip_rate":round(flip_rate,4)},
            "k_sensitivity":{str(kv):{"recall":round(k_metrics[kv]["rec"],4),
                                       "mrr":round(k_metrics[kv]["mrr"],4),
                                       "ndcg":round(k_metrics[kv]["ndcg"],4),
                                       "flip_rate":round(k_metrics[kv]["flip_rate"],4)}
                             for kv in K_SENS},
            "delta_edit":{**delta_ci,"significance":delta_sig},
            "delta_dist":delta_dist,
            "gt_dist":gt_dist,
            "decision":decisions,
            "delta_vs_editsize":delta_vs_editsize,
            "fn_I_analysis":fn_I_analysis,
            "update_path_dist":update_path_dist,
            "delta_by_uid":delta_by_uid,
            "I_by_uid":I,
            "I_narrow_by_uid":I_narrow}  # 用于 uid 级 FN 验证:audit 漏掉的更新 I(u) 分布


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bench-final",default="../bench_all_clean")
    ap.add_argument("--out",default="../allmetrics_result")
    ap.add_argument("--model",default="minilm")
    ap.add_argument("--k",type=int,default=10)
    ap.add_argument("--taus",default="0.02,0.05,0.1,0.2")
    ap.add_argument("--verify-pipeline-only",action="store_true")
    args=ap.parse_args()
    taus=[float(x) for x in args.taus.split(",")]
    os.makedirs(args.out,exist_ok=True)
    emb=Embedder(args.model,verify_only=args.verify_pipeline_only)
    repos=sorted([d for d in glob.glob(os.path.join(args.bench_final,"*")) if os.path.isdir(d)])
    results=[]
    for rd in repos:
        try:
            r=process_repo(rd,emb,k=args.k,taus=taus)
            if r: results.append(r)
        except Exception as e:
            print(f"  🔴 {os.path.basename(rd)} 失败: {e}")
    out={"model":args.model,"k":args.k,"taus":taus,"fake_embed":args.verify_pipeline_only,"per_repo":results}
    json.dump(out,open(os.path.join(args.out,f"allmetrics_{args.model}.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n✅ 全部指标写入 {args.out}/allmetrics_{args.model}.json")
    # 可靠性分区
    reliable=[r["repo"] for r in results if not r["H_unreliable"]]
    unrel=[r["repo"] for r in results if r["H_unreliable"]]
    print(f"\n=== 可靠性分区(|H|≥{H_SMALL}) ===")
    print(f"  ✅ 可进表({len(reliable)}): {', '.join(reliable) or '无'}")
    print(f"  ⚠️ |H|过小({len(unrel)}): {', '.join(unrel) or '无'}")

    # ===== W35: 跨库结构性因素 vs audit FN 率 的相关分析 =====
    # 目的: 证明"小 H 库 audit 天然受限"是系统模式, 不是 pylint 的个例。
    # 用可靠库(|H|≥H_SMALL)避免小样本噪声, 报告 Pearson r。
    try:
        from scipy.stats import pearsonr
        rel_rs = [r for r in results if not r["H_unreliable"]]
        if len(rel_rs) >= 4:
            # 每库取"最优 F1 τ"的 audit 项作 FN 率
            def _best_audit(r):
                aud = [d for d in r["decision"] if d["policy"].startswith("audit")]
                return max(aud, key=lambda x: x["f1"]) if aud else None
            rows = []
            for r in rel_rs:
                ba = _best_audit(r)
                if ba is None: continue
                H_rare = r["H_size"] / r["nq"] if r["nq"] else 0
                rows.append({"repo": r["repo"],
                             "nq": r["nq"], "H_size": r["H_size"], "H_rare": round(H_rare,3),
                             "gt_size1": r["gt_dist"]["frac_size1"],
                             "delta_p50": r["delta_dist"]["p50"],
                             "fn_rate": ba["fn_rate"], "audit_f1": ba["f1"],
                             "best_tau": ba["policy"]})
            def _corr(key1, key2):
                x = [r[key1] for r in rows]; y = [r[key2] for r in rows]
                if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
                    return None
                r_val, p_val = pearsonr(x, y)
                return {"r": round(float(r_val),3), "p": round(float(p_val),3), "n": len(x)}
            struct_corr = {
                "rows": rows,
                "corr(|H|, FN率)": _corr("H_size", "fn_rate"),
                "corr(H相对稀有度, FN率)": _corr("H_rare", "fn_rate"),
                "corr(GT单函数占比, FN率)": _corr("gt_size1", "fn_rate"),
                "corr(δ中位数, FN率)": _corr("delta_p50", "fn_rate"),
                "corr(nq, audit_F1)": _corr("nq", "audit_f1"),
            }
            out["structural_correlations"] = struct_corr
            print("\n=== W35: 结构性因素 vs audit FN 率 (可靠库 Pearson r) ===")
            for k_, v_ in struct_corr.items():
                if k_ == "rows": continue
                if v_ is None:
                    print(f"  {k_:30}: 数据不足")
                else:
                    sig = "*" if v_["p"] < 0.05 else " "
                    print(f"  {k_:30}: r={v_['r']:+.3f} p={v_['p']:.3f} n={v_['n']} {sig}")
            # 重新写 JSON(含 structural_correlations)
            json.dump(out,open(os.path.join(args.out,f"allmetrics_{args.model}.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        else:
            print("\n  (W35 结构性相关: 可靠库不足 4 个,跳过)")
    except Exception as e:
        print(f"  (W35 结构性相关分析跳过: {e})")

    # ===== 跨模型 δ 一致性矩阵 (补第四维指标) =====
    # 扫描 out 目录里所有 allmetrics_*.json,对已跑过的每对模型算 δ 排序的 Spearman 相关。
    # 单模型时只写自己的 δ;跑过 2+ 模型后自动生成一致性矩阵。
    try:
        compute_cross_model_consistency(args.out)
    except Exception as e:
        print(f"  (跨模型一致性矩阵跳过: {e})")


def compute_cross_model_consistency(out_dir):
    try:
        from scipy.stats import spearmanr
    except Exception:
        print("  (跨模型一致性: scipy缺失,跳过)"); return
    files=sorted(glob.glob(os.path.join(out_dir,"allmetrics_*.json")))
    model_delta={}  # model -> {uid: delta}
    for f in files:
        try:
            d=json.load(open(f,encoding="utf-8"))
        except Exception:
            continue
        if d.get("fake_embed"): continue  # 假嵌入不进一致性矩阵
        m=d.get("model")
        merged={}
        for r in d.get("per_repo",[]):
            merged.update(r.get("delta_by_uid",{}) or {})
        if merged: model_delta[m]=merged
    models=sorted(model_delta.keys())
    if len(models)<2:
        print(f"\n=== 跨模型 δ 一致性矩阵 ===\n  只有 {len(models)} 个真实模型的结果,需≥2个才能算。"
              f"对 mpnet/bge 各再跑一次即可自动生成。"); return
    print(f"\n=== 跨模型 δ 排序一致性 (Spearman, 共 {len(models)} 模型) ===")
    header="        "+"".join(f"{m[:8]:>10}" for m in models); print(header)
    matrix={}
    for mi in models:
        row={}
        line=f"{mi[:8]:>8}"
        for mj in models:
            common=[u for u in model_delta[mi] if u in model_delta[mj]]
            if len(common)<3:
                rho=float("nan")
            else:
                a=[model_delta[mi][u] for u in common]; b=[model_delta[mj][u] for u in common]
                rho=float(spearmanr(a,b).correlation)
            row[mj]=None if rho!=rho else round(rho,3)
            line+=f"{(row[mj] if row[mj] is not None else 'NA'):>10}"
        matrix[mi]=row; print(line)
    json.dump({"models":models,"spearman":matrix},
              open(os.path.join(out_dir,"cross_model_delta_consistency.json"),"w",encoding="utf-8"),
              ensure_ascii=False,indent=2)
    print(f"  → 一致性矩阵写入 {out_dir}/cross_model_delta_consistency.json")

if __name__=="__main__":
    main()
