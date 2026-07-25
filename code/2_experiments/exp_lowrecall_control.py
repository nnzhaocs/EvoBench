#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低-recall 天花板对照实验（回应审稿人 weakness #1）
==================================================
审稿人质疑：always/never 差异小(±0.02)，但可能只是因为 recall 绝对值太低
（gin 0.36），检索器本来就没检对，所以动不动都无所谓——这跟"重算无收益"
是两码事，天花板太低时任何策略都拉不开。

本实验排除这个替代解释：只在"检索器本来就能命中的查询子集"上重算 always/never。
若命中子集上 always/never 仍贴合(±0.02) → 差异小不是低recall退化，Finding 1 稳；
若明显拉开 → 低recall确实掩盖了差异，Finding 1 需改写。

命中子集用三个口径同时报告，以堵死审稿人可能的两个追问：
  (1) hit_union  —— GT 在 always 或 never 任一状态进 top-k（并集）。
      修正原版"只看 always 命中"的 selection bias：只看 always 会天然排除
      "只在旧向量下命中"的查询，人为压低 never 的劣势、偏袒 null 结果。并集不偏袒任何一方。
  (2) hit_always —— 仅 always 命中（保留原口径，作对照，便于看两口径是否结论一致）。
  (3) gtle_k     —— 在 hit_union 基础上再限定 |GT|<=k 的查询。
      修正"分母压低"：大 commit 的 GT 指向几十个函数，k=10 装不下，
      recall=命中数/|GT| 上限就<1，这部分低 recall 是分母造成的、与检索器无关，
      会把 0.36 进一步误读成"检索器差"。限定 |GT|<=k 后 recall 上限恢复为 1。

判读：以 hit_union 为主口径。gtle_k 作为"最干净"口径复核。
三个口径下 |子集Δ| 都在 ±0.02 内 → Finding 1 最稳。

用法（需真 MiniLM）：
    python3 exp_lowrecall_control.py --bench-final 清洗后数据_576条 --out lowrecall_out
"""
import json, os, sys, glob, argparse
import numpy as np

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def valid_queries(Q, qtexts, gt_by_text, unit_ids):
    """返回有有效GT的查询 qi -> gt_idx 集合，及其 |GT| 大小。"""
    id2idx={uid:i for i,uid in enumerate(unit_ids)}
    out={}
    for qi,qt in enumerate(qtexts):
        gt=gt_by_text.get(qt)
        if not gt: continue
        gt_idx=set(id2idx[u] for u in gt if u in id2idx)
        if not gt_idx: continue
        out[qi]=gt_idx
    return out

def hit_in_state(U, Q, qi, gt_idx, k=10):
    """该库状态下 GT 是否至少1个进 top-k。"""
    top=set(np.argsort(-(U@Q[qi]))[:k])
    return len(top & gt_idx)>0

def recall_on_subset(U, Q, vq, subset_qi, k=10):
    """只在 subset_qi 这些查询上算平均 recall。vq: qi->gt_idx。"""
    recs=[]
    for qi in subset_qi:
        gt_idx=vq.get(qi)
        if not gt_idx: continue
        top=set(np.argsort(-(U@Q[qi]))[:k])
        recs.append(len(top & gt_idx)/len(gt_idx))
    return float(np.mean(recs)) if recs else 0.0

def build_lib_state(policy, tau, U0, U_new_map, unit_ids, updates):
    """走完所有更新后的库向量状态。"""
    U=U0.copy()
    id2idx={uid:i for i,uid in enumerate(unit_ids)}
    for up in updates:
        uid=up["unit_id"]
        if uid not in id2idx or uid not in U_new_map: continue
        idx=id2idx[uid]
        delta=1.0-float(U0[idx] @ U_new_map[uid])
        do = (policy=="always") or (policy=="audit" and delta>tau)
        if do: U[idx]=U_new_map[uid]
    return U

def process(rd, emb, k=10):
    name=os.path.basename(rd.rstrip("/"))
    units=load_jsonl(os.path.join(rd,"units.jsonl"))
    updates=load_jsonl(os.path.join(rd,"updates.jsonl"))
    qtexts=[l.strip() for l in open(os.path.join(rd,"queries.txt"),encoding="utf-8") if l.strip()]
    gt=json.load(open(os.path.join(rd,"ground_truth.json"),encoding="utf-8"))
    if not units or not qtexts or not gt: return None
    unit_ids=[u["unit_id"] for u in units]
    uid_set=set(unit_ids)

    # 版本对齐(同修复版)：U0用first_old, U_new用last_new
    valid=[u for u in updates if u.get("new_text") and u["unit_id"] in uid_set]
    first_old={}; last_new={}
    for u in valid:
        uid=u["unit_id"]
        if uid not in first_old and (u.get("old_text") or "").strip():
            first_old[uid]=u["old_text"]
        last_new[uid]=u["new_text"]
    u0_texts=[first_old.get(u["unit_id"], u.get("text","")) for u in units]
    # ★归一化必须与 step4/probe 一致：不归一化则 U@Q 不是余弦、δ 不是余弦距离，
    #   recall 与 audit-τ 会静默偏离其他实验，破坏"同 MiniLM 同度量"的一致性。
    U0=emb.encode(u0_texts, normalize_embeddings=True, show_progress_bar=False)
    Q=emb.encode(qtexts, normalize_embeddings=True, show_progress_bar=False)
    U_new_map={}
    if last_new:
        keys=list(last_new.keys())
        NEW=emb.encode([last_new[uid] for uid in keys],
                       normalize_embeddings=True, show_progress_bar=False)
        for uid,v in zip(keys,NEW): U_new_map[uid]=v

    # always库状态 = 理论最优；never = 全旧向量
    U_always=build_lib_state("always",0,U0,U_new_map,unit_ids,valid)
    U_never =build_lib_state("never",0,U0,U_new_map,unit_ids,valid)

    # 有有效GT的全部查询
    vq=valid_queries(Q,qtexts,gt,unit_ids)   # qi -> gt_idx
    all_qi=list(vq.keys())

    # 三个命中子集口径
    hit_union=[]      # (1) always 或 never 任一命中——主口径，无 selection bias
    hit_always=[]     # (2) 仅 always 命中——原口径对照
    for qi,gt_idx in vq.items():
        h_a=hit_in_state(U_always,Q,qi,gt_idx,k)
        h_n=hit_in_state(U_never ,Q,qi,gt_idx,k)
        if h_a or h_n: hit_union.append(qi)
        if h_a:        hit_always.append(qi)
    # (3) 在 union 基础上再限 |GT|<=k——去掉分母压低
    gtle_k=[qi for qi in hit_union if len(vq[qi])<=k]

    def diff(subset):
        ra=recall_on_subset(U_always,Q,vq,subset,k)
        rn=recall_on_subset(U_never ,Q,vq,subset,k)
        return round(ra,4),round(rn,4),round(ra-rn,4)

    a_all,n_all,d_all=diff(all_qi)
    a_uni,n_uni,d_uni=diff(hit_union)
    a_alw,n_alw,d_alw=diff(hit_always)
    a_gtk,n_gtk,d_gtk=diff(gtle_k)

    return {
        "repo":name,
        "n_queries":len(all_qi),
        "n_hit_union":len(hit_union),
        "n_hit_always":len(hit_always),
        "n_gtle_k":len(gtle_k),
        "hit_rate_union":round(len(hit_union)/max(len(all_qi),1),3),
        # 全集
        "全集_always":a_all,"全集_never":n_all,"全集_Δ":d_all,
        # 主口径：并集命中子集
        "并集_always":a_uni,"并集_never":n_uni,"并集_Δ":d_uni,
        # 对照：仅always命中
        "仅always命中_always":a_alw,"仅always命中_never":n_alw,"仅always命中_Δ":d_alw,
        # 最干净：并集 且 |GT|<=k
        "GTleqk_always":a_gtk,"GTleqk_never":n_gtk,"GTleqk_Δ":d_gtk,
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--bench-final",required=True)
    ap.add_argument("--out",default="lowrecall_out")
    ap.add_argument("--k",type=int,default=10)
    args=ap.parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ 需要 sentence-transformers"); sys.exit(1)
    emb=SentenceTransformer("all-MiniLM-L6-v2")
    os.makedirs(args.out,exist_ok=True)

    results=[]
    for rd in sorted(glob.glob(os.path.join(args.bench_final,"*"))):
        if not os.path.isdir(rd): continue
        r=process(rd,emb,args.k)
        if r: results.append(r)

    print("="*110)
    print("低-recall 对照实验：全集 vs 三种命中子集口径下的 always/never 差异")
    print("  主口径 = 并集(always∪never命中，无selection bias)；GTleqk = 并集且|GT|<=k(去分母压低)")
    print("="*110)
    hdr=(f"{'仓库':9}{'查询':>5}{'并集n':>6}{'GTlekn':>7}"
         f"  |{'全集Δ':>8}"
         f"  |{'并集_alw':>9}{'并集_nev':>9}{'并集Δ':>8}"
         f"  |{'仅alwΔ':>8}"
         f"  |{'GTk_alw':>9}{'GTk_nev':>9}{'GTkΔ':>8}")
    print(hdr)
    for r in results:
        print(f"{r['repo']:9}{r['n_queries']:>5}{r['n_hit_union']:>6}{r['n_gtle_k']:>7}"
              f"  |{r['全集_Δ']:>+8}"
              f"  |{r['并集_always']:>9}{r['并集_never']:>9}{r['并集_Δ']:>+8}"
              f"  |{r['仅always命中_Δ']:>+8}"
              f"  |{r['GTleqk_always']:>9}{r['GTleqk_never']:>9}{r['GTleqk_Δ']:>+8}")
    json.dump(results,open(os.path.join(args.out,"lowrecall_control.json"),"w"),ensure_ascii=False,indent=2)

    print("\n"+"="*110)
    print("判读（以【并集Δ】为主口径，【GTkΔ】为最干净复核）：")
    print("  三口径 |Δ| 都在 ±0.02 内 → 差异小不是低recall退化，Finding1 最稳，堵死两个追问 ✅")
    print("  某口径 |Δ| 明显放大        → 低recall/分母确实掩盖了差异，该口径下 Finding1 需改写 🔴")
    avg_all =np.mean([abs(r['全集_Δ'])            for r in results])
    avg_uni =np.mean([abs(r['并集_Δ'])            for r in results])
    avg_alw =np.mean([abs(r['仅always命中_Δ'])    for r in results])
    avg_gtk =np.mean([abs(r['GTleqk_Δ'])          for r in results])
    print(f"\n  平均|全集Δ|={avg_all:.4f}  |并集Δ|={avg_uni:.4f}  |仅alwaysΔ|={avg_alw:.4f}  |GTleqkΔ|={avg_gtk:.4f}")
    verdict=[]
    for name,v in [("并集",avg_uni),("GTleqk",avg_gtk)]:
        if v<0.03: verdict.append(f"{name}稳✅")
        elif v>avg_all*2: verdict.append(f"{name}放大🔴")
        else: verdict.append(f"{name}中间⚠")
    print("  → 主口径判读：" + "  ".join(verdict))
    print("  （若并集与GTleqk结论一致且都<0.03，是最强结果；若两者分歧，说明分母效应在起作用，需在论文里分列报告）")
