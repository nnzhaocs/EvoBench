#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_step4_multimodel.py
=============================
用 benchmark 评估「重嵌入/索引维护策略」—— 这是 benchmark 判别力的核心证据（EA&B 审稿人最看重）。

回答的问题：有了这个 benchmark，能不能把不同的重算策略排出差异、且排序符合直觉？
若能（recall/cost 拉得开、audit-threshold 在中间取得好 trade-off），则 benchmark 有判别力，
是「benchmark」而非「a characterized dataset」。这是评审指出的、差的那条腿。

评估的三类策略（policy sweep）：
  - always-rebuild：每次更新都重算 emb（recall 上界，cost 上界）
  - never-rebuild ：从不重算（cost=0，recall 随漂移退化，下界）
  - audit-threshold(τ)：δ>τ 才重算（本文方法；扫 τ 得到 trade-off 曲线）

输出：一张 sweep 表（policy × [recall, 重算次数/cost, staleness]），即 benchmark 论文的 Table 1。

★★★ 铁律 ★★★
  真 MiniLM（--verify-pipeline-only 仅验管道，假数标记不进论文）。
  只在真实 GT 上算 recall，不合成。

依赖 step2.5/step3 产出的四件套（units/updates/queries.txt/ground_truth.json）。
用法：
  python build_step4_multimodel.py --model minilm --bench-final ./bench_final --out ./sweep_result
"""
import argparse, os, json, sys, glob
import numpy as np


# ---- 多模型注册表：每类模型的加载方式不同，必须分别处理对 ----
# kind:
#   "st"       -> SentenceTransformer 原生（MiniLM/MPNet），直接 encode
#   "st_bge"   -> SentenceTransformer 原生，但 BGE 检索需给查询加指令前缀
#   "hf_mean"  -> transformers 原生编码器（CodeBERT），需手动 mean-pooling
MODEL_REGISTRY = {
    "minilm":   {"name": "all-MiniLM-L6-v2",              "kind": "st",      "dim": 384},
    "mpnet":    {"name": "all-mpnet-base-v2",             "kind": "st",      "dim": 768},
    "bge":      {"name": "BAAI/bge-small-en-v1.5",        "kind": "st_bge",  "dim": 384},
    "codebert": {"name": "microsoft/codebert-base",       "kind": "hf_mean", "dim": 768},
    # ---- 代码专用嵌入模型（已核实 arXiv2310.16803：均用 mean-pooling）----
    "graphcodebert": {"name": "microsoft/graphcodebert-base",        "kind": "hf_mean",       "dim": 768},
    "unixcoder":     {"name": "microsoft/unixcoder-base",            "kind": "hf_mean",       "dim": 768},
    "starencoder":   {"name": "bigcode/starencoder",                 "kind": "hf_mean",       "dim": 768},
    "jina_code":     {"name": "jinaai/jina-embeddings-v2-base-code", "kind": "hf_mean_trust", "dim": 768},
    "codet5p":       {"name": "Salesforce/codet5p-110m-embedding",   "kind": "hf_mean_trust", "dim": 256},
}
# BGE 官方建议：检索查询前加此指令（文档/单元不加）。代码单元当作"文档"，不加前缀。
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_key="minilm", verify_only=False):
        self.fake = verify_only
        self.model = None
        self.tok = None
        self.kind = None
        self.key = model_key
        if verify_only:
            print("  ⚠️ 验管道模式：假嵌入，数字不真实、不进论文")
            return
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"未知模型 {model_key}，可选: {list(MODEL_REGISTRY)}")
        cfg = MODEL_REGISTRY[model_key]
        self.kind = cfg["kind"]
        if self.kind in ("st", "st_bge"):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(cfg["name"])
            got = self.model.encode(["x"], normalize_embeddings=True).shape[1]
            assert got == cfg["dim"], f"{model_key} 维度应 {cfg['dim']} 实为 {got}"
            print(f"  ✅ 真实 {model_key}（{cfg['name']}, {got}维, ST原生）")
        elif self.kind in ("hf_mean", "hf_mean_trust"):
            # 代码编码器：transformers + mean-pooling（这些不是句向量模型，不能直接 encode）
            # 已核实(arXiv2310.16803)：CodeBERT/GraphCodeBERT/UniXcoder/StarEncoder 代码检索均用 mean-pool
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.torch = torch
            trust = (self.kind == "hf_mean_trust")   # jina/codet5p 需 trust_remote_code
            self.tok = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=trust)
            self.model = AutoModel.from_pretrained(cfg["name"], trust_remote_code=trust)
            self.model.eval()
            # 软维度校验：不同权重版本维度可能微调，不强退，只提醒（避免"维度错但静默"）
            try:
                probe = self._encode_hf(["def f(): pass"])
                if probe.shape[1] != cfg["dim"]:
                    print(f"  ⚠️ {model_key} 实际维度 {probe.shape[1]} ≠ 登记 {cfg['dim']}，请核对")
            except Exception:
                pass
            print(f"  ✅ 真实 {model_key}（{cfg['name']}, {cfg['dim']}维, HF+mean-pool{'+trust' if trust else ''}）")

    def _mean_pool(self, last_hidden, mask):
        # 对 token 表示做 mean-pooling（用 attention mask 排除 padding）
        m = mask.unsqueeze(-1).float()
        summed = (last_hidden * m).sum(1)
        counts = m.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts, is_query=False):
        if isinstance(texts, str):
            texts = [texts]
        if self.fake:
            import hashlib
            out = []
            for t in texts:
                h = hashlib.md5(t.encode()).digest()
                v = np.frombuffer(h * 24, dtype=np.uint8)[:64].astype("float32") - 128
                out.append(v / (np.linalg.norm(v) + 1e-9))
            return np.array(out, dtype="float32")
        if self.kind in ("st", "st_bge"):
            enc_texts = texts
            if self.kind == "st_bge" and is_query:
                enc_texts = [BGE_QUERY_PREFIX + t for t in texts]  # 仅查询加前缀
            return self.model.encode(enc_texts, normalize_embeddings=True,
                                     convert_to_numpy=True, batch_size=64).astype("float32")
        elif self.kind in ("hf_mean", "hf_mean_trust"):
            return self._encode_hf(texts)

    def _encode_hf(self, texts):
        torch = self.torch
        out = []
        B = 32
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
            with torch.no_grad():
                o = self.model(**enc)
            # 取 last_hidden_state（有的模型返回 tuple，o[0] 即是）
            lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            v = self._mean_pool(lhs, enc["attention_mask"])
            v = torch.nn.functional.normalize(v, p=2, dim=1)  # L2归一，和ST口径一致
            out.append(v.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0)


def load_jsonl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]


def recall_at_k(U, Q, qtexts, gt_by_text, unit_ids, k=10):
    """在当前库向量 U 上，算所有有 GT 查询的平均 recall@k（真实 GT）。"""
    id2idx={uid:i for i,uid in enumerate(unit_ids)}
    recs=[]
    for qi,qt in enumerate(qtexts):
        gt=gt_by_text.get(qt)
        if not gt: continue
        gt_idx=set(id2idx[u] for u in gt if u in id2idx)
        if not gt_idx: continue
        top=set(np.argsort(-(U@Q[qi]))[:k])
        recs.append(len(top & gt_idx)/len(gt_idx))
    return float(np.mean(recs)) if recs else 0.0


def run_policy(policy, tau, units, updates, U0, U_new_map, Q, qtexts, gt, unit_ids, k):
    """模拟一条 policy 走完所有更新后的库状态，返回 (recall, n_recompute)。
    U0=初始(旧)向量; U_new_map: unit_id->新向量(更新后)。"""
    U=U0.copy()
    id2idx={uid:i for i,uid in enumerate(unit_ids)}
    n_recompute=0
    for up in updates:
        uid=up["unit_id"]
        if uid not in id2idx or uid not in U_new_map: continue
        idx=id2idx[uid]
        delta=1.0-float(U0[idx] @ U_new_map[uid])  # 这次更新的真实漂移
        do=False
        if policy=="always": do=True
        elif policy=="never": do=False
        elif policy=="audit": do = (delta > tau)
        if do:
            U[idx]=U_new_map[uid]; n_recompute+=1
        # never/未触发：库里保留旧向量（= stale）
    rec=recall_at_k(U, Q, qtexts, gt, unit_ids, k)
    return rec, n_recompute


def process_repo(rd, emb, k, taus):
    name=os.path.basename(rd.rstrip("/"))
    units=load_jsonl(os.path.join(rd,"units.jsonl"))
    updates=load_jsonl(os.path.join(rd,"updates.jsonl"))
    qtexts=[l.strip() for l in open(os.path.join(rd,"queries.txt"),encoding="utf-8") if l.strip()]
    gt=json.load(open(os.path.join(rd,"ground_truth.json"),encoding="utf-8"))
    if not units or not qtexts or not gt:
        print(f"  ⚠️ {name}: 缺 units/queries/gt，跳过"); return None
    unit_ids=[u["unit_id"] for u in units]
    uid_set=set(unit_ids)
    # ---- 版本对齐修复（关键）----
    # never/always 必须比较"同一函数的更新前 vs 更新后"，而不是 units.text（多为不相干版本）。
    # 所以：初始"旧"库 U0 中，被更新覆盖的单元用其"第一次更新的 old_text"（=真正的更新前版本）；
    #        未被任何更新覆盖的单元才回退到 units.text。
    # 新向量 U_new_map 用"最后一次更新的 new_text"（=更新后最终态）。
    valid=[u for u in updates if u.get("new_text") and u["unit_id"] in uid_set]
    # 每个单元第一次更新的 old_text（保序取首个）
    first_old={}
    last_new={}
    for u in valid:
        uid=u["unit_id"]
        if uid not in first_old and (u.get("old_text") or "").strip():
            first_old[uid]=u["old_text"]
        last_new[uid]=u["new_text"]  # 后者覆盖=最终新版本
    # 构造 U0 的文本：被更新单元用 first_old（无 old_text 时退回 units.text），其余用 units.text
    u0_texts=[]
    for u in units:
        uid=u["unit_id"]
        u0_texts.append(first_old.get(uid, u.get("text","")))
    n_old_aligned=sum(1 for u in units if u["unit_id"] in first_old)
    print(f"  [版本对齐] {n_old_aligned}/{len(units)} 个单元用 old_text 作更新前基准（其余用 units.text）")
    U0=emb.encode(u0_texts)
    Q=emb.encode(qtexts, is_query=True)
    U_new_map={}
    if last_new:
        keys=list(last_new.keys())
        NEW=emb.encode([last_new[uid] for uid in keys])
        for uid,v in zip(keys,NEW): U_new_map[uid]=v
    # ground_truth key 是查询文本
    rows=[]
    r_always,c_always=run_policy("always",0,units,valid,U0,U_new_map,Q,qtexts,gt,unit_ids,k)
    r_never,c_never=run_policy("never",0,units,valid,U0,U_new_map,Q,qtexts,gt,unit_ids,k)
    rows.append(("always-rebuild", r_always, c_always))
    rows.append(("never-rebuild",  r_never,  c_never))
    for tau in taus:
        r,cc=run_policy("audit",tau,units,valid,U0,U_new_map,Q,qtexts,gt,unit_ids,k)
        rows.append((f"audit(τ={tau})", r, cc))
    total_updates=len(valid)
    print(f"\n=== {name}（{total_updates} 更新, {len(unit_ids)} 单元）===")
    print(f"  {'policy':16} {'recall@'+str(k):10} {'重算次数':10} {'省算%':8}")
    for pol,rec,cost in rows:
        save=100*(1-cost/max(c_always,1))
        print(f"  {pol:16} {rec:10.4f} {cost:10} {save:7.1f}%")
    return {"repo":name,"n_updates":total_updates,"n_units":len(unit_ids),
            "always_recall":r_always,"always_cost":c_always,
            "never_recall":r_never,
            "sweep":[{"policy":p,"recall":r,"recompute":c,
                      "save_pct":round(100*(1-c/max(c_always,1)),1)} for p,r,c in rows]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bench-final",default="./bench_final")
    ap.add_argument("--out",default="./sweep_result")
    ap.add_argument("--k",type=int,default=10)
    ap.add_argument("--taus",default="0.02,0.05,0.1,0.2",
                    help="audit-threshold 扫的 τ 值(逗号分隔)")
    ap.add_argument("--model",default="minilm",
                    help="minilm/mpnet/bge/codebert（多模型对照，逐个跑）")
    ap.add_argument("--verify-pipeline-only",action="store_true")
    args=ap.parse_args()
    taus=[float(x) for x in args.taus.split(",")]

    print("【自检】模型="+args.model+("（验管道,假嵌入）" if args.verify_pipeline_only else "（真实嵌入）"))
    try:
        emb=Embedder(model_key=args.model, verify_only=args.verify_pipeline_only)
    except Exception as e:
        print(f"  ❌ 加载模型 {args.model} 失败:{e}\n  → 换能连HF的机器;绝不用假嵌入出真数。"); sys.exit(1)

    rds=sorted(d for d in glob.glob(os.path.join(args.bench_final,"*"))
               if os.path.isdir(d) and os.path.exists(os.path.join(d,"queries.txt")))
    if not rds:
        print(f"❌ {args.bench_final} 下无四件套。先跑 step2.5。"); sys.exit(1)
    os.makedirs(args.out,exist_ok=True)
    results=[process_repo(rd,emb,args.k,taus) for rd in rds]
    results=[r for r in results if r]
    fn=("PIPELINE_VERIFY_ONLY_sweep_%s.json"%args.model) if args.verify_pipeline_only else ("baseline_sweep_%s.json"%args.model)
    json.dump({"model":args.model,"k":args.k,"taus":taus,"fake_embed":args.verify_pipeline_only,
               "per_repo":results},
              open(os.path.join(args.out,fn),"w",encoding="utf-8"),
              ensure_ascii=False,indent=2)
    print(f"\n{'✅ 真实' if not args.verify_pipeline_only else '⚠️ 验管道'} sweep 存到 {fn}")
    print("这张表 = benchmark 论文的 Table 1（判别力证据）。")
    print("解读：always recall 最高 cost 最高、never cost=0 recall 最低、")
    print("      audit 随 τ 增大省算增多但 recall 缓降——若 audit 能在小 recall 损失下大幅省算，")
    print("      则证明 benchmark 有判别力 + 你的方法有价值。")


if __name__=="__main__":
    main()
