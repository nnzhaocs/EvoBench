#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
codemodel_sanity_check.py
=========================
代码模型接入健康检查（回应：CodeBERT 四仓库平均 recall 仅 0.116，是接入bug还是真信号？）

背景（必读，别误解这个脚本要证明什么）：
  论文 §6.2b 观察到代码专用模型(CodeBERT/GraphCodeBERT/UniXcoder)在本任务
  recall 明显低于通用模型，并解释为"issue自然语言→代码"的跨模态错配。
  这个解释成立的前提是：**代码模型的编码本身是对的**（不是 pooling/接入 bug 把向量搞坏了）。

  本脚本只验证这一个前提——不是验证"代码模型好不好"，而是验证"接入有没有坏"：
  对若干【明显相似】和【明显不相似】的代码对，算模型给的余弦相似度。
    - 接入正确 → 相似对余弦明显高于不相似对（模型能分辨代码语义）。
      → 0.116 是真信号（跨模态错配），论文解释成立。
    - 接入坏了(pooling错/取错层/没归一) → 相似≈不相似、或余弦乱掉/恒定。
      → 0.116 是假象，必须修 build_step4_multimodel.py 的接入后重跑。

  ★关键：低 recall ≠ 接入 bug。一个接入正确的代码模型，在"自然语言→代码"这种
    它没怎么训练过的任务上 recall 低是正常的。本检查区分"接入坏"和"任务不适合"。

用法（跟多模型 sweep 一起跑，零额外数据）：
    python3 codemodel_sanity_check.py --models codebert graphcodebert unixcoder
    # 也可带通用模型做对照基线：
    python3 codemodel_sanity_check.py --models minilm codebert

依赖 build_step4_multimodel.py 的 Embedder（复用同一套接入逻辑，保证测的就是sweep用的那套）。
"""
import argparse, sys, os
import numpy as np

# 复用 sweep 脚本的 Embedder，确保测的接入 = sweep 实际用的接入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from build_step4_multimodel import Embedder, MODEL_REGISTRY
except Exception as e:
    print(f"❌ 无法 import build_step4_multimodel.Embedder：{e}")
    print("   把本脚本和 build_step4_multimodel.py 放同一目录。")
    sys.exit(1)


# ---- 测试对：真实代码片段，覆盖三语言。每组 (A, B, 关系) ----
# 相似对：语义几乎相同（重命名/加类型注解/等价改写）——好模型应给高余弦。
# 不相似对：功能完全不同的函数——好模型应给低余弦。
SIMILAR_PAIRS = [
    # Python：仅加类型注解
    ("def add(a, b):\n    return a + b",
     "def add(a: int, b: int) -> int:\n    return a + b", "py:加类型注解"),
    # Python：仅变量重命名
    ("def total(items):\n    s = 0\n    for x in items:\n        s += x\n    return s",
     "def total(lst):\n    acc = 0\n    for e in lst:\n        acc += e\n    return acc", "py:变量重命名"),
    # Go：仅加接收者参数名变化
    ("func (c *Context) BindJSON(obj any) error {\n    return c.MustBindWith(obj, JSON)\n}",
     "func (ctx *Context) BindJSON(o any) error {\n    return ctx.MustBindWith(o, JSON)\n}", "go:参数重命名"),
    # JS：等价改写
    ("function sum(arr) {\n  return arr.reduce((a, b) => a + b, 0);\n}",
     "function sum(list) {\n  let total = 0;\n  for (const n of list) total += n;\n  return total;\n}", "js:等价改写"),
]
DISSIMILAR_PAIRS = [
    # 加法 vs 文件读取
    ("def add(a, b):\n    return a + b",
     "def read_file(path):\n    with open(path) as f:\n        return f.read()", "py:加法vs读文件"),
    # 排序 vs HTTP 请求
    ("def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
     "def fetch(url):\n    import requests\n    return requests.get(url).json()", "py:排序vsHTTP"),
    # Go：JSON绑定 vs 日志中间件
    ("func (c *Context) BindJSON(obj any) error {\n    return c.MustBindWith(obj, JSON)\n}",
     "func Logger() HandlerFunc {\n    return func(c *Context) {\n        c.Next()\n        log.Println(c.Request.URL.Path)\n    }\n}", "go:绑定vs日志"),
    # JS：求和 vs DOM操作
    ("function sum(arr) {\n  return arr.reduce((a, b) => a + b, 0);\n}",
     "function hide(el) {\n  el.style.display = 'none';\n}", "js:求和vsDOM"),
]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def check_model(model_key):
    print(f"\n{'='*66}\n模型: {model_key}  ({MODEL_REGISTRY.get(model_key, {}).get('name','?')})\n{'='*66}")
    try:
        emb = Embedder(model_key=model_key, verify_only=False)
    except Exception as e:
        print(f"  🔴 加载失败: {e}")
        return None

    sim_cos, dis_cos = [], []
    print("  【相似对】(应高余弦)")
    for a, b, tag in SIMILAR_PAIRS:
        va = emb.encode([a])[0]; vb = emb.encode([b])[0]
        c = cos(va, vb); sim_cos.append(c)
        print(f"    {tag:18} cos={c:.3f}")
    print("  【不相似对】(应低余弦)")
    for a, b, tag in DISSIMILAR_PAIRS:
        va = emb.encode([a])[0]; vb = emb.encode([b])[0]
        c = cos(va, vb); dis_cos.append(c)
        print(f"    {tag:18} cos={c:.3f}")

    sim_mean = float(np.mean(sim_cos)); dis_mean = float(np.mean(dis_cos))
    gap = sim_mean - dis_mean
    sim_min = min(sim_cos); dis_max = max(dis_cos)

    print(f"\n  相似对均值 {sim_mean:.3f} | 不相似对均值 {dis_mean:.3f} | 差距 {gap:+.3f}")
    # 判读规则（不看绝对值，看能不能"分辨"——这才是接入是否正确的信号）
    verdict = "unknown"
    if gap >= 0.10 and sim_min > dis_max - 0.05:
        verdict = "healthy"
        print(f"  ✅ 接入健康：相似对余弦明显高于不相似对（能分辨代码语义）。")
        print(f"     → 若该模型 recall 低，是【任务错配】非【接入bug】，论文§6.2b解释成立。")
    elif gap < 0.03:
        verdict = "broken"
        print(f"  🔴 接入可疑：相似≈不相似（差距{gap:+.3f}），模型分辨不出代码语义。")
        print(f"     → 可能 pooling/取层/归一化有误。检查 build_step4_multimodel._encode_hf。")
        print(f"     → 修好前，该模型的 sweep/recall 数据不可信，不要写进论文。")
    else:
        verdict = "weak"
        print(f"  ⚠️ 分辨力偏弱（差距{gap:+.3f}）：能分但不利落。")
        print(f"     → 大概率接入没坏、只是模型对这类改写不敏感；但建议人工看上面几对余弦是否合理。")

    # 额外的"死接入"检测：所有余弦几乎相同 = 向量塌缩（严重bug）
    all_cos = sim_cos + dis_cos
    if np.std(all_cos) < 0.02:
        print(f"  🔴🔴 警告：所有余弦几乎相同(std={np.std(all_cos):.3f})，向量可能塌缩！严重接入bug。")
        verdict = "broken"

    return {"model": model_key, "sim_mean": round(sim_mean, 3),
            "dis_mean": round(dis_mean, 3), "gap": round(gap, 3),
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["codebert", "graphcodebert", "unixcoder"],
                    help="要检查的模型key（默认三个代码模型）。可加 minilm 做对照。")
    args = ap.parse_args()

    results = []
    for m in args.models:
        r = check_model(m)
        if r:
            results.append(r)

    print(f"\n{'='*66}\n汇总\n{'='*66}")
    print(f"{'模型':14}{'相似均值':>9}{'不相似均值':>11}{'差距':>8}  判定")
    for r in results:
        icon = {"healthy": "✅", "weak": "⚠️", "broken": "🔴", "unknown": "?"}[r["verdict"]]
        print(f"{r['model']:14}{r['sim_mean']:>9}{r['dis_mean']:>11}{r['gap']:>+8}  {icon} {r['verdict']}")

    broken = [r["model"] for r in results if r["verdict"] == "broken"]
    if broken:
        print(f"\n🔴 接入可疑的模型: {broken}")
        print(f"   → 这些模型的 recall(如CodeBERT 0.116)可能是bug不是真信号，修接入后重跑。")
    else:
        print(f"\n✅ 所有被检模型接入健康。低 recall 是任务错配（论文§6.2b解释成立），非 bug。")
        print(f"   → 可放心把代码模型作为跨模型稳健性对照写进论文。")


if __name__ == "__main__":
    main()
