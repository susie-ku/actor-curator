"""Collect peak validation accuracy within the first 100 steps from TensorBoard logs.

usage: python repro/collect_results.py [tensorboard_log/actor_curator_repro]
Prints one row per (experiment, data_source) and a Table-1-style summary next to the paper's numbers.
"""
import glob, os, sys
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = sys.argv[1] if len(sys.argv) > 1 else "tensorboard_log/actor_curator_repro"
MAX_STEP = 100
PAPER = {  # Table 1 (Qwen2.5-3B): benchmark -> {method: value}
    "countdown":      dict(uniform=44.74, sec=58.87, pcl=57.24, ac=62.12),
    "countdown_hard": dict(uniform=41.00, sec=51.50, pcl=48.00, ac=58.00),
    "zebra":          dict(uniform=35.12, sec=36.00, pcl=34.12, ac=37.62),
    "zebra_hard":     dict(uniform=30.50, sec=27.50, pcl=26.00, ac=34.50),
    "arc":            dict(uniform=26.74, sec=27.87, pcl=26.37, ac=36.37),
    "arc_hard":       dict(uniform=19.50, sec=18.50, pcl=18.50, ac=31.00),
    "math500":        dict(uniform=83.00, sec=81.00, pcl=79.79, ac=81.00),
    "aime24":         dict(uniform=23.33, sec=20.00, pcl=23.33, ac=30.00),
}

def load(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0}); ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        if tag.startswith("val-core/") and tag.endswith("/mean@1"):
            out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out

rows = []
for run_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    exp = os.path.basename(run_dir)
    if exp.startswith("smoke_"): continue
    for tag, pts in load(run_dir).items():
        src = tag.split("/")[1]
        pts = [(s, v) for s, v in pts if s <= MAX_STEP]
        if not pts: continue
        peak_step, peak = max(pts, key=lambda p: p[1])
        rows.append((exp, src, 100 * peak, peak_step, max(s for s, _ in pts), 100 * pts[0][1]))

print(f"{'experiment':40s} {'data_source':22s} {'peak%':>7s} {'@step':>5s} {'last_eval':>9s} {'step0%':>7s}")
for r in rows: print(f"{r[0]:40s} {r[1]:22s} {r[2]:7.2f} {r[3]:5d} {r[4]:9d} {r[5]:7.2f}")

# Table-1 style summary (paper rows: "<bench>" = mean over the 4 test levels, "<bench>_hard" = very_hard level; MATH: test_math=MATH500, test_aime=AIME24)
def paper_key(bench, src):
    if src == "test_math": return "math500"
    if src == "test_aime": return "aime24"
    if src.endswith("-very_hard-test"): return f"{bench}_hard"
    if src == "mean4": return bench
    return None
summary = defaultdict(dict)   # (bench, src) -> method -> (peak, last)
per_run = defaultdict(dict)   # (bench, method) -> src -> list of (step, value)
for run_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    exp = os.path.basename(run_dir)
    if exp.startswith("smoke_"): continue
    bench, method = exp.split("_")[0], exp.split("_")[1]
    for tag, pts in load(run_dir).items():
        per_run[(bench, method)][tag.split("/")[1]] = [(s, v) for s, v in pts if s <= MAX_STEP]
for (bench, method), srcs in per_run.items():
    for src, pts in srcs.items():
        if pts:
            peak_step, peak = max(pts, key=lambda q: q[1]); summary[(bench, src)][method] = (100 * peak, max(s for s, _ in pts))
    levels = [src for src in srcs if src.endswith("-test")]
    if len(levels) == 4:   # mean over the 4 levels at each eval step, then peak
        steps = sorted(set(s for src in levels for s, _ in srcs[src]))
        means = []
        for st in steps:
            vals = [v for src in levels for s, v in srcs[src] if s == st]
            if len(vals) == 4: means.append((st, sum(vals) / 4))
        if means:
            peak_step, peak = max(means, key=lambda q: q[1]); summary[(bench, "mean4")][method] = (100 * peak, max(s for s, _ in means))
print("\n| benchmark / data_source | Uniform | SEC | PCL | AC | paper (Uniform / SEC / PCL / AC) |")
print("|---|---|---|---|---|---|")
for (bench, src), d in sorted(summary.items()):
    cells = []
    for m in ["uniform", "sec", "pcl", "ac"]:
        if m in d:
            peak, last = d[m]; cells.append(f"{peak:.2f}" + ("" if last >= MAX_STEP else f" (≤{last})"))
        else: cells.append("—")
    p = PAPER.get(paper_key(bench, src) or "", {})
    pap = " / ".join(f"{p[m]:.2f}" if m in p else "—" for m in ["uniform", "sec", "pcl", "ac"]) if p else "—"
    print(f"| {bench} / {src} | " + " | ".join(cells) + f" | {pap} |")
