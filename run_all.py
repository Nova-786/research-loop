"""Run both autonomous research loops and write a results summary + plot.

Usage: python3 run_all.py [budget_per_domain]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medical_domain import MedicalDomain
from research_loop import EvolutionaryProposer, run
from robotics_domain import RoboticsDomain

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60
out = Path("results")

runs = [(RoboticsDomain(), EvolutionaryProposer(seed=1)),
        (MedicalDomain(), EvolutionaryProposer(seed=2))]
summary, fig_data = {}, {}
for dom, prop in runs:
    r = run(dom, prop, budget, out, time_limit=240)
    summary[dom.name] = {
        "experiments": r.n_experiments, "improvements_kept": r.n_kept, "seconds": round(r.seconds, 1),
        "baseline_val": r.baseline_val, "best_val": r.best_val,
        "baseline_test": r.baseline_test, "best_test": r.best_test, "best_config": r.best_config}
    fig_data[dom.name] = r.history
(out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, hist) in zip(axes, fig_data.items()):
    sc = [e.val["score"] for e in hist]
    best, run_best = [], -1e18
    for s in sc:
        run_best = max(run_best, s)
        best.append(run_best)
    ax.scatter(range(len(sc)), sc, s=12, alpha=0.4, label="experiment")
    ax.plot(best, lw=2, label="best so far")
    lo = min(best[0], sorted(sc)[len(sc) // 5])
    ax.set_ylim(lo - 0.02 * abs(lo) - 1e-3, max(best) + 0.05 * abs(max(best)) + 1e-3)
    ax.set_title(name.replace("_", " "))
    ax.set_xlabel("experiment #")
    ax.set_ylabel("validation score (higher = better)")
    ax.legend()
plt.tight_layout()
plt.savefig(out / "progress.png", dpi=140)
print(json.dumps(summary, indent=2, default=float))
