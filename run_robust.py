"""Second robotics run: domain-randomised validation, scored on a FRESH test set (seed 456)."""
import json
from pathlib import Path
from research_loop import EvolutionaryProposer, run
from robotics_domain import RoboticsDomain

d = RoboticsDomain(robust_val=True, test_seed=456)
d.name = "robotics_arm_reach_robust"
r = run(d, EvolutionaryProposer(seed=1), 60, Path("results"))
out = {"baseline_val": r.baseline_val, "best_val": r.best_val, "baseline_test": r.baseline_test,
       "best_test": r.best_test, "best_config": r.best_config, "improvements_kept": r.n_kept}
Path("results/summary_robust.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))
