# Autonomous research loop: robotics + medical

A small, runnable harness where an agent proposes experiments, runs them, and keeps what works.

    python3 run_all.py 60      # both domains, 60 experiments each -> results/
    python3 run_robust.py      # robotics with domain-randomised validation, fresh test set

Needs only numpy, scipy, scikit-learn, matplotlib. No GPU, no network, no API key.

## Files
- `research_loop.py`   domain-agnostic loop: propose -> run -> record -> keep/revert. Hidden test split is evaluated once, at the end.
- `robotics_domain.py` simulated 2-link arm; tune a Cartesian PD controller (kp, kd, joint damping, torque limit, feedforward). Test split adds heavier mass error, sensor noise, actuation delay (the "reality gap").
- `medical_domain.py`  diagnostic classifier on the public Breast Cancer Wisconsin dataset. Preprocessing inside CV, locked 30% test set, bootstrap CI. Research prototype, not a clinical tool.
- `results/`           JSONL log of every experiment, summary JSON, progress plot.

## Results (what actually happened)
| Run | Baseline test | Best test | Transferred? |
|---|---|---|---|
| Robotics, plain validation | cost 0.400 | cost 0.463 | No: overfit the clean sim |
| Robotics, domain-randomised validation (fresh test set) | cost 0.450 | cost 0.158 | Yes |
| Medical (AUROC) | 0.9956 [0.990, 1.000] | 0.9947 [0.987, 1.000] | No real change; task is at ceiling |

Note: the robust robotics run was designed after seeing the first test result, so it uses a new
test set (seed 456) to keep the check honest. Treat it as one demonstration, not a benchmark.

## Plugging in an LLM
`LLMProposer(ask=fn)` takes any function `prompt -> JSON string`. The prompt carries the search
space, the incumbent and the recent log, never the test split. Malformed output falls back to the
evolutionary proposer, so the loop never crashes. With an LLM the search space can grow from
"tune numbers" to "edit code", which is where it becomes real research, and also where
guardrails matter most.

## Known limits
- The default proposer is a simple evolutionary search, not an LLM; it shows the loop, not intelligence.
- Robotics is a 2D toy simulator, not a physics engine. Real hardware needs resets, safety limits and a human.
- The medical dataset is small and easy. Real medical research needs ethics review, external validation and clinicians.
