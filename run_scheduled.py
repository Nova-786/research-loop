"""Scheduled, resumable research run (designed for GitHub Actions).

Each invocation: load state -> run a bounded batch of experiments -> save state.
When a stop condition fires (budget cap or no-improvement patience) the hidden test split
is evaluated once and the domain is marked finished.

Usage: python run_scheduled.py [--per-run 50] [--time-limit 600] [--max-total 300] [--patience 60]
                               [--domains robotics,medical]

Env (all optional):
  USE_LLM=1            use the LLM proposer (needs ANTHROPIC_API_KEY); default is the free evolutionary one
  LLM_MAX_CALLS=100    hard cap on TOTAL LLM calls across all runs (then falls back to evolutionary)
  LLM_MODEL            model id (default claude-sonnet-5-5)
"""
import argparse
import json
import os
import traceback
import urllib.request
from pathlib import Path

from medical_domain import MedicalDomain
from research_loop import EvolutionaryProposer, LLMProposer, run_resumable
from robotics_domain import RoboticsDomain

OUT = Path("results")


def build_domains() -> dict:
    # FRESH test sets: robotics seed 456 and medical split seed 0 were already seen in earlier runs,
    # so the cloud runs use new ones (robotics test_seed=789, medical split seed=1).
    robotics = RoboticsDomain(robust_val=True, test_seed=789)     # domain-randomised validation
    robotics.name = "robotics_arm_reach_cloud"
    medical = MedicalDomain(seed=1)
    return {"robotics": robotics, "medical": medical}


def make_ask(state: dict, cap: int, model: str):
    key = os.environ["ANTHROPIC_API_KEY"]

    def ask(prompt: str) -> str:
        if state["llm_calls"] >= cap:
            raise RuntimeError("LLM call cap reached")       # LLMProposer falls back to evolutionary
        state["llm_calls"] += 1                               # counted before the call: failures cost too
        body = json.dumps({"model": model, "max_tokens": 600,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = json.load(r)["content"][0]["text"]
        a, b = text.find("{"), text.rfind("}")
        return text[a:b + 1]

    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run", type=int, default=50, help="max experiments per domain per invocation")
    ap.add_argument("--time-limit", type=float, default=600, help="seconds per domain per invocation")
    ap.add_argument("--max-total", type=int, default=300, help="total experiment cap per domain (budget)")
    ap.add_argument("--patience", type=int, default=60, help="stop after this many experiments without improvement")
    ap.add_argument("--domains", default="robotics,medical")
    args = ap.parse_args()

    use_llm = os.environ.get("USE_LLM") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))
    llm_cap = int(os.environ.get("LLM_MAX_CALLS") or 100)
    model = os.environ.get("LLM_MODEL") or "claude-sonnet-5-5"
    print(f"proposer: {'LLM (capped at %d calls)' % llm_cap if use_llm else 'evolutionary (free)'}")

    domains = build_domains()
    statuses = []
    for key in [d.strip() for d in args.domains.split(",") if d.strip()]:
        dom = domains[key]

        def make_proposer(n_done, state, _seed={"robotics": 11, "medical": 22}[key]):
            evo = EvolutionaryProposer(seed=_seed * 100003 + n_done)   # new seed per resume, still reproducible
            if use_llm:
                return LLMProposer(ask=make_ask(state, llm_cap, model), fallback=evo)
            return evo

        try:
            st = run_resumable(dom, make_proposer, OUT / f"state_{dom.name}.json", OUT,
                               per_run_budget=args.per_run, time_limit=args.time_limit,
                               max_total=args.max_total, patience=args.patience)
            statuses.append(st["status"])
        except Exception:
            traceback.print_exc()           # one broken domain must not block the other or the commit step
            statuses.append("error")

    all_finished = bool(statuses) and all(s == "finished" for s in statuses)
    print(f"statuses: {statuses}  all_finished={all_finished}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"all_finished={'true' if all_finished else 'false'}\n")


if __name__ == "__main__":
    main()
