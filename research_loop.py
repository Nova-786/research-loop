"""Domain-agnostic autonomous research loop.

The loop repeats: PROPOSE a config -> RUN an experiment on the validation split
-> RECORD the result -> KEEP if better, else revert. The hidden TEST split is
evaluated exactly once, at the end, and its score is never shown to the proposer.

Two entry points:
  * run()            one-shot loop (used by run_all.py / run_robust.py)
  * run_resumable()  state is loaded/saved between invocations, so a scheduled job
                     (GitHub Actions) can continue the search where the last run stopped.
                     The test split is evaluated ONCE, when a stop condition fires.

To plug in an LLM, subclass `Proposer` (or pass `LLMProposer(ask=your_fn)`).
The default `EvolutionaryProposer` needs no API and makes the loop runnable offline.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

Config = dict[str, Any]


# --------------------------------------------------------------------------- #
# Domain interface
# --------------------------------------------------------------------------- #
class Domain:
    """A research problem. Subclasses define a search space and an evaluator."""

    name = "domain"
    # param -> ("log", lo, hi) | ("lin", lo, hi) | ("int", lo, hi) | ("cat", [options])
    space: dict[str, tuple] = {}

    def baseline(self) -> Config:
        raise NotImplementedError

    def evaluate(self, config: Config, split: str) -> dict[str, float]:
        """Return metrics incl. 'score' (higher is better). split in {'val','test'}."""
        raise NotImplementedError

    def valid(self, config: Config) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Proposers
# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    idx: int
    config: Config
    rationale: str
    val: dict[str, float]
    kept: bool
    seconds: float


class Proposer:
    def propose(self, domain: Domain, history: list[Experiment], best: Experiment) -> tuple[Config, str]:
        raise NotImplementedError


def _sample(spec: tuple, rng: random.Random):
    kind = spec[0]
    if kind == "log":
        return math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2])))
    if kind == "lin":
        return rng.uniform(spec[1], spec[2])
    if kind == "int":
        return rng.randint(spec[1], spec[2])
    if kind == "cat":
        return rng.choice(spec[1])
    raise ValueError(kind)


def _perturb(value, spec: tuple, rng: random.Random, scale: float):
    kind = spec[0]
    if kind == "log":
        v = math.exp(math.log(value) + rng.gauss(0, scale * (math.log(spec[2]) - math.log(spec[1]))))
        return min(max(v, spec[1]), spec[2])
    if kind == "lin":
        v = value + rng.gauss(0, scale * (spec[2] - spec[1]))
        return min(max(v, spec[1]), spec[2])
    if kind == "int":
        v = round(value + rng.gauss(0, max(1.0, scale * (spec[2] - spec[1]))))
        return int(min(max(v, spec[1]), spec[2]))
    if kind == "cat":
        return rng.choice(spec[1])
    raise ValueError(kind)


class EvolutionaryProposer(Proposer):
    """(1+1)-style search: mutate the incumbent with a shrinking step, plus random restarts."""

    def __init__(self, seed: int = 0, restart_prob: float = 0.15):
        self.rng = random.Random(seed)
        self.restart_prob = restart_prob

    def propose(self, domain, history, best):
        rng = self.rng
        space = domain.space
        n = len(history)
        for _ in range(50):
            if rng.random() < self.restart_prob:
                cfg = {k: _sample(s, rng) for k, s in space.items()}
                why = "random restart: explore a new region of the search space"
            else:
                scale = max(0.03, 0.25 * (1 - n / 200))
                keys = rng.sample(list(space), k=min(len(space), rng.choice([1, 1, 2, 3])))
                cfg = dict(best.config)
                for k in keys:
                    cfg[k] = _perturb(cfg[k], space[k], rng, scale)
                why = f"local mutation of incumbent #{best.idx}: tweak {', '.join(keys)} (step={scale:.2f})"
            if domain.valid(cfg):
                return cfg, why
        return dict(best.config), "fallback: no valid mutation found"


class LLMProposer(Proposer):
    """Hook for an LLM. `ask(prompt) -> str` must return a JSON object:
    {"config": {...}, "rationale": "..."}. The prompt contains the search space,
    the incumbent, and the recent experiment log, but never the test split."""

    def __init__(self, ask: Callable[[str], str], fallback: Proposer | None = None, recent: int = 15):
        self.ask, self.fallback, self.recent = ask, fallback or EvolutionaryProposer(), recent

    def propose(self, domain, history, best):
        lines = [
            f"#{e.idx} score={e.val['score']:.4f} kept={e.kept} cfg={json.dumps(e.config, default=float)} why={e.rationale}"
            for e in history[-self.recent:]
        ]
        prompt = (
            f"You are an autonomous researcher on: {domain.name}.\n"
            f"Search space: {json.dumps(domain.space)}\n"
            f"Incumbent: {json.dumps(best.config, default=float)} (score {best.val['score']:.4f})\n"
            "Recent experiments:\n" + "\n".join(lines) +
            "\nPropose ONE new config as JSON {\"config\": {...}, \"rationale\": \"...\"}. "
            "State a falsifiable hypothesis in the rationale."
        )
        try:
            out = json.loads(self.ask(prompt))
            cfg = {**best.config, **out["config"]}
            if domain.valid(cfg):
                return cfg, "LLM: " + str(out.get("rationale", ""))[:300]
        except Exception as exc:  # malformed output -> never crash the loop
            return (*self.fallback.propose(domain, history, best)[:1], f"LLM failed ({type(exc).__name__}); fallback")
        return self.fallback.propose(domain, history, best)


# --------------------------------------------------------------------------- #
# The one-shot loop (unchanged behaviour)
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    domain: str
    baseline_val: dict
    baseline_test: dict
    best_val: dict
    best_test: dict
    best_config: Config
    n_experiments: int
    n_kept: int
    seconds: float
    history: list[Experiment] = field(default_factory=list)


def run(domain: Domain, proposer: Proposer, budget: int, out_dir: Path, time_limit: float | None = None,
        min_gain: float = 0.0) -> Result:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{domain.name}_log.jsonl"
    log = log_path.open("w")
    t_start = time.time()

    base_cfg = domain.baseline()
    base_val = domain.evaluate(base_cfg, "val")
    best = Experiment(0, base_cfg, "baseline", base_val, True, 0.0)
    history = [best]
    log.write(json.dumps({"idx": 0, "config": base_cfg, "val": base_val, "kept": True, "why": "baseline"}, default=float) + "\n")
    print(f"[{domain.name}] baseline val score = {base_val['score']:.4f}")

    for i in range(1, budget + 1):
        if time_limit and time.time() - t_start > time_limit:
            print(f"[{domain.name}] time limit reached after {i-1} experiments")
            break
        cfg, why = proposer.propose(domain, history, best)
        t0 = time.time()
        try:
            val = domain.evaluate(cfg, "val")
        except Exception as exc:  # a crashed experiment is a failed experiment, not a crashed loop
            val = {"score": -1e9, "error": f"{type(exc).__name__}: {exc}"}
        if not math.isfinite(val["score"]):
            val["score"] = -1e9
        kept = val["score"] > best.val["score"] + min_gain
        exp = Experiment(i, cfg, why, val, kept, time.time() - t0)
        history.append(exp)
        log.write(json.dumps({"idx": i, "config": cfg, "val": val, "kept": kept, "why": why,
                              "seconds": round(exp.seconds, 3)}, default=float) + "\n")
        log.flush()
        if kept:
            best = exp
            print(f"[{domain.name}] #{i:3d} new best val = {val['score']:.4f}  ({why})")

    # Hidden test split: touched once, after the search is over.
    base_test = domain.evaluate(base_cfg, "test")
    best_test = domain.evaluate(best.config, "test")
    log.write(json.dumps({"final": True, "baseline_test": base_test, "best_test": best_test,
                          "best_config": best.config}, default=float) + "\n")
    log.close()
    return Result(domain.name, base_val, base_test, best.val, best_test, best.config,
                  len(history) - 1, sum(e.kept for e in history) - 1, time.time() - t_start, history)


# --------------------------------------------------------------------------- #
# Resumable loop (for scheduled cloud runs)
# --------------------------------------------------------------------------- #
def _exp_to_dict(e: Experiment) -> dict:
    return {"idx": e.idx, "config": e.config, "rationale": e.rationale, "val": e.val,
            "kept": e.kept, "seconds": round(e.seconds, 3)}


def _exp_from_dict(d: dict) -> Experiment:
    return Experiment(d["idx"], d["config"], d.get("rationale", ""), d["val"], d["kept"], d.get("seconds", 0.0))


def load_state(path: Path, domain_name: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"domain": domain_name, "status": "new", "stop_reason": None, "runs": 0, "history": [],
            "best_idx": 0, "llm_calls": 0, "total_seconds": 0.0, "baseline_test": None,
            "best_test": None, "best_config": None, "updated": None}


def save_state(path: Path, state: dict) -> None:
    state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=float))
    tmp.replace(path)          # atomic: a killed job never leaves a half-written state file


def stop_reason(state: dict, max_total: int, patience: int) -> str | None:
    """Stop conditions: total experiment cap (budget) and no-improvement patience."""
    n = len(state["history"]) - 1
    if n >= max_total:
        return f"budget cap reached ({max_total} experiments)"
    if patience and n - state["best_idx"] >= patience:
        return f"converged: no improvement in {patience} experiments"
    return None


def run_resumable(domain: Domain, make_proposer: Callable[[int, dict], Proposer], state_path: Path,
                  out_dir: Path, per_run_budget: int = 50, time_limit: float | None = None,
                  max_total: int = 300, patience: int = 60, min_gain: float = 0.0) -> dict:
    """Continue the search from `state_path`, run up to `per_run_budget` experiments, save state.

    The hidden TEST split is evaluated once, only when a stop condition fires; after that the
    state is marked 'finished' and later invocations do nothing (so the test set stays unseen
    by the search, and a result is never re-scored)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path, domain.name)
    if state["status"] == "finished":
        print(f"[{domain.name}] already finished ({state['stop_reason']}); nothing to do")
        return state

    t_start = time.time()
    log = (out_dir / f"{domain.name}_log.jsonl").open("a")

    if not state["history"]:
        base_cfg = domain.baseline()
        base_val = domain.evaluate(base_cfg, "val")
        base = Experiment(0, base_cfg, "baseline", base_val, True, 0.0)
        state["history"].append(_exp_to_dict(base))
        state["best_idx"] = 0
        log.write(json.dumps({"idx": 0, "config": base_cfg, "val": base_val, "kept": True, "why": "baseline"},
                             default=float) + "\n")
        print(f"[{domain.name}] baseline val score = {base_val['score']:.4f}")

    history = [_exp_from_dict(d) for d in state["history"]]
    best = history[state["best_idx"]]
    proposer = make_proposer(len(history), state)
    print(f"[{domain.name}] resuming at experiment #{len(history)} (best val {best.val['score']:.4f})")

    done_this_run = 0
    while done_this_run < per_run_budget:
        if stop_reason(state, max_total, patience):
            break
        if time_limit and time.time() - t_start > time_limit:
            print(f"[{domain.name}] per-run time limit reached")
            break
        i = len(history)
        cfg, why = proposer.propose(domain, history, best)
        t0 = time.time()
        try:
            val = domain.evaluate(cfg, "val")
        except Exception as exc:
            val = {"score": -1e9, "error": f"{type(exc).__name__}: {exc}"}
        if not math.isfinite(val["score"]):
            val["score"] = -1e9
        kept = val["score"] > best.val["score"] + min_gain
        exp = Experiment(i, cfg, why, val, kept, time.time() - t0)
        history.append(exp)
        state["history"].append(_exp_to_dict(exp))
        log.write(json.dumps({"idx": i, "config": cfg, "val": val, "kept": kept, "why": why,
                              "seconds": round(exp.seconds, 3)}, default=float) + "\n")
        log.flush()
        if kept:
            best = exp
            state["best_idx"] = i
            print(f"[{domain.name}] #{i:3d} new best val = {val['score']:.4f}  ({why})")
        done_this_run += 1
        if done_this_run % 10 == 0:
            save_state(state_path, state)       # checkpoint

    reason = stop_reason(state, max_total, patience)
    if reason:
        # Hidden test split: touched once, now that the search is over.
        base_cfg = history[0].config
        state["baseline_test"] = domain.evaluate(base_cfg, "test")
        state["best_test"] = domain.evaluate(best.config, "test")
        state["best_config"] = best.config
        state["status"] = "finished"
        state["stop_reason"] = reason
        log.write(json.dumps({"final": True, "baseline_test": state["baseline_test"],
                              "best_test": state["best_test"], "best_config": best.config,
                              "stop_reason": reason}, default=float) + "\n")
        print(f"[{domain.name}] FINISHED: {reason}")
    else:
        state["status"] = "running"
        state["best_config"] = best.config
    log.close()
    state["runs"] += 1
    state["total_seconds"] = round(state["total_seconds"] + time.time() - t_start, 1)
    save_state(state_path, state)
    return state
