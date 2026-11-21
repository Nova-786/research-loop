# Handoff note

Paste this into a new chat in the project that holds these files.

This project holds an autonomous research loop (`research_loop.py`) with two domains:
a simulated 2-link robot arm controller (`robotics_domain.py`) and a medical diagnostic
classifier on the public breast-cancer dataset (`medical_domain.py`). See `README.md` for results.

## Key findings so far
- Plain validation overfit the robotics sim: validation improved, the hidden test got worse.
- Domain-randomised validation fixed it: test cost 0.450 -> 0.158 on a fresh test set.
- The medical task is at ceiling (AUROC ~0.995), so it needs a harder dataset.

## Next goal
Run this free in the cloud while I'm offline, using GitHub Actions on a schedule.

Needed:
1. Make the loop resumable (load and save state between runs).
2. Add a scheduled workflow file (`.github/workflows/research.yml`).
3. Commit results back to the repo after each run.
4. Publish a progress page on GitHub Pages.
5. Add a stop condition and a budget cap.

Keep the free evolutionary proposer as the default. The LLM proposer is optional because API calls cost money.

## Rules to keep
- The hidden test split is evaluated once, at the end, and never shown to the proposer.
- If a test set has been seen, make a fresh one before claiming a result.
- Medical work is a research prototype only, never clinical advice.
