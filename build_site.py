"""Build the readable progress page (docs/index.html) and results/SUMMARY.md
from results/state_*.json. Pure standard library, no installs.

Everything is written in plain English: a one-sentence verdict per task, before/after
numbers, and the best settings as a simple list. Charts are inline SVG.
"""
import html
import json
import shutil
from pathlib import Path

RESULTS, DOCS = Path("results"), Path("docs")

NAMES = {
    "robotics": ("Robot arm controller",
                 "A simulated robot arm learns to reach targets accurately. "
                 "Cost is the error, so lower is better."),
    "medical": ("Cancer-screening classifier (prototype)",
                "A model that tells benign from malignant cases on a public dataset. "
                "Research prototype only, not medical advice."),
    "exosuit": ("Exosuit assist controller",
                "A simulated powered suit learns to help its wearer move. Lower cost is better."),
}


def describe(domain):
    for key, (title, desc) in NAMES.items():
        if key in domain:
            return title, desc
    return domain.replace("_", " ").title(), ""


def is_cost(st):
    """Scores are 'higher is better'. Negative scores are costs (cost = -score)."""
    return st["history"][0]["val"]["score"] < 0


def shown(x, cost):
    return -x if cost else x


def pct(base, best):
    return (best - base) / abs(base) * 100 if base else 0.0


def verdict(st):
    """Return (kind, plain-English sentence). kind: good | flat | bad | run."""
    hist, cost = st["history"], is_cost(st)
    word = "cost" if cost else "score"
    vb, vbest = hist[0]["val"]["score"], hist[st["best_idx"]]["val"]["score"]
    vp = pct(vb, vbest)
    if st["status"] != "finished":
        return "run", (f"Still searching. On practice data the {word} has improved {vp:.0f}% so far. "
                       "The final exam (the hidden test) stays sealed until the search stops.")
    bt, et = st["baseline_test"], st["best_test"]
    if "ci95_low" in bt and "ci95_low" in et:
        if et["ci95_low"] <= bt["ci95_high"] and bt["ci95_low"] <= et["ci95_high"]:
            return "flat", ("No real improvement. The score was already near the top, and the "
                            "before and after results are too close to tell apart. "
                            "The tiny difference is noise.")
    tp = pct(bt["score"], et["score"])
    if tp > 5:
        txt = f"Real improvement. On unseen test data the {word} {'fell' if cost else 'rose'} {tp:.0f}%."
        if vp > 0 and tp < 0.7 * vp:
            txt += (f" That is less than the {vp:.0f}% gain on practice data, so part of what it "
                    "learned was specific to the practice conditions.")
        return "good", txt
    if tp < -5:
        return "bad", ("Did not carry over. It looked better on practice data but got worse "
                       "on unseen test data.")
    return "flat", "About the same on unseen test data. No meaningful improvement."


def num(x, cost):
    return f"{shown(x, cost):.3f}"


def settings(st):
    cfg = st.get("best_config") or st["history"][st["best_idx"]]["config"]
    out = []
    for k, v in cfg.items():
        out.append((k, f"{v:.3g}" if isinstance(v, float) else str(v)))
    return out


def chart(st, w=520, h=250):
    cost = is_cost(st)
    scores = [e["val"]["score"] for e in st["history"]]      # higher = better, always
    best, m = [], -1e18
    for s in scores:
        m = max(m, s)
        best.append(m)
    srt = sorted(scores)
    lo, hi = min(best[0], srt[len(srt) // 5]), max(best)     # clip awful outliers
    pad = (hi - lo) * 0.08 or 1e-3
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 58, 10, 10, 30
    px = lambda i: L + (w - L - R) * i / max(1, len(scores) - 1)
    py = lambda v: T + (h - T - B) * (1 - (v - lo) / (hi - lo))
    dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(s):.1f}" r="3" class="dot"/>'
                   for i, s in enumerate(scores) if lo <= s <= hi)
    line = " ".join(f"{px(i):.1f},{py(b):.1f}" for i, b in enumerate(best))
    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += (f'<line x1="{L}" x2="{w - R}" y1="{py(v):.1f}" y2="{py(v):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{py(v) + 5:.1f}" text-anchor="end" class="tick">'
                 f'{shown(v, cost):.3f}</text>')
    xl = (f'<text x="{(L + w - R) / 2:.0f}" y="{h - 7}" text-anchor="middle" class="tick">'
          f'experiment number</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="results of each experiment, higher on the chart is better">'
            f'{grid}{dots}<polyline points="{line}" class="best"/>{xl}</svg>')


def card(st):
    title, desc = describe(st["domain"])
    cost = is_cost(st)
    word = "Cost" if cost else "Score"
    hist = st["history"]
    n = len(hist) - 1
    kept = sum(e["kept"] for e in hist) - 1
    kind, text = verdict(st)
    fin = st["status"] == "finished"
    vb, vbest = hist[0]["val"]["score"], hist[st["best_idx"]]["val"]["score"]
    rows = [f"<tr><th>Experiments tried</th><td>{n} ({kept} were improvements)</td></tr>",
            f"<tr><th>{word} on practice data</th><td>{num(vb, cost)} &rarr; <b>{num(vbest, cost)}</b></td></tr>"]
    if fin:
        bt, et = st["baseline_test"], st["best_test"]
        extra = " (the ranges overlap)" if ("ci95_low" in bt and kind == "flat") else ""
        rows.append(f"<tr><th>{word} on unseen test data</th><td>{num(bt['score'], cost)} &rarr; "
                    f"<b>{num(et['score'], cost)}</b>{extra}</td></tr>")
        rows.append(f"<tr><th>Why it stopped</th><td>{html.escape(st['stop_reason'])}</td></tr>")
    else:
        rows.append("<tr><th>Unseen test data</th><td>sealed until the search stops</td></tr>")
    rows.append(f"<tr><th>Last updated (UTC)</th><td>{html.escape(str(st['updated']))}</td></tr>")
    sett = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in settings(st))
    badge = "finished" if fin else "running"
    return f"""<section class="{kind}">
<h2>{html.escape(title)} <span class="badge {badge}">{badge}</span></h2>
<p class="desc">{html.escape(desc)}</p>
<p class="verdict">{html.escape(text)}</p>
{chart(st)}
<p class="cap">Each dot is one experiment. The line is the best result so far. Higher on the chart is better.</p>
<table>{''.join(rows)}</table>
<details><summary>Best settings found</summary><ul>{sett}</ul></details></section>"""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Autonomous research loop</title>
<style>
:root{{--bg:#f6f6f6;--fg:#111;--mut:#444;--card:#fff;--line:#ccc;--acc:#0b5cad;--good:#1b7a2f;--flat:#8a6100;--bad:#b3261e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#101010;--fg:#f4f4f4;--mut:#bdbdbd;--card:#1b1b1b;--line:#3a3a3a;--acc:#7fb8f0;--good:#6fd184;--flat:#e8b84a;--bad:#ff8a80}}}}
body{{margin:0;padding:20px;background:var(--bg);color:var(--fg);font:18px/1.6 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto}} h1{{font-size:28px;margin:0 0 6px}} .sub{{color:var(--mut);margin:0 0 18px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}}
section{{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--acc);border-radius:10px;padding:18px}}
section.good{{border-left-color:var(--good)}} section.flat{{border-left-color:var(--flat)}} section.bad{{border-left-color:var(--bad)}}
h2{{font-size:21px;margin:0 0 4px}} .desc{{color:var(--mut);margin:0 0 10px;font-size:16px}}
.verdict{{font-size:19px;font-weight:600;margin:0 0 12px}}
.badge{{font-size:13px;border-radius:99px;padding:2px 10px;border:1px solid var(--line);font-weight:400;vertical-align:middle}}
svg{{width:100%;height:auto}} .dot{{fill:var(--acc);opacity:.55}} .best{{fill:none;stroke:var(--acc);stroke-width:3}}
.grid{{stroke:var(--line)}} .tick{{fill:var(--mut);font-size:13px}} .cap{{color:var(--mut);font-size:14px;margin:2px 0 8px}}
table{{border-collapse:collapse;width:100%}} th{{text-align:left;color:var(--mut);font-weight:500;padding:5px 12px 5px 0;vertical-align:top;font-size:16px}}
td{{padding:5px 0;font-size:16px}} details{{margin-top:10px;font-size:16px}} summary{{cursor:pointer;color:var(--acc)}}
ul{{margin:6px 0 0 20px;padding:0}} footer{{margin-top:24px;color:var(--mut);font-size:15px}}
</style></head><body><main>
<h1>Autonomous research loop</h1>
<p class="sub">Results so far, in plain English. Page rebuilt: {built}</p>
<div class="grid2">{cards}</div>
<footer><p><b>How to read this:</b> practice data is what the search can see while it works. Unseen test data is a final exam
checked once, when a search stops, so it shows whether the improvement is real.</p>
<p>The cancer-screening task is a research prototype only, not medical advice. The robot arm is a 2D toy simulation.</p></footer>
</main></body></html>"""


def summary_md(states, built):
    out = ["# Research loop: plain-English summary", "", f"Rebuilt: {built}", ""]
    for st in states:
        title, desc = describe(st["domain"])
        cost = is_cost(st)
        word = "Cost" if cost else "Score"
        hist = st["history"]
        kind, text = verdict(st)
        vb, vbest = hist[0]["val"]["score"], hist[st["best_idx"]]["val"]["score"]
        out += [f"## {title} ({st['status']})", "", f"*{desc}*", "", f"**{text}**", "",
                f"- Experiments tried: {len(hist) - 1}",
                f"- {word} on practice data: {num(vb, cost)} -> {num(vbest, cost)}"]
        if st["status"] == "finished":
            bt, et = st["baseline_test"], st["best_test"]
            out.append(f"- {word} on unseen test data: {num(bt['score'], cost)} -> {num(et['score'], cost)}")
            out.append(f"- Why it stopped: {st['stop_reason']}")
        out += ["- Best settings:"] + [f"  - {k}: {v}" for k, v in settings(st)] + [""]
    return "\n".join(out)


def main():
    from datetime import datetime, timezone
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    states = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("state_*.json"))]
    states = [s for s in states if s["history"]]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data").mkdir(exist_ok=True)
    for p in RESULTS.glob("state_*.json"):
        shutil.copy(p, DOCS / "data" / p.name)
    cards = "\n".join(card(s) for s in states) or "<p>No results yet. The first scheduled run will fill this in.</p>"
    (DOCS / "index.html").write_text(PAGE.format(cards=cards, built=built))
    if states:
        (RESULTS / "SUMMARY.md").write_text(summary_md(states, built))
    print(f"wrote {DOCS / 'index.html'} and results/SUMMARY.md ({len(states)} tasks)")


if __name__ == "__main__":
    main()
