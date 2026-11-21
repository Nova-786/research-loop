"""Build docs/index.html (the GitHub Pages progress page) from results/state_*.json.

Pure standard library: charts are inline SVG, so there is nothing to install or load.
"""
import html
import json
import shutil
from pathlib import Path

RESULTS, DOCS = Path("results"), Path("docs")


def chart(scores, w=520, h=240):
    best, m = [], -1e18
    for s in scores:
        m = max(m, s)
        best.append(m)
    srt = sorted(scores)
    lo = min(best[0], srt[len(srt) // 5])          # clip awful outliers so progress stays visible
    hi = max(best)
    pad = (hi - lo) * 0.08 or 1e-3
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 56, 10, 10, 26
    px = lambda i: L + (w - L - R) * i / max(1, len(scores) - 1)
    py = lambda v: T + (h - T - B) * (1 - (v - lo) / (hi - lo))
    dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(s):.1f}" r="2.5" class="dot"/>'
                   for i, s in enumerate(scores) if lo <= s <= hi)
    line = " ".join(f"{px(i):.1f},{py(b):.1f}" for i, b in enumerate(best))
    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += (f'<line x1="{L}" x2="{w - R}" y1="{py(v):.1f}" y2="{py(v):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="tick">{v:.3f}</text>')
    xl = f'<text x="{(L + w - R) / 2:.0f}" y="{h - 6}" text-anchor="middle" class="tick">experiment #</text>'
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="validation score by experiment">'
            f'{grid}{dots}<polyline points="{line}" class="best"/>{xl}</svg>')


def fmt(d):
    if not d:
        return "&mdash;"
    s = f"{d['score']:.4f}"
    if "ci95_low" in d:
        s += f" <small>[{d['ci95_low']:.3f}, {d['ci95_high']:.3f}]</small>"
    return s


def card(st):
    hist = st["history"]
    scores = [e["val"]["score"] for e in hist]
    n = len(hist) - 1
    kept = sum(e["kept"] for e in hist) - 1
    base, best = hist[0]["val"]["score"], hist[st["best_idx"]]["val"]["score"]
    fin = st["status"] == "finished"
    badge = f'<span class="badge {"done" if fin else "run"}">{"finished" if fin else "running"}</span>'
    test = (f"<tr><th>Hidden test (once)</th><td>{fmt(st['baseline_test'])} &rarr; {fmt(st['best_test'])}</td></tr>"
            if fin else "<tr><th>Hidden test</th><td>sealed until the search stops</td></tr>")
    stop = f"<tr><th>Stopped because</th><td>{html.escape(st['stop_reason'])}</td></tr>" if fin else ""
    cfg = html.escape(json.dumps(st.get("best_config") or hist[st["best_idx"]]["config"], default=float))
    return f"""<section><h2>{html.escape(st['domain'].replace('_', ' '))} {badge}</h2>
{chart(scores)}
<table>
<tr><th>Experiments</th><td>{n} ({kept} improvements kept) over {st['runs']} runs</td></tr>
<tr><th>Validation score</th><td>{base:.4f} &rarr; {best:.4f}</td></tr>
{test}{stop}
<tr><th>LLM calls used</th><td>{st.get('llm_calls', 0)}</td></tr>
<tr><th>Last update (UTC)</th><td>{html.escape(str(st['updated']))}</td></tr>
</table>
<details><summary>Best config</summary><pre>{cfg}</pre></details></section>"""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Autonomous research loop</title>
<style>
:root{{--bg:#fafafa;--fg:#1a1a1a;--mut:#666;--card:#fff;--line:#ddd;--acc:#1f77b4;--ok:#2e7d32}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#eee;--mut:#999;--card:#1c1c1c;--line:#333;--acc:#6aaee0;--ok:#66bb6a}}}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto}} h1{{margin:0 0 4px}} p.sub{{color:var(--mut);margin-top:0}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}}
h2{{font-size:17px;margin:0 0 8px;text-transform:capitalize}}
.badge{{font-size:12px;border-radius:99px;padding:2px 9px;border:1px solid var(--line);text-transform:none}}
.badge.done{{color:var(--ok)}} .badge.run{{color:var(--acc)}}
svg{{width:100%;height:auto}} .dot{{fill:var(--acc);opacity:.4}} .best{{fill:none;stroke:var(--acc);stroke-width:2.5}}
.grid{{stroke:var(--line)}} .tick{{fill:var(--mut);font-size:11px}}
table{{border-collapse:collapse;width:100%;margin-top:8px}} th{{text-align:left;color:var(--mut);font-weight:500;padding:3px 10px 3px 0;white-space:nowrap}}
td{{padding:3px 0}} small{{color:var(--mut)}} pre{{white-space:pre-wrap;word-break:break-all;font-size:12px}}
footer{{margin-top:22px;color:var(--mut);font-size:13px}}
</style></head><body><main>
<h1>Autonomous research loop</h1>
<p class="sub">Scheduled GitHub Actions runs. Dots = experiments, line = best validation score so far (higher is better).</p>
<div class="grid2">{cards}</div>
<footer>The hidden test split is evaluated once, when a domain stops, and is never shown to the proposer.
The medical domain is a research prototype only, not clinical advice. The robotics domain is a 2D toy simulator.</footer>
</main></body></html>"""


def main():
    states = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("state_*.json"))]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data").mkdir(exist_ok=True)
    for p in RESULTS.glob("state_*.json"):
        shutil.copy(p, DOCS / "data" / p.name)
    cards = "\n".join(card(s) for s in states if s["history"]) or "<p>No results yet. The first scheduled run will fill this in.</p>"
    (DOCS / "index.html").write_text(PAGE.format(cards=cards))
    print(f"wrote {DOCS / 'index.html'} ({len(states)} domains)")


if __name__ == "__main__":
    main()
