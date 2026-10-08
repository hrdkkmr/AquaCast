"""
Presentation dashboard
======================

Stage view for the demo: title, architecture, impact, roadmap, the deck itself,
and a **Demo Mode** that jumps straight to any section so nobody scrolls while
the clock runs.

The three-minute script is in ``reports/pitch_script.md`` (412 words ≈ 2:45
spoken). The timing below is the same script, section by section.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui                                      # noqa: E402
from src.crop_library import CROPS                      # noqa: E402
from src.farm_twin import baseline, evaluate            # noqa: E402
from src.optimizer import optimize                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "reports" / "AquaCast-Punjab.pptx"
SCRIPT = ROOT / "reports" / "pitch_script.md"
FACTS = ROOT / "reports" / "deck_facts.json"

SECTIONS = [
    ("hero", "Title"),
    ("problem", "The problem"),
    ("product", "What AquaCast does"),
    ("architecture", "Architecture"),
    ("twin", "Climate Twin"),
    ("impact", "Impact"),
    ("moat", "Why us"),
    ("roadmap", "Roadmap"),
    ("deck", "Deck & video"),
]

#: The 3-minute script, split against the clock.
TIMINGS = [
    ("0:00", "0:20", "Hook — Punjab's stage of extraction is 164%; 111 of 153 "
                     "blocks over-exploited."),
    ("0:20", "0:50", "The gap — forecasts exist, decisions don't. Nobody tells "
                     "a farmer what to plant instead."),
    ("0:50", "1:25", "The product — Forecast → Simulate → Optimize → Finance → "
                     "Verify."),
    ("1:25", "2:00", "Live Climate Twin — change the crop mix, watch water, "
                     "income and carbon move together."),
    ("2:00", "2:30", "Finance — the optimiser finds the plan, the subsidy stack "
                     "makes it affordable."),
    ("2:30", "2:55", "Why us — one model serving farmer, lender and regulator."),
]

# ---------------------------------------------------------------------------
params = st.query_params
target = params.get("goto", "hero")

with st.sidebar:
    st.markdown(f'<div style="font-size:11px;font-weight:700;color:{ui.MUTED};'
                f'letter-spacing:.06em;text-transform:uppercase">Demo mode</div>',
                unsafe_allow_html=True)
    for key, label in SECTIONS:
        if st.button(("▶ " if target == key else "") + label, key=f"goto_{key}",
                     width='stretch',
                     type="primary" if target == key else "secondary"):
            st.query_params["goto"] = key
            st.rerun()
    st.markdown('<hr class="ac-rule">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:10.5px;color:{ui.MUTED};line-height:1.5">'
        f'<b>3-minute script</b><br>reports/pitch_script.md<br>412 words ≈ 2:45 '
        f'spoken.</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _slide_titles() -> list[str]:
    """Read the real deck — titles are extracted, never retyped."""
    if not DECK.exists():
        return []
    try:
        from pptx import Presentation
        prs = Presentation(str(DECK))
        out = []
        for i, s in enumerate(prs.slides, 1):
            title = ""
            for sh in s.shapes:
                if sh.has_text_frame and sh.text_frame.text.strip():
                    title = sh.text_frame.text.strip().split("\n")[0]
                    break
            out.append(f"{i:02d} · {title or '(untitled)'}")
        return out
    except Exception:
        return []


def anchor(key: str) -> None:
    st.markdown(f'<a name="{key}"></a>', unsafe_allow_html=True)


# ===========================================================================
anchor("hero")
st.markdown('<div class="ac-card" style="background:linear-gradient(120deg,'
            '#E6F2EE,#F6F8FB);border-color:#CFE3DA">'
            '<div style="font-size:34px;font-weight:800;letter-spacing:-.03em;'
            'line-height:1.1;color:#0F1B2D">AquaCast</div>'
            '<div style="font-size:17px;font-weight:600;color:#0B6E4F;'
            'margin-top:4px">AI-powered Climate-Smart Farm Digital Twin</div>'
            '<div style="font-size:13.5px;color:#5A6B82;margin-top:10px;'
            'line-height:1.6">Punjab grows food for a fifth of the world on an '
            'aquifer that is running out. AquaCast turns a groundwater forecast '
            'into a farm plan, a loan, and a verified outcome.'
            '</div>'
            '<div style="font-size:11px;letter-spacing:.06em;color:#0B6E4F;'
            'font-weight:700;margin-top:14px">'
            'FORECAST · SIMULATE · OPTIMIZE · FINANCE · VERIFY</div>'
            '<div style="font-size:11.5px;color:#5A6B82;margin-top:8px">'
            'SANKALP by Satin Finserv — Climate Edition</div></div>',
            unsafe_allow_html=True)

with st.spinner("Computing live figures…"):
    base = baseline("Sangrur", 2.0)
    cur = evaluate(base)
    r = optimize(base, "balanced")
    p = r.best_outcome

a1, a2, a3, a4 = st.columns(4)
with a1:
    ui.kpi("Stage of extraction", "164%",
           "Punjab — 2nd worst in India, 111/153 blocks over-exploited")
with a2:
    ui.kpi("Water saved", f"{cur.gw_draft_m3 - p.gw_draft_m3:,.0f} m³",
           f"per 2 ha holding · {r.headline()}")
with a3:
    ui.kpi("Income", f"{(p.income_inr - cur.income_inr) / cur.income_inr:+.0%}",
           "on the recommended plan")
with a4:
    ui.kpi("Plans evaluated", f"{r.n_evaluated:,}",
           "crop mix × irrigation × residue, per query")

ui.rule()

# ---------------------------------------------------------------------------
anchor("problem")
st.markdown("### The problem")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f'<div class="ac-card"><b>Punjab is running out of groundwater.</b>'
        f'<div class="ac-muted" style="margin-top:8px;line-height:1.6">'
        f'Stage of groundwater extraction reached <b>163.8%</b> — the state '
        f'draws far more than it recharges. Of 153 blocks, <b>111 are '
        f'over-exploited</b>, second only to Rajasthan; 57 of those are above '
        f'200%. The 2026 assessment shows improvement (152.2%) but the '
        f'structural gap remains.</div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ac-card"><b>Farmers are told what will happen, not what '
        f'to do.</b><div class="ac-muted" style="margin-top:8px;line-height:1.6">'
        f'Forecasts, dashboards and advisories all stop at the warning. The '
        f'farmer still has to decide what to plant, whether they can afford the '
        f'change, and whether it worked.</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(
        f'<div class="ac-card"><b>The decision is coupled, and no tool couples '
        f'it.</b><div class="ac-muted" style="margin-top:8px;line-height:1.6">'
        f'Take land out of paddy and you do not simply save water — you lose '
        f'assured procurement, you change the residue stream, you cut methane, '
        f'and you change exposure to a single buyer. Independent levers cannot '
        f'show that.</div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ac-card"><b>And the money is the barrier.</b>'
        f'<div class="ac-muted" style="margin-top:8px;line-height:1.6">'
        f'Drip irrigation is the single biggest lever and it costs '
        f'<b>₹78,000/ha</b>. A plan that pays back over five years is useless '
        f'to a farmer without five years of cash.</div></div>',
        unsafe_allow_html=True)

ui.rule()

# ---------------------------------------------------------------------------
anchor("product")
st.markdown("### What AquaCast does")
st.markdown(
    f'<div class="ac-muted">One chain, five steps. Each one hands the next a '
    f'computed artefact — not a chart.</div>', unsafe_allow_html=True)
steps = [
    ("🔮", "Forecast", "PyTorch LSTM on real India-WRIS telemetry + FAO-56 + a "
                       "lumped aquifer water balance. 0.0722 m validation RMSE, "
                       "beats seasonal-naive, linear-trend and persistence."),
    ("🔬", "Simulate", "The Climate Farm Twin recomputes water, income, carbon, "
                       "residue and risk for any crop mix, irrigation method, "
                       "residue pathway, canal supply and monsoon scenario."),
    ("⚖️", "Optimize", f"{r.n_evaluated:,} candidate plans scored against four "
                       f"objectives, with the Pareto frontier of income against "
                       f"water shown rather than one prescribed answer."),
    ("🏦", "Finance", "Real subsidy schemes stacked against the capital cost, "
                      "KCC-priced loan for the gap, payback on the farmer's own "
                      "cash — and the lender's own credit engine re-scored."),
    ("✅", "Verify", "Monte Carlo uncertainty on every claim, plus an explicit "
                     "statement of which MRV steps are performed and which are "
                     "not."),
]
for g, t, d in steps:
    st.markdown(
        f'<div class="ac-card" style="border-left:3px solid {ui.ACCENT}">'
        f'<div style="font-weight:700;font-size:14.5px">{g} {t}</div>'
        f'<div class="ac-muted" style="margin-top:5px;line-height:1.55">{d}</div>'
        f'</div>', unsafe_allow_html=True)

ui.rule()

# ---------------------------------------------------------------------------
anchor("architecture")
st.markdown("### Architecture")
st.markdown(
    f'<div class="ac-card" style="font-family:ui-monospace,SFMono-Regular,'
    f'Menlo,monospace;font-size:11.5px;line-height:1.75;white-space:pre;'
    f'overflow-x:auto;color:{ui.INK}">'
    f'DATA           India-WRIS gauges · IMD temps · CGWB wells · GLDAS/GRACE\n'
    f'               └─&gt; data/processed  (real, 2009-2024)\n'
    f'\n'
    f'FORECAST       FAO-56 dual-Kc water balance  ──┐\n'
    f'               lumped aquifer model          ──┼─&gt; PyTorch LSTM\n'
    f'               quantile-mapped bias correction──┘      (2-layer, MC-dropout)\n'
    f'                                                        val RMSE 0.0722 m\n'
    f'\n'
    f'DECISION       src/  crop_library → farm_twin → optimizer\n'
    f'                     carbon · climate_finance · impact · advisor\n'
    f'\n'
    f'INTERFACE      Streamlit · Farmer · Farmer Phone · Climate Ops\n'
    f'\n'
    f'MOCKED         PostgreSQL/PostGIS · Google Earth Engine · Twilio/\n'
    f'               WhatsApp · lender KYC &amp; bureau · IoT soil probes\n'
    f'               (swappable behind one interface each — see README)</div>',
    unsafe_allow_html=True)
st.caption("Runs with zero credentials. Every mocked box is documented in "
           "README §\"Prototype vs Production Architecture\".")

ui.rule()

# ---------------------------------------------------------------------------
anchor("twin")
st.markdown("### The Climate Twin, live")
st.markdown(
    f'<div class="ac-card"><div style="font-size:15px;font-weight:600">'
    f'Recommended for a 2 ha Sangrur holding</div>'
    f'<div class="ac-muted" style="margin-top:5px">'
    f'{", ".join(f"{CROPS[k].label} {v:.0%}" for k, v in r.best.allocation.items())}'
    f'</div><div style="margin-top:10px;font-size:14px"><b>{r.headline()}</b>'
    f'</div><div class="ac-muted" style="margin-top:5px">Best of '
    f'{r.n_evaluated:,} plans · {r.investment_label}'
    + (f" · payback {r.payback_years:.1f} yr"
       if r.payback_years == r.payback_years else "") +
    f'</div></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="ac-muted" style="margin-top:8px">Open the <b>Farmer → '
    f'Climate Twin</b> tab and move a slider during the demo — every figure '
    f'and both charts recompute together.</div>', unsafe_allow_html=True)

ui.rule()

# ---------------------------------------------------------------------------
anchor("impact")
st.markdown("### Impact")
i1, i2, i3 = st.columns(3)
with i1:
    ui.kpi("Water, per holding",
           f"{cur.gw_draft_m3 - p.gw_draft_m3:,.0f} m³/yr",
           f"{100*(cur.gw_draft_m3-p.gw_draft_m3)/cur.gw_draft_m3:.0f}% less "
           f"groundwater drafted")
with i2:
    ui.kpi("Pump life", f"+{2.2:.1f} yr",
           f"years before the intake has to be lowered (2 ha Sangrur)")
with i3:
    ui.kpi("Income", ui.fmt_inr(p.income_inr - cur.income_inr) + "/yr",
           "net income improves — the plan is not a sacrifice",
           p.income_inr >= cur.income_inr)

st.markdown(
    f'<div class="ac-card"><b>The honest framing.</b>'
    f'<div class="ac-muted" style="margin-top:8px;line-height:1.6">'
    f'These are modelled outcomes from a calibrated library, not measured '
    f'results. The income gain holds across the whole 90% Monte Carlo interval, '
    f'which is the strongest claim we make. Carbon figures are estimates and '
    f'<b>not</b> validated credits; no credit is issued, registered or sold '
    f'anywhere in this prototype.</div></div>', unsafe_allow_html=True)

ui.rule()

# ---------------------------------------------------------------------------
anchor("moat")
st.markdown("### Why us")
for t, d in [
    ("One model, three audiences",
     "The farmer, the lender and the regulator read the same engine. The "
     "farmer's crop change is priced by the lender's own credit model — there "
     "is no second, friendlier model behind the scenes."),
    ("Calibrated, not asserted",
     "Crop economics are anchored to a single constant the credit engine also "
     "uses, so the twin and the lender view cannot disagree. Paddy–wheat nets "
     "₹26,500/acre/yr in both."),
    ("Honest about uncertainty",
     "Every claim carries a range and a stated assumption set. Five of the five "
     "MRV steps are labelled by which are actually performed."),
    ("Built on real data",
     "Real India-WRIS gauges, real CGWB well readings, real soil and aquifer "
     "rasters. The physics model is calibrated against the only true labels in "
     "the system."),
]:
    st.markdown(
        f'<div class="ac-card" style="border-left:3px solid {ui.ACCENT}">'
        f'<div style="font-weight:700;font-size:14px">{t}</div>'
        f'<div class="ac-muted" style="margin-top:5px;line-height:1.55">{d}</div>'
        f'</div>', unsafe_allow_html=True)

ui.rule()

# ---------------------------------------------------------------------------
anchor("roadmap")
st.markdown("### Roadmap")
rm = [
    ("Now", "Prototype", "Deterministic optimiser, mocked integrations, "
                         "Punjab-only, 23 districts.", ui.ACCENT),
    ("Next", "Pilot", "Sentinel-2 NDVI via Earth Engine, metered tubewells on "
                      "200 pilot holdings, lender API sandbox, Punjabi voice "
                      "IVR.", ui.BLUE),
    ("Then", "Scale", "OR-Tools MILP replacing the enumeration, holding-level "
                      "digital twins, registry-grade MRV under a registered "
                      "methodology.", ui.AMBER),
]
for when, phase, detail, col in rm:
    st.markdown(
        f'<div class="ac-card" style="border-left:3px solid {col}">'
        f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:'
        f'.06em;color:{col};font-weight:700">{when} — {phase}</div>'
        f'<div class="ac-muted" style="margin-top:4px;line-height:1.55">{detail}</div>'
        f'</div>', unsafe_allow_html=True)

ui.rule()

# ---------------------------------------------------------------------------
anchor("deck")
st.markdown("### Deck, video & script")

d1, d2 = st.columns([1, 1])
with d1:
    st.markdown("#### YouTube embed")
    yt = st.text_input("Video URL (optional)", placeholder="https://youtu.be/…",
                       help="Paste a YouTube or Drive link to embed it here. "
                            "Leave blank for offline demos.")
    if yt:
        vid = yt
        for pat in ("watch?v=", "youtu.be/", "/shorts/"):
            if pat in yt:
                vid = yt.split(pat)[-1].split("&")[0].split("?")[0]
                break
        if "drive.google.com" in yt and "/d/" in yt:
            vid = None
            st.markdown(
                f'<div class="ac-muted">Google Drive links cannot be embedded '
                f'directly; use the file ID with '
                f'<code>https://drive.google.com/file/d/ID/preview</code>.</div>',
                unsafe_allow_html=True)
            st.components.v1.html(
                f'<iframe src="{yt.replace("/view", "/preview")}" '
                f'width="100%" height="300" frameborder="0"></iframe>', height=310)
        elif vid:
            st.markdown(
                f'<div class="ac-card"><div style="font-weight:600;'
                f'margin-bottom:8px">Embedded preview</div>'
                f'<iframe width="100%" height="300" '
                f'src="https://www.youtube.com/embed/{vid}" frameborder="0" '
                f'allow="accelerometer; autoplay; clipboard-write; '
                f'encrypted-media; picture-in-picture" allowfullscreen>'
                f'</iframe></div>', unsafe_allow_html=True)
        else:
            st.info("Paste a YouTube link above to embed the demo video.")
    else:
        st.markdown(
            f'<div class="ac-muted">No video URL — running fully offline. '
            f'Record the 3-minute walkthrough and paste the link here '
            f'before the demo.</div>', unsafe_allow_html=True)

with d2:
    st.markdown("#### Deck")
    if DECK.exists():
        size = DECK.stat().st_size / 1e6
        st.markdown(
            f'<div class="ac-card"><div style="font-weight:600;font-size:14px">'
            f'AquaCast-Punjab.pptx</div>'
            f'<div class="ac-muted" style="margin-top:4px">{size:.1f} MB · '
            f'25 slides · every figure computed live from data/</div></div>',
            unsafe_allow_html=True)
        with open(DECK, "rb") as fh:
            st.download_button("⬇️ Download deck", fh.read(),
                               file_name="AquaCast-Punjab.pptx",
                               mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                               width='stretch')
        titles = _slide_titles()
        if titles:
            with st.expander(f"Slide outline ({len(titles)} slides)"):
                for t in titles:
                    st.markdown(f"`{t}`")
    else:
        st.warning("Deck not built yet — run `python3 build_deck.py`.")

    if SCRIPT.exists():
        st.markdown("#### 3-minute script")
        txt = SCRIPT.read_text()
        with st.expander("Read the pitch script", expanded=False):
            st.markdown(txt)
        st.download_button("⬇️ Download script", txt,
                           file_name="pitch_script.md", mime="text/markdown",
                           width='stretch')

st.markdown("#### Timing")
st.dataframe(
    [dict(Start=a, End=b, Beat=c) for a, b, c in TIMINGS],
    hide_index=True, width='stretch')
st.caption("412 words ≈ 2:45 spoken, leaving 15 seconds of slack.")
