"""
Farmer dashboard
================

One farm, seen whole. Eight tabs, in the order a decision is actually made:
what is happening (Overview), why (Water, Crop), what to do instead
(Climate Twin, Circularity), how to pay for it (Climate Finance), whether it
worked (Impact), and what to do this week (AI Advisor).

Every tab answers "so what?" before it shows a second chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui                                             # noqa: E402
from src.crop_library import CROPS, crop_frame, HA_PER_ACRE     # noqa: E402
from src.farm_twin import (FarmScenario, evaluate, compare, baseline,   # noqa: E402
                           risk_band, SEASON_ORDER, season_of)
from src.carbon import (residue_pathways, residue_economics,    # noqa: E402
                        IRRIGATION_LABELS, RESIDUE_PATHS, PADDY_RESIDUE_T_HA)
from src.optimizer import optimize, OBJECTIVES, IRRIGATION_CAPEX_INR_PER_HA  # noqa: E402
from src.climate_finance import (build_package, lender_view,    # noqa: E402
                                 years_to_pump_failure)
from src.impact import verify, verified_carbon, MRV_STEPS        # noqa: E402
from src.advisor import recommend, llm_narrate, PRIORITY_GLYPH   # noqa: E402
from src.satellite import field_timeseries, SIMULATED_NOTE       # noqa: E402
from src.config import DISTRICTS, get_prior                      # noqa: E402
from src.climate_finance import PUMP_INTAKE_BELOW_BASE_M              # noqa: E402


# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def _opt(district, hectares, mode, _alloc_key):
    return optimize(baseline(district, hectares), mode)


@st.cache_data(show_spinner=False, ttl=600)
def _verify(cur_key, prop_key, district, hectares):
    return verify(FarmScenario(**cur_key), FarmScenario(**prop_key), draws=1200)


def _scn_tuple(s: FarmScenario) -> dict:
    return dict(name=s.name, district=s.district, hectares=s.hectares,
                allocation=dict(s.allocation), irrigation=s.irrigation,
                residue=s.residue, canal=s.canal, rainfall=s.rainfall,
                transplant_shift_days=s.transplant_shift_days)


# ---------------------------------------------------------------------------
# Farm setup — shared across every tab
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div style="font-size:11px;font-weight:700;color:{ui.MUTED};'
                f'letter-spacing:.06em;text-transform:uppercase">Your farm</div>',
                unsafe_allow_html=True)
    district = st.selectbox("District", DISTRICTS, index=DISTRICTS.index("Sangrur")
                            if "Sangrur" in DISTRICTS else 0)
    hectares = st.slider("Land (hectares)", 0.5, 10.0, 2.0, 0.5,
                         help="1 hectare = 2.47 acres")
    st.caption(f"{hectares / HA_PER_ACRE:.1f} acres · "
               f"double-cropped = {hectares * 2:.1f} ha of crops a year")

base = baseline(district, hectares)
cur = evaluate(base)

st.markdown(f"## 🌾 {district} · {hectares:.1f} ha "
            f"({hectares / HA_PER_ACRE:.1f} acres)")
st.markdown(
    f'<div class="ac-muted">Baseline plan: paddy–wheat rotation, flood '
    f'irrigation, residue burnt — the system Punjab has today. Every figure '
    f'below is computed from the calibrated crop library and this farm\'s own '
    f'water balance.</div>', unsafe_allow_html=True)

T = st.tabs(["Overview", "Water Intelligence", "Crop Intelligence",
             "Climate Twin", "Circularity", "Climate Finance", "Impact",
             "AI Advisor"])


# ===========================================================================
# 1 · OVERVIEW
# ===========================================================================
with T[0]:
    with st.spinner("Searching crop plans…"):
        r = _opt(district, hectares, "balanced", tuple(sorted(base.allocation)))
    b, p = r.baseline_outcome, r.best_outcome

    yrs = years_to_pump_failure(base)
    band, colour = risk_band(cur.climate_risk)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi("Net income", ui.fmt_inr(cur.income_inr),
               f"{ui.fmt_inr(cur.income_inr / (hectares / HA_PER_ACRE))}/acre/yr",
               "over total cost")
    with c2:
        ui.kpi("Groundwater drafted", f"{cur.gw_draft_m3:,.0f} m³",
               f"{cur.water_demand_m3:,.0f} m³ total field water")
    with c3:
        ui.kpi("Aquifer beneath you", f"{cur.depletion_m:.2f} m/yr",
               f"{cur.depletion_m / 0.8:.1f}× the district average of 0.8 m/yr")
    with c4:
        ui.kpi("Pump intake fails in",
               "—" if not np.isfinite(yrs) else f"{yrs:.1f} yr",
               f"climate risk {cur.climate_risk:.0f}/100 ({band})")

    ui.so_what(
        f"This farm draws the aquifer down <b>{cur.depletion_m:.1f} m a year</b> "
        f"— {cur.depletion_m / 0.8:.1f}× the district average, because paddy–wheat "
        f"is the most water-intensive rotation on the ground.",
        f"Shift to the recommended plan: <b>{r.headline()}</b> and gain "
        f"{years_to_pump_failure(r.best) - yrs:.1f} more years before the pump "
        f"has to be replaced.")

    ui.rule()
    st.markdown("### The one recommendation")
    alloc = ", ".join(f"{CROPS[k].label} {v:.0%}" for k, v in r.best.allocation.items())
    st.markdown(
        f'<div class="ac-card"><div style="font-size:15px;font-weight:600;'
        f'margin-bottom:6px">{alloc}</div>'
        f'<div class="ac-muted">{IRRIGATION_LABELS[r.best.irrigation]} · '
        f'{RESIDUE_PATHS[r.best.residue]["label"]}</div>'
        f'<div style="margin-top:10px;font-size:13.5px">'
        f'<b>{r.headline()}</b> against your current plan, on '
        f'{r.n_evaluated:,} candidate plans evaluated.'
        f'</div></div>', unsafe_allow_html=True)

    g1, g2 = st.columns([1.15, 1])
    with g1:
        deltas = pd.DataFrame([
            dict(Metric="Groundwater", Current=b.gw_draft_m3, Proposed=p.gw_draft_m3),
            dict(Metric="Net income", Current=b.income_inr, Proposed=p.income_inr),
            dict(Metric="CO₂e", Current=b.co2e_t, Proposed=p.co2e_t),
            dict(Metric="Residue burned", Current=b.residue_burned_t,
                 Proposed=p.residue_burned_t),
        ])
        fig = go.Figure()
        for i, m in enumerate(deltas.Metric):
            bv = float(deltas.loc[i, "Current"]); pv = float(deltas.loc[i, "Proposed"])
            fig.add_trace(go.Bar(
                name=m, x=[100 * (pv - bv) / max(bv, 1e-9)],
                y=[m], orientation="h",
                marker_color=[ui.ACCENT if pv <= bv else ui.BLUE][0],
                text=[f"{100*(pv-bv)/max(bv,1e-9):+.0f}%"],
                textposition="outside", hovertemplate=f"{m}: "
                f"{bv:,.0f} → {pv:,.0f}<extra></extra>"))
        ui.style(fig, 260, "% change against your current plan", legend=False)
        fig.update_xaxes(range=[-110, 40], ticksuffix="%", title=None)
        st.plotly_chart(fig, width='stretch')
    with g2:
        cd = b.carbon_breakdown
        labels = [k.replace("_t_ha", "").replace("_", " ").title()
                  for k in ("rice_ch4", "fertiliser", "irrigation", "residue",
                            "machinery")]
        vals = [cd["rice_ch4"], cd["fertiliser"], cd["irrigation"],
                cd["residue"], cd["machinery"]]
        fig = go.Figure(go.Pie(
            labels=labels, values=vals, hole=.62,
            marker=dict(colors=[ui.RED, ui.AMBER, ui.BLUE, "#7A4E9E", ui.MUTED]),
            textinfo="percent", textfont=dict(size=11)))
        ui.style(fig, 260, f"Where this farm's {b.co2e_t:.1f} t CO₂e comes from")
        st.plotly_chart(fig, width='stretch')

    st.caption(f"Carbon split is a prototype estimate from published emission "
               f"factors. A further {b.co2e_biogenic_t:.1f} t of biogenic CO₂ "
               f"from burning last season's stubble is reported separately and "
               f"not counted in the total.")

    ui.rule()
    st.markdown("### Seasonal greenness")
    ts = field_timeseries(cur.crop_area_ha, hectares)
    fig = go.Figure(go.Scatter(x=ts.date, y=ts.ndvi, mode="lines",
                               line=dict(color=ui.ACCENT, width=2),
                               fill="tozeroy", fillcolor="rgba(11,110,79,.10)"))
    ui.style(fig, 230, "Crop cover through the year", legend=False)
    fig.update_yaxes(title="NDVI", range=[0, .95])
    st.plotly_chart(fig, width='stretch')
    st.caption(SIMULATED_NOTE)


# ===========================================================================
# 2 · WATER INTELLIGENCE
# ===========================================================================
with T[1]:
    prior = get_prior(district)
    st.markdown("### 💧 Water intelligence")

    a1, a2, a3 = st.columns(3)
    with a1:
        ui.kpi("Groundwater draft", f"{cur.gw_draft_m3:,.0f} m³/yr",
               f"{cur.gw_draft_m3 / hectares:,.0f} m³ per hectare")
    with a2:
        ui.kpi("Canal supply", f"{cur.canal_supply_m3:,.0f} m³/yr",
               f"{prior.canal_share_pct:.0f}% of the district's irrigation")
    with a3:
        ui.kpi("Net depletion", f"{cur.depletion_m:.2f} m/yr",
               "beneath this farm, after recharge and return flow")

    ui.so_what(
        f"Pumping <b>{cur.gw_draft_m3:,.0f} m³</b> a year drops the water table "
        f"under your own land by <b>{cur.depletion_m:.2f} m/yr</b> — deeper than "
        f"the {prior.long_term_decline_m_per_yr:.1f} m/yr district trend, because "
        f"the district figure is diluted by fallow and canal-irrigated land.",
        f"At that rate a pump set {PUMP_INTAKE_BELOW_BASE_M:.0f} m below the "
        f"2021 level has "
        + (f"<b>{yrs:.1f} years</b> left." if np.isfinite(yrs)
           else "no near-term limit."))

    ui.rule()
    st.markdown("#### Where the water goes")
    pc = cur.per_crop.copy()
    if len(pc):
        fig = go.Figure(go.Bar(
            x=pc["Crop"], y=pc["Groundwater (m³)"],
            marker=dict(color=[CROPS[k].colour for k in cur.crop_area_ha]),
            text=[f"{v:,.0f}" for v in pc["Groundwater (m³)"]],
            textposition="outside"))
        ui.style(fig, 300, "Groundwater draft by crop", legend=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Aquifer context")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(
            f'<div class="ac-card"><div class="ac-label" style="font-size:11px;'
            f'font-weight:600;color:{ui.MUTED};text-transform:uppercase;'
            f'letter-spacing:.06em">Water level, 2021 baseline</div>'
            f'<div class="ac-value" style="font-size:28px;font-weight:700">'
            f'{prior.base_depth_2021_mbgl:.1f} m</div>'
            f'<div class="ac-muted">below ground level · falling '
            f'{prior.long_term_decline_m_per_yr:.2f} m/yr district-wide</div>'
            f'</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ac-card"><div class="ac-label" style="font-size:11px;'
            f'font-weight:600;color:{ui.MUTED};text-transform:uppercase;'
            f'letter-spacing:.06em">Specific yield</div>'
            f'<div class="ac-value" style="font-size:28px;font-weight:700">'
            f'{prior.specific_yield:.2f}</div>'
            f'<div class="ac-muted">Central Punjab unconfined alluvium — every '
            f'm³ removed drops the table '
            f'{1/(hectares*1e4*prior.specific_yield)*1000:.2f} mm under this farm'
            f'</div></div>', unsafe_allow_html=True)
    with b2:
        years = np.arange(0, 11)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=2026 + years, y=prior.base_depth_2021_mbgl + cur.depletion_m * years,
            mode="lines", name="Your current plan",
            line=dict(color=ui.RED, width=2.5)))
        with st.spinner("Searching…"):
            rb = _opt(district, hectares, "balanced", tuple(sorted(base.allocation)))
        yrp = years_to_pump_failure(rb.best)
        # reuse the optimiser drawdown reduction rather than re-deriving
        red = rb.best_outcome.depletion_m
        fig.add_trace(go.Scatter(
            x=2026 + years, y=prior.base_depth_2021_mbgl + red * years,
            mode="lines", name="Recommended plan",
            line=dict(color=ui.ACCENT, width=2.5)))
        fig.add_hline(y=prior.base_depth_2021_mbgl + PUMP_INTAKE_BELOW_BASE_M,
                      line_dash="dot", line_color=ui.AMBER,
                      annotation_text="typical pump intake")
        ui.style(fig, 300, "Depth to water under this farm")
        fig.update_yaxes(title="metres below ground", autorange="reversed")
        st.plotly_chart(fig, width='stretch')


# ===========================================================================
# 3 · CROP INTELLIGENCE
# ===========================================================================
with T[2]:
    st.markdown("### 🌱 Crop intelligence")
    st.markdown(
        f'<div class="ac-muted">Rupees earned per cubic metre of groundwater is '
        f'the number that matters when the aquifer is the constraint. Ranked '
        f'below for every crop in the library.</div>', unsafe_allow_html=True)

    cf = crop_frame().sort_values("₹/m³ water", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cf["Crop"], y=cf["₹/m³ water"],
        marker=dict(color=[CROPS[k].colour for k in cf["key"]]),
        text=[f"₹{v:.1f}" for v in cf["₹/m³ water"]], textposition="outside",
        name="Net ₹/m³"))
    ui.style(fig, 300, "Net rupees per cubic metre of groundwater", legend=False)
    st.plotly_chart(fig, width='stretch')

    best_alt = cf[cf["key"] != "paddy"].iloc[0]
    ui.so_what(
        f"Paddy returns <b>₹{CROPS['paddy'].water_productivity_inr_m3:.1f}</b> "
        f"per cubic metre; <b>{best_alt['Crop']}</b> returns "
        f"<b>₹{best_alt['₹/m³ water']:.1f}</b> — "
        f"{best_alt['₹/m³ water'] / CROPS['paddy'].water_productivity_inr_m3:.1f}× "
        f"the value from the same water.",
        f"Move part of the paddy area across and the water saved is worth more "
        f"than the margin given up.")

    ui.rule()
    st.markdown("#### What each crop actually costs you")
    show = cf[["Crop", "Season", "Water (m³/ha)", "Net ₹/ha", "₹/m³ water",
               "Residue (t/ha)"]].copy()
    st.dataframe(show, hide_index=True, width='stretch')
    st.caption("Net returns are quoted over total cost — including imputed "
               "family labour and land rent. Gross MSP returns are roughly "
               "double and would flatter every option.")

    ui.rule()
    st.markdown("#### The catch the table does not show")
    st.markdown(
        f'<div class="ac-card">Non-paddy crops carry a <b>market '
        f'realisation</b> discount of up to '
        f'{100*(1-min(CROPS[k].market_realisation for k in CROPS)):.0f}% against '
        f'MSP, because paddy and wheat have assured procurement and the '
        f'alternatives do not. That is the real reason diversification is hard, '
        f'and it is why the model does not simply say "grow maize".</div>',
        unsafe_allow_html=True)
    for k in ("paddy", "maize", "pulses"):
        c = CROPS[k]
        st.markdown(f"- **{c.label}** — realises "
                    f"{c.market_realisation:.0%} of MSP ({ui.fmt_inr(c.msp_inr_per_t)}/t)"
                    f" · {c.notes}")


# ===========================================================================
# 4 · CLIMATE TWIN
# ===========================================================================
with T[3]:
    st.markdown("### 🔬 Climate Farm Twin")
    st.markdown(
        f'<div class="ac-muted">Change the farm, watch every outcome move '
        f'together. Crop mix, irrigation, residue, canal supply and monsoon are '
        f'coupled — the twin recomputes the whole system on each change.</div>',
        unsafe_allow_html=True)

    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("**Kharif (monsoon season)**")
        kh = {}
        for k in ("paddy", "maize", "cotton"):
            kh[k] = st.slider(f"{CROPS[k].glyph} {CROPS[k].label}", 0.0, 1.0,
                              1.0 if k == "paddy" else 0.0, 0.05,
                              key=f"tw_kh_{k}")
        st.markdown("**Rabi (winter season)**")
        rb_ = {}
        for k in ("wheat", "oilseeds"):
            rb_[k] = st.slider(f"{CROPS[k].glyph} {CROPS[k].label}", 0.0, 1.0,
                               1.0 if k == "wheat" else 0.0, 0.05,
                               key=f"tw_rb_{k}")
        st.markdown("**Zaid (summer window)**")
        zd = st.slider(f"{CROPS['pulses'].glyph} {CROPS['pulses'].label}",
                       0.0, 1.0, 0.0, 0.05, key="tw_zd",
                       help="Short-duration moong between the wheat harvest and "
                            "the paddy transplant. Capped at half the farm — it "
                            "is a rotating catch crop, not a third season.")
        alloc = {**kh, **rb_}
        if zd > 0:
            alloc["pulses"] = zd

        irrigation = st.selectbox("Irrigation", list(IRRIGATION_LABELS),
                                  format_func=lambda k: IRRIGATION_LABELS[k])
        residue = st.selectbox("Residue management", list(RESIDUE_PATHS),
                               format_func=lambda k: RESIDUE_PATHS[k]["label"])
        canal = st.selectbox("Canal availability", ["low", "normal", "high"])
        rainfall = st.selectbox("Monsoon scenario",
                                ["-20%", "-10%", "normal", "+10%"], index=2)
        shift = st.slider("Delay paddy transplant (days)", 0, 30, 0, 1)

    prop = FarmScenario(name="Proposed", district=district, hectares=hectares,
                        allocation=alloc, irrigation=irrigation, residue=residue,
                        canal=canal, rainfall=rainfall,
                        transplant_shift_days=shift)
    comp = compare(base, prop)
    po = comp["proposed"]

    with right:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ui.kpi("Water", f"{po.water_demand_m3:,.0f} m³",
                   f"{comp['water_pct']:+.0f}%", comp["water_pct"] < 0)
        with c2:
            ui.kpi("Net income", ui.fmt_inr(po.income_inr),
                   f"{comp['income_pct']:+.1f}%", comp["income_pct"] >= 0)
        with c3:
            ui.kpi("CO₂e", f"{po.co2e_t:.1f} t",
                   f"{comp['co2_pct']:+.0f}%", comp["co2_pct"] < 0)
        with c4:
            ui.kpi("Drawdown", f"{po.depletion_m:.2f} m/yr",
                   f"{comp['depletion_delta_m']:+.2f} m",
                   comp["depletion_delta_m"] < 0)

        band_p, col_p = risk_band(po.climate_risk)
        band_c, col_c = risk_band(cur.climate_risk)
        st.markdown(
            f'<div class="ac-card"><span style="font-size:12px;color:{ui.MUTED}">'
            f'CLIMATE RISK</span> &nbsp;'
            f'<span style="color:{col_c};font-weight:700">{band_c}</span>'
            f' &nbsp;→&nbsp; '
            f'<span style="color:{col_p};font-weight:700">{band_p}</span>'
            f'<span style="float:right;color:{ui.MUTED};font-size:12px">'
            f'{cur.climate_risk:.0f} → {po.climate_risk:.0f} of 100</span>'
            f'</div>', unsafe_allow_html=True)

        st.dataframe(comp["rows"], hide_index=True, width='stretch')

        cats = ["Water use (m³)", "Net income (₹)", "CO₂e (t/yr)",
                "Residue burned (t)", "Climate risk"]
        bv = [cur.water_demand_m3, cur.income_inr, cur.co2e_t,
              cur.residue_burned_t, cur.climate_risk]
        pv = [po.water_demand_m3, po.income_inr, po.co2e_t,
              po.residue_burned_t, po.climate_risk]
        nb = [v / max(max(abs(x) for x in (a, b2_)), 1e-9)
              for v, a, b2_ in zip(bv, bv, pv)]
        nv = [v / max(max(abs(x) for x in (a, b2_)), 1e-9)
              for v, a, b2_ in zip(pv, bv, pv)]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=nb + [nb[0]], theta=cats + [cats[0]],
                                      fill="toself", name="Current",
                                      line=dict(color=ui.RED)))
        fig.add_trace(go.Scatterpolar(r=nv + [nv[0]], theta=cats + [cats[0]],
                                      fill="toself", name="Proposed",
                                      line=dict(color=ui.ACCENT)))
        ui.style(fig, 380, "Both plans, normalised")
        fig.update_polars(radialaxis=dict(visible=True, range=[0, 1.15]))
        st.plotly_chart(fig, width='stretch')

    ui.rule()
    st.markdown("#### Let the optimiser search for you")
    m1, m2 = st.columns([1, 3])
    with m1:
        mode = st.radio("Objective", list(OBJECTIVES),
                        format_func=lambda k: f"{OBJECTIVES[k][1]} {OBJECTIVES[k][0]}")
    with m2:
        with st.spinner(f"Evaluating candidate plans…"):
            ro = _opt(district, hectares, mode, tuple(sorted(base.allocation)))
        st.markdown(
            f'<div class="ac-sowhat"><b>{OBJECTIVES[mode][2]}.</b> Best of '
            f'{ro.n_evaluated:,} plans: '
            f'{", ".join(f"{CROPS[k].label} {v:.0%}" for k, v in ro.best.allocation.items())}'
            f' · {IRRIGATION_LABELS[ro.best.irrigation]} · '
            f'{RESIDUE_PATHS[ro.best.residue]["label"]}. '
            f'<b>{ro.headline()}</b>.</div>', unsafe_allow_html=True)
        st.caption(f"Needs {ro.investment_label}"
                   + (f" — {ro.payback_years:.1f} yr payback."
                      if np.isfinite(ro.payback_years) else "."))

    st.markdown("#### The trade-off you are choosing between")
    pf = ro.pareto
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pf.gw_m3, y=pf.income_inr, mode="lines+markers",
        marker=dict(size=7, color=ui.ACCENT),
        line=dict(color=ui.ACCENT, width=2), name="Efficient frontier",
        text=[", ".join(f"{CROPS[k].label} {v:.0%}" for k, v in a.items()
                        if v > 0) for a in pf.allocation],
        hovertemplate="%{text}<br>Water %{x:,.0f} m³<br>Income ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[cur.gw_draft_m3], y=[cur.income_inr], mode="markers",
        marker=dict(size=15, color=ui.RED, symbol="x", line=dict(width=2)),
        name="Your farm today"))
    fig.add_trace(go.Scatter(
        x=[ro.best_outcome.gw_draft_m3], y=[ro.best_outcome.income_inr],
        mode="markers", marker=dict(size=13, color=ui.AMBER,
                                    line=dict(color="white", width=2)),
        name="Recommended"))
    ui.style(fig, 430, "Income against groundwater — every point is a real option")
    fig.update_xaxes(title="Groundwater drafted (m³/yr)")
    fig.update_yaxes(title="Net income (₹/yr)")
    st.plotly_chart(fig, width='stretch')
    st.caption("The line is the efficient frontier: plans on it cannot be beaten "
               "on income without using more water, or on water without giving "
               "up income. Your current farm sits well inside it — strictly "
               "worse than plans that are available to you.")


# ===========================================================================
# 5 · CIRCULARITY
# ===========================================================================
with T[4]:
    st.markdown("### ♻️ Circularity & residue")
    circ = cur.circularity
    a1, a2, a3 = st.columns(3)
    with a1:
        ui.kpi("Residue generated", f"{circ['residue_generated_t']:.1f} t/yr",
               f"across {cur.cropped_ha:.1f} ha cropped")
    with a2:
        ui.kpi("Currently burnt", f"{circ['residue_burned_t']:.1f} t/yr",
               f"{circ['burned_share_pct']:.0f}% of what the farm produces")
    with a3:
        ui.kpi("CO₂e from burning", f"{circ['co2_if_burned_t']:.2f} t/yr",
               f"plus {circ['biogenic_co2_t']:.1f} t biogenic CO₂ (not counted)")

    ui.so_what(
        f"Only <b>paddy straw</b> is hard to sell — at about "
        f"₹{CROPS['paddy'].residue_price_inr_per_t:,.0f}/t it is poor fodder. "
        f"<b>Wheat straw</b> is prime dry fodder at "
        f"₹{CROPS['wheat'].residue_price_inr_per_t:,.0f}/t, which is why nobody "
        f"burns it. That price gap is the whole problem.",
        f"Divert the paddy straw — the table below prices every route on "
        f"this farm's actual tonnage.")

    ui.rule()
    st.markdown("#### Every route, priced on your tonnage")
    rows = []
    for key, p_ in RESIDUE_PATHS.items():
        e = residue_economics(cur.crop_area_ha, key)
        rows.append(dict(
            Pathway=f"{p_['glyph']} {p_['label']}",
            **{"Water saving": f"{p_['water_saving_pct']:.0f}%",
               "Revenue ₹/yr": round(e["value_inr"]),
               "Cost ₹/yr": round(e["cost_inr"]),
               "Net ₹/yr": round(e["net_value_inr"]),
               "Soil impact": p_["soil_impact"]}))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    best_key = max(RESIDUE_PATHS, key=lambda k: residue_economics(
        cur.crop_area_ha, k)["net_value_inr"])
    best_e = residue_economics(cur.crop_area_ha, best_key)
    st.markdown(
        f'<div class="ac-sowhat"><b>Best value here:</b> '
        f'{RESIDUE_PATHS[best_key]["label"]} at '
        f'<b>₹{best_e["net_value_inr"]:,.0f}/yr</b>. '
        f'{RESIDUE_PATHS[best_key]["note"]}</div>', unsafe_allow_html=True)

    ui.rule()
    st.markdown("#### Residue by crop")
    rd = pd.DataFrame([
        dict(Crop=f"{CROPS[k].glyph} {CROPS[k].label}",
             **{"Generated (t)": round(CROPS[k].residue_t_ha * ha, 2),
                "Burnt (t)": round(CROPS[k].residue_burned_t_ha * ha, 2),
                "Burnt share": f"{CROPS[k].residue_burned_frac:.0%}",
                "Farm price ₹/t": int(CROPS[k].residue_price_inr_per_t)})
        for k, ha in cur.crop_area_ha.items()])
    st.dataframe(rd, hide_index=True, width='stretch')

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sold / used", x=rd["Crop"],
                         y=[g - b_ for g, b_ in zip(rd["Generated (t)"], rd["Burnt (t)"])],
                         marker_color=ui.ACCENT))
    fig.add_trace(go.Bar(name="Burnt", x=rd["Crop"], y=rd["Burnt (t)"],
                         marker_color=ui.RED))
    ui.style(fig, 300, "Residue fate by crop", legend=True)
    fig.update_layout(barmode="stack")
    st.plotly_chart(fig, width='stretch')


# ===========================================================================
# 6 · CLIMATE FINANCE
# ===========================================================================
with T[5]:
    st.markdown("### 🏦 Climate finance" + ui.badge(
        "Illustrative — not an offer of credit"))
    with st.spinner("Searching…"):
        rf = _opt(district, hectares, "balanced", tuple(sorted(base.allocation)))

    size = st.selectbox("Farm size class", ["marginal", "small", "medium", "large"],
                        index=1,
                        help="Sets the micro-irrigation subsidy rate: 55 % for "
                             "marginal and small holdings, 45 % otherwise.")
    tenure = st.slider("Loan tenure (years)", 3, 7, 5)
    solar = st.checkbox("Include a PM-KUSUM solar pump (₹3.2 L)", value=False)

    pkg = build_package(base, rf.best, farm_size_class=size,
                        tenure_years=tenure, include_solar=solar,
                        solar_capex_inr=320_000.0 if solar else 0.0)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        ui.kpi("Capital required", ui.fmt_inr(pkg.gross_capex_inr),
               f"{IRRIGATION_LABELS[rf.best.irrigation]} + "
               f"{RESIDUE_PATHS[rf.best.residue]['label']}")
    with f2:
        ui.kpi("Government subsidy", ui.fmt_inr(pkg.subsidy_inr),
               f"{pkg.subsidy_inr / max(pkg.gross_capex_inr, 1):.0%} of capital")
    with f3:
        ui.kpi("Loan needed", ui.fmt_inr(pkg.loan_inr),
               f"{pkg.rate_pct:.0f}% KCC rate after subvention")
    with f4:
        ui.kpi("Net after EMI", ui.fmt_inr(pkg.net_annual_benefit_inr) + "/yr",
               f"EMI ₹{pkg.emi_inr:,.0f}/month")

    ui.so_what(
        f"The plan needs <b>{ui.fmt_inr(pkg.gross_capex_inr)}</b> up front, but "
        f"subsidies cover <b>{ui.fmt_inr(pkg.subsidy_inr)}</b> and the farmer "
        f"only puts in <b>{ui.fmt_inr(pkg.own_contribution_inr)}</b>.",
        f"Payback is <b>{pkg.payback_years:.1f} years</b> on the total capital, "
        f"or <b>{pkg.equity_payback_years:.1f} years</b> on the farmer's own "
        f"cash — because the subsidy does most of the work.")

    ui.rule()
    st.dataframe(pkg.summary(), hide_index=True, width='stretch')

    st.markdown("#### Which schemes apply")
    for s in pkg.schemes:
        st.markdown(
            f'<div class="ac-card"><div style="font-weight:600;font-size:14px">'
            f'{s["scheme"]}</div><div class="ac-muted">{s["basis"]} · '
            f'{s["rate"]:.0%} · {s["note"]}</div></div>', unsafe_allow_html=True)
    if not pkg.schemes:
        st.info("This plan needs no new capital equipment, so no scheme applies.")

    ui.rule()
    st.markdown("#### What the lender sees")
    lv = lender_view(base, rf.best)
    st.dataframe(lv["table"], hide_index=True, width='stretch')
    st.caption(lv["note"])
    st.caption(f"Drawdown {lv['drawdown_delta']:+.2f} m/yr · "
               f"{lv['years_gained']:+.1f} years of pump life gained.")

    ui.rule()
    st.markdown("#### Carbon revenue" + ui.badge("Estimate — not a validated credit"))
    v = verified_carbon(_verify(_scn_tuple(base), _scn_tuple(rf.best),
                                district, hectares))
    st.markdown(f'<div class="ac-card">{v["note"]}</div>', unsafe_allow_html=True)
    st.caption(f"Priced at ₹2,500/t CO₂e, an assumption inside the observed "
               f"₹1,600–₹4,100 voluntary-market range. Revenue only materialises "
               f"after a registered methodology, validation and monitoring — "
               f"none of which this prototype performs.")


# ===========================================================================
# 7 · IMPACT
# ===========================================================================
with T[6]:
    st.markdown("### 📊 Impact & verification")
    with st.spinner("Running Monte Carlo…"):
        ro_ = _opt(district, hectares, "balanced", tuple(sorted(base.allocation)))
        rep = _verify(_scn_tuple(base), _scn_tuple(ro_.best), district, hectares)

    st.markdown(
        f'<div class="ac-muted">Every figure in AquaCast is a model output, so '
        f'it is reported as a range. {rep.draws:,} Monte Carlo draws propagate '
        f'uncertainty in prices, crop water use, emission factors, residue '
        f'markets and aquifer properties — with the same draw applied to both '
        f'plans, so shared errors cancel instead of double-counting.</div>',
        unsafe_allow_html=True)

    st.dataframe(rep.frame(), hide_index=True, width='stretch')

    e_water = next(e for e in rep.estimates if e.metric == "Groundwater saved")
    e_inc = next(e for e in rep.estimates if e.metric == "Net income change")
    ui.so_what(
        f"The water saving is <b>robust</b> — {e_water.lo:,.0f} to "
        f"{e_water.hi:,.0f} m³ across the whole 90 % interval.",
        f"So is the income gain: {ui.fmt_inr(e_inc.lo)} to "
        f"{ui.fmt_inr(e_inc.hi)}, positive in every draw. The conclusion does "
        f"not depend on the assumptions being exactly right.")

    ui.rule()
    st.markdown("#### Uncertainty in the income change")
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta", value=e_inc.central,
        number=dict(prefix="₹", font=dict(size=34)),
        delta=dict(reference=0), title=dict(text="Net income change / yr")))
    ui.style(fig, 150, legend=False)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### What real MRV would require")
    for s in MRV_STEPS:
        col = {"modelled": ui.BLUE, "partial": ui.AMBER}.get(s["status"], ui.MUTED)
        st.markdown(
            f'<div class="ac-card" style="border-left:3px solid {col}">'
            f'<div style="font-weight:600;font-size:13.5px">{s["step"]} '
            f'<span style="color:{col};font-size:11px;text-transform:uppercase;'
            f'letter-spacing:.05em">{s["status"]}</span></div>'
            f'<div class="ac-muted">{s["detail"]}</div></div>',
            unsafe_allow_html=True)


# ===========================================================================
# 8 · AI ADVISOR
# ===========================================================================
with T[7]:
    st.markdown("### 🤖 AI Climate Advisor")
    with st.spinner("Searching…"):
        ra = _opt(district, hectares, "balanced", tuple(sorted(base.allocation)))
    recs = recommend(base, proposed=ra.best)

    st.markdown(
        f'<div class="ac-muted">A deterministic engine scored against this '
        f'farm\'s own numbers. Each item names the rule and the figure that '
        f'triggered it, so you can disagree with a specific assumption rather '
        f'than with an inscrutable output. {len(recs)} recommendations.</div>',
        unsafe_allow_html=True)

    for r_ in recs:
        ui.rec(r_.priority, r_.title, r_.action, r_.impact)
        with st.expander(f"Why · rule `{r_.rule}` · confidence {r_.confidence}"):
            st.markdown(f"**Finding.** {r_.finding}")
            st.markdown(f"**Mechanism.** {r_.why}")
            if r_.evidence:
                st.json(r_.evidence)

    ui.rule()
    st.markdown("#### Language layer")
    n = llm_narrate(recs)
    if n["available"]:
        st.success(n["text"])
    else:
        st.markdown(
            f'<div class="ac-card"><div style="font-weight:600">LLM narration '
            f'off {ui.badge("No API key — running fully offline")}</div>'
            f'<div class="ac-muted" style="margin-top:6px">{n["reason"]} '
            f'In production this rewrites the same recommendations in the '
            f'farmer\'s language; it never originates advice.</div></div>',
            unsafe_allow_html=True)

    st.markdown("#### Deliver the same advice by WhatsApp")
    st.markdown(
        f'<div class="ac-muted">Message delivery is mocked in this prototype — '
        f'see the <b>Farmer Phone</b> dashboard for the handset view, and '
        f'README §"Prototype vs Production Architecture" for the Twilio / '
        f'Gupshup integration path.</div>', unsafe_allow_html=True)
