"""
Climate Operations (Admin)
==========================

The lender and administrator view: where to intervene, what it is worth, and
what has actually been verified.

Seven tabs, ordered the way an operations team works — see the region, find the
risk, understand the cropping, size the climate impact, target interventions,
structure the finance, then verify what happened.

Built on the district network that modules A-F already produce, so nothing here
is a separate pipeline: the same aquifer model, the same well network, the same
credit engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui                                              # noqa: E402
from src.config import DISTRICTS, get_prior                      # noqa: E402
from src.crop_library import CROPS                               # noqa: E402
from src.farm_twin import (FarmScenario, evaluate, baseline,     # noqa: E402
                           risk_band)
from src.optimizer import optimize, OBJECTIVES                   # noqa: E402
from src.carbon import RESIDUE_PATHS, IRRIGATION_LABELS          # noqa: E402
from src.impact import MRV_STEPS, verified_carbon                # noqa: E402
from src.credit_risk_engine import score as credit_score         # noqa: E402


@st.cache_data(show_spinner=False, ttl=900)
def _district_table() -> pd.DataFrame:
    rows = []
    for d in DISTRICTS:
        p = get_prior(d)
        o = evaluate(baseline(d, 2.0))
        r = credit_score(district=d,
                         predicted_level_mbgl=p.base_depth_2021_mbgl
                         + p.long_term_decline_m_per_yr * 5,
                         pump_set_depth_mbgl=p.base_depth_2021_mbgl + 6.0,
                         current_level_mbgl=p.base_depth_2021_mbgl,
                         decline_m_per_yr=p.long_term_decline_m_per_yr)
        rows.append(dict(
            District=d, **{
                "Water level 2021 (m)": round(p.base_depth_2021_mbgl, 1),
                "Decline (m/yr)": round(p.long_term_decline_m_per_yr, 2),
                "Canal share %": round(p.canal_share_pct, 0),
                "Tubewells": int(p.tubewells),
                "Active loans": int(p.active_farm_loans),
                "Avg ticket ₹": int(p.avg_ticket_inr),
                "Risk score": round(float(r.score), 1),
                "Band": r.band_label,
                "_depletion_2ha": round(o.depletion_m, 2),
                "_co2_2ha": round(o.co2e_t, 2),
            }))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=900)
def _optimise_all(district: str, hectares: float, mode: str):
    return optimize(baseline(district, hectares), mode)


df = _district_table()

st.markdown("## 🏛️ Climate Operations")
st.markdown(
    f'<div class="ac-muted">Portfolio view across {len(df)} Punjab districts. '
    f'Groundwater levels, decline rates, credit risk and the modelled climate '
    f'footprint of the default paddy–wheat rotation — all computed from the '
    f'calibrated model, not sampled from a synthetic portfolio.</div>',
    unsafe_allow_html=True)

T = st.tabs(["Regional Overview", "Water Risk", "Crop Patterns",
             "Climate Impact", "Interventions", "Finance", "Verification"])


# ===========================================================================
with T[0]:
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        ui.kpi("Districts", f"{len(df)}", "Punjab, all in scope")
    with a2:
        ui.kpi("Mean decline", f"{df['Decline (m/yr)'].mean():.2f} m/yr",
               f"worst: {df.loc[df['Decline (m/yr)'].idxmax(), 'District']} "
               f"({df['Decline (m/yr)'].max():.2f} m)")
    with a3:
        ui.kpi("Loan book exposure",
               ui.fmt_inr((df["Active loans"] * df["Avg ticket ₹"]).sum()),
               f"{df['Active loans'].sum():,} active farm loans")
    with a4:
        ui.kpi("Avg credit risk", f"{df['Risk score'].mean():.1f}",
               f"{len(df[df['Risk score'] >= 60])} districts above 60")

    ui.so_what(
        f"<b>{len(df[df['Decline (m/yr)'] > 0.8])} districts</b> are declining "
        f"faster than the 0.8 m/yr state average, and those same districts "
        f"carry <b>{ui.fmt_inr((df[df['Decline (m/yr)'] > 0.8]['Active loans'] * df[df['Decline (m/yr)'] > 0.8]['Avg ticket ₹']).sum())}</b> "
        f"of exposure. Aquifer risk and credit risk are the same risk.",
        f"Prioritise intervention on "
        f"<b>{df.loc[df['Decline (m/yr)'].idxmax(), 'District']}</b>, "
        f"<b>{df.nlargest(2, 'Decline (m/yr)').iloc[1]['District']}</b> and "
        f"<b>{df.nlargest(3, 'Decline (m/yr)').iloc[2]['District']}</b>.")

    ui.rule()
    st.markdown("#### District league table")
    st.dataframe(df.drop(columns=[c for c in df.columns if c.startswith("_")]),
                 hide_index=True, width='stretch')

    fig = px.bar(df.sort_values("Decline (m/yr)", ascending=False),
                 x="District", y="Decline (m/yr)",
                 color="Decline (m/yr)", color_continuous_scale="Reds")
    ui.style(fig, 320, "Groundwater decline by district (m/yr)", legend=False)
    st.plotly_chart(fig, width='stretch')


# ===========================================================================
with T[1]:
    st.markdown("### 💧 Water risk")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Water level 2021 (m)", x=df["District"],
                         y=df["Water level 2021 (m)"], marker_color=ui.BLUE))
    fig.add_trace(go.Scatter(name="Canal share %", x=df["District"],
                             y=df["Canal share %"], yaxis="y2",
                             line=dict(color=ui.ACCENT, width=2),
                             mode="lines+markers"))
    ui.style(fig, 340, "Depth to water and canal dependence")
    fig.update_yaxes(title="metres below ground")
    fig.update_layout(yaxis2=dict(title="canal share %", overlaying="y",
                                  side="right", showgrid=False))
    st.plotly_chart(fig, width='stretch')

    ui.so_what(
        f"Districts with a <b>low canal share</b> depend almost entirely on "
        f"groundwater, so a dry monsoon hits them twice: less recharge and more "
        f"pumping. <b>{df.loc[df['Canal share %'].idxmin(), 'District']}</b> "
        f"has just {df['Canal share %'].min():.0f}% canal cover.",
        f"Target micro-irrigation and canal-synchronised transplanting there "
        f"first.")

    ui.rule()
    st.markdown("#### Tubewell density against decline")
    fig = px.scatter(df, x="Tubewells", y="Decline (m/yr)", text="District",
                     size="Active loans", color="Risk score",
                     color_continuous_scale="RdYlGn_r")
    fig.update_traces(textposition="top center", textfont=dict(size=9))
    ui.style(fig, 420, "More wells, faster decline — and more exposure",
             legend=False)
    st.plotly_chart(fig, width='stretch')


# ===========================================================================
with T[2]:
    st.markdown("### 🌾 Crop patterns")
    st.markdown(
        f'<div class="ac-muted">The model assumes the paddy–wheat rotation as '
        f'the baseline, because that is what Punjab grows. This tab shows what '
        f'that pattern costs, and what the alternatives return per unit of '
        f'water.</div>', unsafe_allow_html=True)

    rows = []
    for k, c in CROPS.items():
        rows.append(dict(Crop=f"{c.glyph} {c.label}", Season=c.season,
                         **{"Water (m³/ha)": int(c.water_m3_ha),
                            "Net ₹/ha": int(c.net_return_inr_ha),
                            "₹/m³": round(c.water_productivity_inr_m3, 1),
                            "Duration (days)": int(c.duration_days),
                            "Residue (t/ha)": round(c.residue_t_ha, 1),
                            "Burnt share": f"{c.residue_burned_frac:.0%}",
                            "MSP realisation": f"{c.market_realisation:.0%}"}))
    cf = pd.DataFrame(rows).sort_values("₹/m³", ascending=False)
    st.dataframe(cf, hide_index=True, width='stretch')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cf["Water (m³/ha)"], y=cf["Net ₹/ha"], mode="markers+text",
        text=cf["Crop"], textposition="top center",
        marker=dict(size=cf["₹/m³"] * 3.4, color=cf["₹/m³"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="₹/m³")),
        textfont=dict(size=10)))
    ui.style(fig, 430, "Net return against water use — bubble size is ₹/m³",
             legend=False)
    fig.update_xaxes(title="Water (m³/ha)")
    fig.update_yaxes(title="Net ₹/ha")
    st.plotly_chart(fig, width='stretch')

    ui.so_what(
        f"Paddy sits far right — the highest water use for a mid-tier return. "
        f"<b>{cf.iloc[0]['Crop']}</b> earns "
        f"<b>₹{cf.iloc[0]['₹/m³']:.1f}</b> per cubic metre against paddy's "
        f"₹{CROPS['paddy'].water_productivity_inr_m3:.1f}.",
        f"The constraint on shifting is procurement, not agronomy — non-paddy "
        f"crops realise as little as "
        f"{min(c.market_realisation for c in CROPS.values()):.0%} of MSP.")


# ===========================================================================
with T[3]:
    st.markdown("### 🌍 Climate impact")
    st.markdown(
        f'<div class="ac-muted">Modelled net CO₂e of the default paddy–wheat '
        f'rotation on a 2 ha holding in each district. Excludes biogenic CO₂ '
        f'from residue burning, which is reported separately and not counted '
        f'towards warming.</div>' + ui.badge("Estimates — not validated credits"),
        unsafe_allow_html=True)

    fig = px.bar(df.sort_values("_co2_2ha", ascending=False), x="District",
                 y="_co2_2ha", color="_co2_2ha", color_continuous_scale="Oranges")
    ui.style(fig, 330, "Modelled net CO₂e, 2 ha paddy–wheat (t/yr)", legend=False)
    st.plotly_chart(fig, width='stretch')

    cur = evaluate(baseline("Sangrur", 2.0))
    cd = cur.carbon_breakdown
    labels = ["Paddy methane", "Fertiliser", "Irrigation energy",
              "Residue burning", "Machinery"]
    vals = [cd["rice_ch4"], cd["fertiliser"], cd["irrigation"],
            cd["residue"], cd["machinery"]]
    fig = go.Figure(go.Bar(y=labels, x=vals, orientation="h",
                           marker_color=[ui.RED, ui.AMBER, ui.BLUE,
                                         "#7A4E9E", ui.MUTED],
                           text=[f"{v:.2f}" for v in vals],
                           textposition="outside"))
    ui.style(fig, 300, "Where the emissions come from (t CO₂e/yr, 2 ha Sangrur)",
             legend=False)
    st.plotly_chart(fig, width='stretch')

    ui.so_what(
        f"<b>Fertiliser and paddy methane together are "
        f"{(cd['fertiliser'] + cd['rice_ch4']) / sum(vals):.0%}</b> of the "
        f"footprint. Residue burning — the visible, politicised part — is only "
        f"{cd['residue'] / sum(vals):.0%} of the warming impact.",
        f"A strategy aimed only at stubble burning addresses the smallest "
        f"slice. Alternate wetting and drying plus nitrogen management move "
        f"the number that matters.")


# ===========================================================================
with T[4]:
    st.markdown("### 🎯 Interventions")
    d = st.selectbox("District to target", DISTRICTS,
                     index=DISTRICTS.index("Sangrur")
                     if "Sangrur" in DISTRICTS else 0)
    mode = st.radio("Objective", list(OBJECTIVES), horizontal=True,
                    format_func=lambda k: f"{OBJECTIVES[k][1]} {OBJECTIVES[k][0]}")

    with st.spinner("Searching crop plans…"):
        r = _optimise_all(d, 2.0, mode)

    b, p = r.baseline_outcome, r.best_outcome
    a1, a2, a3 = st.columns(3)
    with a1:
        ui.kpi("Water saved", f"{b.gw_draft_m3 - p.gw_draft_m3:,.0f} m³",
               f"per 2 ha holding · {r.headline()}")
    with a2:
        ui.kpi("Income", ui.fmt_inr(p.income_inr - b.income_inr) + "/yr",
               "change per holding", p.income_inr >= b.income_inr)
    with a3:
        ui.kpi("Capital needed", r.investment_label,
               f"payback {r.payback_years:.1f} yr"
               if np.isfinite(r.payback_years) else "")

    st.markdown(
        f'<div class="ac-card"><div style="font-weight:600;font-size:14px">'
        f'Recommended intervention for {d}</div>'
        f'<div class="ac-muted" style="margin-top:5px">'
        f'{", ".join(f"{CROPS[k].label} {v:.0%}" for k, v in r.best.allocation.items())}'
        f' · {IRRIGATION_LABELS[r.best.irrigation]} · '
        f'{RESIDUE_PATHS[r.best.residue]["label"]}'
        f'<br>Best of {r.n_evaluated:,} candidate plans.</div></div>',
        unsafe_allow_html=True)

    ui.rule()
    st.markdown("#### Scale it to the district")
    p_ = get_prior(d)
    holdings = int(p_.active_farm_loans)
    s1, s2, s3 = st.columns(3)
    with s1:
        ui.kpi("Water saved", f"{(b.gw_draft_m3 - p.gw_draft_m3) * holdings / 1e6:,.1f} MCM",
               f"if {holdings:,} holdings adopt")
    with s2:
        ui.kpi("CO₂e avoided", f"{(b.co2e_t - p.co2e_t) * holdings:,.0f} t/yr",
               "modelled, before verification")
    with s3:
        ui.kpi("Capital deployed",
               ui.fmt_inr(r.investment_inr * holdings),
               f"₹{r.investment_inr:,.0f} per holding")
    st.caption("Linear scaling of a single representative 2 ha holding — "
               "illustrative of order of magnitude, not a district plan. Real "
               "targeting needs holding-level data on area, soil and pump depth.")


# ===========================================================================
with T[5]:
    st.markdown("### 🏦 Finance" + ui.badge("Illustrative — not an offer"))
    st.markdown(
        f'<div class="ac-muted">Total loan-book exposure by district against '
        f'the aquifer risk sitting behind it. A falling water table is a credit '
        f'event: it raises the cost of irrigation, shortens pump life and '
        f'depresses the collateral value of the land.</div>',
        unsafe_allow_html=True)

    dfx = df.copy()
    dfx["Exposure ₹"] = dfx["Active loans"] * dfx["Avg ticket ₹"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Loan book exposure", x=dfx["District"],
                         y=dfx["Exposure ₹"], marker_color=ui.BLUE))
    fig.add_trace(go.Scatter(name="Risk score", x=dfx["District"],
                             y=dfx["Risk score"], yaxis="y2",
                             line=dict(color=ui.RED, width=2),
                             mode="lines+markers"))
    ui.style(fig, 360, "Exposure and underwriting risk")
    fig.update_yaxes(title="₹ exposure")
    fig.update_layout(yaxis2=dict(title="risk score", overlaying="y",
                                  side="right", range=[0, 100], showgrid=False))
    st.plotly_chart(fig, width='stretch')

    ui.so_what(
        f"<b>{ui.fmt_inr(dfx['Exposure ₹'].sum())}</b> of farm-loan exposure "
        f"sits on land whose aquifer is falling. Slower decline lowers that "
        f"risk directly — a crop change is a credit intervention.",
        f"Price the transition: subsidised micro-irrigation converts an "
        f"unsecured climate risk into a secured asset-backed loan.")

    ui.rule()
    st.markdown("#### Scheme stack available to every intervention")
    for k, v in [
        ("PMKSY — Per Drop More Crop", "55% micro-irrigation (small/marginal), 45% others"),
        ("PM-KUSUM", "60% stand-alone solar pump subsidy (30% central + 30% state)"),
        ("SMAM", "50% on residue-management machinery (Happy Seeder, super SMS)"),
        ("Kisan Credit Card", "7% less 3% prompt-repayment subvention = 4% effective"),
        ("NABARD refinance", "Concessional windows for climate adaptation"),
    ]:
        st.markdown(f"- **{k}** — {v}")
    st.caption("Published scheme rates. Eligibility is not assessed here; this "
               "prototype performs no KYC, no bureau check and no sanction.")


# ===========================================================================
with T[6]:
    st.markdown("### ✅ Verification")
    st.markdown(
        f'<div class="ac-muted">MRV is where most climate claims collapse. This '
        f'tab states plainly which steps the platform performs and which it '
        f'does not.</div>' + ui.badge("Modelled — no monitoring performed"),
        unsafe_allow_html=True)

    for s in MRV_STEPS:
        col = {"modelled": ui.BLUE, "partial": ui.AMBER}.get(s["status"], ui.MUTED)
        st.markdown(
            f'<div class="ac-card" style="border-left:3px solid {col}">'
            f'<div style="font-weight:600;font-size:13.5px">{s["step"]} '
            f'<span style="color:{col};font-size:11px;text-transform:uppercase;'
            f'letter-spacing:.05em">{s["status"]}</span></div>'
            f'<div class="ac-muted">{s["detail"]}</div></div>',
            unsafe_allow_html=True)

    ui.rule()
    st.markdown("#### What would make this verifiable")
    st.markdown(
        f'<div class="ac-card"><b>1 · Meter the tubewells.</b> Flow meters, or '
        f'metered electricity feeders as a proxy where pumps are unmetered. '
        f'This converts the largest claim — water saved — from modelled to '
        f'measured.<br><br>'
        f'<b>2 · Remote-sense the area and the residue.</b> Sentinel-2 at 10 m '
        f'resolves both the cropped area and whether stubble was burnt, at '
        f'farm-block scale.<br><br>'
        f'<b>3 · Register a methodology.</b> Water-saving and soil-carbon '
        f'claims need an approved baseline, and additionality has to be '
        f'argued holding by holding.<br><br>'
        f'<b>4 · Validate and issue.</b> An accredited VVB, then registry '
        f'issuance. Nothing in this prototype issues or sells a credit.</div>',
        unsafe_allow_html=True)

    with st.spinner("Running Monte Carlo…"):
        rr = _optimise_all("Sangrur", 2.0, "balanced")
        from src.impact import verify
        rep = verify(baseline("Sangrur", 2.0), rr.best, draws=800)
    v = verified_carbon(rep)
    st.markdown(f'<div class="ac-sowhat"><b>{v["note"]}</b></div>',
                unsafe_allow_html=True)
    st.dataframe(rep.frame(), hide_index=True, width='stretch')
