"""
AI Climate Advisor
==================

Turns model output into advice a farmer can act on this season.

The "so what?" rule
-------------------
Every recommendation here follows one rule: a number without an action is not
advice. "Groundwater stress expected to increase" is a chart. "Groundwater
stress expected to increase — shift 0.5 ha of paddy to maize and you cut draft
by 1,800 m³, which is 3 more years before your pump intake goes dry" is advice.
Each item below therefore carries four things: the **finding**, the **why**, the
**action**, and the **number** that justifies it.

Deterministic by default, LLM-enhanced when a key exists
--------------------------------------------------------
The reasoning engine is a scored rule set that runs with zero credentials, and
it is deliberately transparent: each recommendation names the rule and the
figure that triggered it, so a user can disagree with a specific assumption
rather than with an inscrutable output.

``llm_narrate()`` is an optional layer that re-phrases those recommendations
into the farmer's language. It is the *only* part that needs an API key, it
never originates a recommendation, and when no key is present the app says so
instead of silently faking a response. That separation is the point: the
deterministic engine is the product, the language model is a nicety.

Swapping in a learned model
---------------------------
The crop-recommendation path is intentionally shaped so an XGBoost / LightGBM /
neural ranker can replace it without touching a single line of frontend code:
``recommend()`` returns ``list[Recommendation]``, and any model that can produce
that list — from these rules, from a trained ranker, or from an LLM — is a
drop-in replacement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .crop_library import CROPS, HA_PER_ACRE
from .farm_twin import (FarmScenario, evaluate, risk_band, season_of,
                        ZAID_MAX_SHARE)
from .carbon import IRRIGATION_LABELS, RESIDUE_PATHS
from .climate_finance import years_to_pump_failure

PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "opportunity": 3}
PRIORITY_GLYPH = {"urgent": "🔴", "high": "🟠", "medium": "🟡", "opportunity": "🟢"}


@dataclass
class Recommendation:
    rule: str
    title: str
    finding: str          # the measured fact
    why: str              # mechanism
    action: str           # what to do, concretely
    impact: str           # the number that justifies it
    priority: str
    confidence: str       # high | medium | low
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
def recommend(scn: FarmScenario,
              district_forecast: dict | None = None,
              proposed: FarmScenario | None = None,
              district: str | None = None) -> list[Recommendation]:
    """Ranked, evidence-bearing recommendations for one farm."""
    out = evaluate(scn)
    out_p = evaluate(proposed) if proposed else None
    recs: list[Recommendation] = []
    areas = out.crop_area_ha
    total_ha = max(sum(areas.values()), 1e-9)
    paddy_ha = areas.get("paddy", 0.0)
    paddy_share = paddy_ha / total_ha

    # ---- 1. pump life ---------------------------------------------------
    yrs = years_to_pump_failure(scn)
    if np.isfinite(yrs) and yrs < 6:
        recs.append(Recommendation(
            rule="pump_horizon",
            title=f"Your pump intake has about {yrs:.1f} years of water left",
            finding=(f"This farm is drawing the aquifer down "
                     f"{out.depletion_m:.2f} m/yr beneath its own footprint."),
            why=("The water table under your land is falling faster than "
                 "rainfall and canal seepage can replace it. Submersible "
                 "intakes sit a fixed depth below the surface; once the level "
                 "passes them, the pump has to be lowered or replaced."),
            action=("Cut groundwater draft now — the two levers that move it "
                    "most are paddy area and irrigation method. Deepening "
                    "later costs more than saving the water now."),
            impact=(f"{out.gw_draft_m3:,.0f} m³/yr drafted · "
                    f"{out.depletion_m:.2f} m/yr decline · "
                    f"{yrs:.1f} yr to intake failure"),
            priority="urgent" if yrs < 3 else "high",
            confidence="medium",
            evidence=dict(depletion_m=round(out.depletion_m, 2),
                          years_to_failure=round(yrs, 1))))

    # ---- 2. paddy concentration ----------------------------------------
    if paddy_share > 0.5:
        alt = min(paddy_ha * 0.5, paddy_ha)
        saving = alt * (CROPS["paddy"].water_m3_ha - CROPS["maize"].water_m3_ha)
        recs.append(Recommendation(
            rule="paddy_concentration",
            title=f"Move {alt:.1f} ha of paddy to a lower-draft kharif crop",
            finding=(f"Paddy is {paddy_share:.0%} of your cropped area and "
                     f"{paddy_ha / max(out.cropped_ha,1e-9):.0%} of your "
                     f"groundwater draft."),
            why=("Paddy needs roughly 5,500 m³/ha of groundwater against "
                 "maize's 2,090 m³/ha, and it is the only crop here that is "
                 "flooded — which is also what generates the methane."),
            action=(f"Shift {alt:.1f} ha to maize or cotton. Maize keeps "
                    f"{CROPS['maize'].net_return_inr_ha / max(CROPS['paddy'].net_return_inr_ha,1):.0%} "
                    "of paddy's net return on 38 % of the water."),
            impact=(f"{saving:,.0f} m³ less groundwater · "
                    f"₹{abs(CROPS['maize'].net_return_inr_ha - CROPS['paddy'].net_return_inr_ha) * alt:,.0f} "
                    f"income difference"),
            priority="high", confidence="high",
            evidence=dict(paddy_share=round(paddy_share, 2),
                          saving_m3=round(saving))))

    # ---- 3. irrigation method ------------------------------------------
    if scn.irrigation == "flood" and paddy_ha > 0:
        recs.append(Recommendation(
            rule="awd",
            title="Switch paddy from continuous flooding to alternate wetting",
            finding="Your paddy is modelled as continuously flooded.",
            why=("Flooding is what makes a rice field anaerobic, and an "
                 "anaerobic field emits methane — about 0.84 t CO₂e/ha/yr "
                 "here, the single largest item in paddy's footprint. "
                 "Alternate wetting and drying roughly halves it and cuts "
                 "the volume pumped by about a fifth."),
            action=("Let the field dry for 2-3 days between irrigations after "
                    "the first three weeks. No capital equipment is needed — "
                    "a field water tube costs about ₹300."),
            impact="≈0.42 t CO₂e/ha/yr less · ≈18 % less water pumped",
            priority="high", confidence="high",
            evidence=dict(ch4_t_ha=0.84)))

    # ---- 4. residue -----------------------------------------------------
    if scn.residue == "burn":
        from .carbon import residue_economics
        econ = {p: residue_economics(areas, p)["net_value_inr"]
                for p in ("incorporate", "mulch", "collect", "bioenergy")}
        bestp = max(econ, key=lambda k: econ[k])
        recs.append(Recommendation(
            rule="residue_burning",
            title=f"Stop burning {out.residue_burned_t:.1f} t of residue a year",
            finding=(f"You generate {out.residue_t:.1f} t of residue and burn "
                     f"{out.residue_burned_t:.1f} t of it."),
            why=("Burning costs nothing and takes a day, which is why it "
                 "persists — but it is the visible half of the problem and "
                 "the part a regulator will eventually stop. Note that only "
                 "paddy straw is hard to sell; wheat straw is prime fodder."),
            action=(f"Best-value route here is "
                    f"'{RESIDUE_PATHS[bestp]['label'].lower()}' at "
                    f"₹{econ[bestp]:,.0f}/yr net. "
                    + ("Baling only pays if a buyer is within haulage — "
                       "confirm the offtake first."
                       if bestp in ("collect", "bioenergy") else
                       "Costs money, but saves an irrigation by holding moisture.")),
            impact=(f"₹{econ[bestp]:,.0f}/yr net · "
                    f"{out.circularity['co2_avoided_t']:.2f} t CO₂e/yr avoided"),
            priority="high", confidence="medium",
            evidence=dict(net_value_inr=round(econ[bestp]),
                          pathway=bestp)))

    # ---- 5. the Zaid window --------------------------------------------
    if areas.get("pulses", 0.0) < scn.hectares * ZAID_MAX_SHARE - 1e-6:
        room = scn.hectares * ZAID_MAX_SHARE - areas.get("pulses", 0.0)
        recs.append(Recommendation(
            rule="zaid_window",
            title=f"Plant short-duration moong on {room:.1f} ha in the summer window",
            finding="The land is idle between the wheat harvest and the paddy transplant.",
            why=("A 60-70 day moong crop fits that window, fixes its own "
                 "nitrogen, and has an MSP with procurement. Because it "
                 "pushes the paddy transplant about three weeks later, the "
                 "paddy that follows needs less irrigation — the peak pumping "
                 "period moves into the monsoon."),
            action=(f"Sow a short-duration variety (PAU 911 / ML 2056) on "
                    f"{room:.1f} ha by the first week of April."),
            impact=(f"≈₹{CROPS['pulses'].net_return_inr_ha * room:,.0f}/yr added · "
                    f"≈8 % less water on the following paddy crop"),
            priority="opportunity", confidence="medium",
            evidence=dict(room_ha=round(room, 2))))

    # ---- 6. diversification --------------------------------------------
    if out.diversity < 0.5:
        recs.append(Recommendation(
            rule="diversification",
            title="Your plan is close to a monoculture",
            finding=f"Crop diversity index is {out.diversity:.2f} of 1.0.",
            why=("A single-crop plan concentrates every risk — price, pest, "
                 "and monsoon timing — into one outcome. It also means one "
                 "buyer sets your price."),
            action="Split each season across at least two crops.",
            impact=f"Diversity {out.diversity:.2f} → up to 1.00",
            priority="medium", confidence="high",
            evidence=dict(diversity=round(out.diversity, 2))))

    # ---- 7. rainfall scenario -------------------------------------------
    if scn.rainfall in ("-10%", "-20%"):
        recs.append(Recommendation(
            rule="rainfall_scenario",
            title=f"Under a {scn.rainfall} monsoon, plan the extra pumping now",
            finding=f"Modelled on a {scn.rainfall} rainfall year.",
            why=("Less rain means both less recharge and more net irrigation "
                 "demand — the deficit hits twice."),
            action=("Hold one irrigation cycle in reserve, and prioritise the "
                    "crop with the best ₹/m³ rather than the best ₹/ha."),
            impact=f"Draft rises to {out.gw_draft_m3:,.0f} m³ in this scenario",
            priority="high" if scn.rainfall == "-20%" else "medium",
            confidence="medium",
            evidence=dict(draft_m3=round(out.gw_draft_m3))))

    # ---- 8. canal dependency -------------------------------------------
    if scn.canal == "low":
        recs.append(Recommendation(
            rule="canal_availability",
            title="Low canal availability shifts the load onto your tubewell",
            finding="Modelled with canal supply at 60 % of normal.",
            why=("Canal water is the cheapest water on the farm. When the "
                 "canal runs short the gap is made up from groundwater, which "
                 "costs electricity or diesel and deepens the drawdown."),
            action="Time the paddy transplant to the canal rotation, not to the calendar.",
            impact=f"Groundwater draft {out.gw_draft_m3:,.0f} m³ under this scenario",
            priority="medium", confidence="medium",
            evidence=dict(draft_m3=round(out.gw_draft_m3))))

    # ---- 9. what the comparison says ------------------------------------
    if out_p is not None:
        d_inc = out_p.income_inr - out.income_inr
        d_w = out_p.gw_draft_m3 - out.gw_draft_m3
        if d_w < 0:
            recs.insert(0, Recommendation(
                rule="plan_comparison",
                title="Your proposed plan, in one line",
                finding=(f"{abs(d_w)/max(out.gw_draft_m3,1e-9):.0%} less "
                         f"groundwater and {d_inc/out.income_inr:+.0%} income."),
                why="Both figures come from the same run, so they are consistent.",
                action=("Adopt it if the income change is acceptable to you — "
                        "the finance tab sizes the loan if it is not."),
                impact=(f"{d_w:,.0f} m³ water · ₹{d_inc:,.0f} income · "
                        f"{out_p.co2e_t - out.co2e_t:+.1f} t CO₂e"),
                priority="high", confidence="medium",
                evidence=dict(water_delta_m3=round(d_w),
                              income_delta_inr=round(d_inc))))

    recs.sort(key=lambda r: PRIORITY_ORDER.get(r.priority, 9))
    return recs


# ---------------------------------------------------------------------------
def advise_dataframe(recs: list[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame([dict(
        Priority=f"{PRIORITY_GLYPH.get(r.priority, '')} {r.priority.title()}",
        Recommendation=r.title,
        **{"Why": r.why, "Action": r.action, "Impact": r.impact,
           "Confidence": r.confidence}) for r in recs])


# ---------------------------------------------------------------------------
def llm_narrate(recs: list[Recommendation], lang: str = "en",
                api_key: str | None = None,
                provider: str = "openai") -> dict:
    """
    Optional language-model layer. Returns ``{"available": False, ...}`` when
    no key is configured — it never invents a response and never originates a
    recommendation. The deterministic engine above is the product; this only
    re-phrases its output for a farmer.

    Production: point ``provider`` at the vendor's chat-completions endpoint
    and pass the key via ``AQUACAST_LLM_KEY``. Nothing else changes.
    """
    import os
    key = api_key or os.environ.get("AQUACAST_LLM_KEY")
    if not key:
        return dict(available=False,
                    reason="No language-model key configured. The "
                           "recommendations above come from the deterministic "
                           "engine and are complete on their own.",
                    text=None)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        payload = "\n".join(f"- {r.title}: {r.action} ({r.impact})" for r in recs)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system",
                       "content": "You are an agricultural extension officer in "
                                  "Punjab, India. Rewrite these points in plain, "
                                  "respectful Punjabi-farmer language. Do not add "
                                  "new advice, numbers or promises."},
                      {"role": "user", "content": payload}])
        return dict(available=True, text=resp.choices[0].message.content,
                    model="gpt-4o-mini")
    except Exception as exc:                      # pragma: no cover
        return dict(available=False, reason=f"Language model unavailable: "
                                            f"{type(exc).__name__}", text=None)


# ---------------------------------------------------------------------------
def self_check() -> None:
    print("\n[advisor] AI Climate Advisor")
    print("-" * 78)
    from .farm_twin import baseline
    from .optimizer import optimize
    cur = baseline("Sangrur", 2.0)
    prop = optimize(cur, "balanced").best
    recs = recommend(cur, proposed=prop)
    for r in recs:
        print(f"  {PRIORITY_GLYPH.get(r.priority,'')} [{r.priority:<11s}] {r.title}")
        print(f"      finding : {r.finding}")
        print(f"      action  : {r.action}")
        print(f"      impact  : {r.impact}")
        print(f"      rule={r.rule} · confidence={r.confidence}")
    print()
    n = llm_narrate(recs)
    print(f"  LLM layer   : available={n['available']} — {n.get('reason','')}")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
