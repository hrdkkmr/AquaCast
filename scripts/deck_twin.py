"""
deck_twin.py — compute every Climate-Twin figure the deck quotes.

Same rule as ``deck_facts.py``: nothing is typed in. The deck reads this JSON,
so if the crop library changes, the slides change with it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.crop_library import CROPS, HA_PER_ACRE                     # noqa: E402
from src.farm_twin import baseline, evaluate                        # noqa: E402
from src.optimizer import optimize                                  # noqa: E402
from src.climate_finance import build_package, years_to_pump_failure  # noqa: E402
from src.impact import verify                                       # noqa: E402
from src.carbon import crop_carbon                                  # noqa: E402

FACTS = ROOT / "reports" / "deck_facts.json"


def pct(a, b):
    return (100.0 * (b - a) / a) if abs(a) > 1e-9 else 0.0


def main() -> None:
    base = baseline("Sangrur", 2.0)
    cur = evaluate(base)

    print("optimising (this takes ~4 s)…")
    r = optimize(base, "balanced")
    p = r.best_outcome

    pkg = build_package(base, r.best, tenure_years=5)
    rep = verify(base, r.best, draws=1200)
    e_inc = next(e for e in rep.estimates if e.metric == "Net income change")
    e_wat = next(e for e in rep.estimates if e.metric == "Groundwater saved")

    yrs_cur = years_to_pump_failure(base)
    yrs_new = years_to_pump_failure(r.best)

    facts = {
        # --- the farm -----------------------------------------------------
        "tw_hectares": 2.0,
        "tw_acres": round(2.0 / HA_PER_ACRE, 1),
        "tw_cropped_ha": round(cur.cropped_ha, 1),
        "tw_income_inr": round(cur.income_inr),
        "tw_income_per_acre": round(cur.income_inr / (2.0 / HA_PER_ACRE)),
        "tw_gw_m3": round(cur.gw_draft_m3),
        "tw_drawdown_m": round(cur.depletion_m, 2),
        "tw_co2_t": round(cur.co2e_t, 2),
        "tw_co2_biogenic_t": round(cur.co2e_biogenic_t, 2),
        "tw_residue_burnt_t": round(cur.residue_burned_t, 1),
        "tw_pump_years": round(yrs_cur, 1),

        # --- the optimiser ------------------------------------------------
        "tw_n_plans": r.n_evaluated,
        "tw_pareto": len(r.pareto),
        "tw_water_pct": round(pct(cur.gw_draft_m3, p.gw_draft_m3)),
        "tw_income_pct": round(pct(cur.income_inr, p.income_inr), 1),
        "tw_co2_pct": round(pct(cur.co2e_t, p.co2e_t)),
        "tw_residue_pct": round(pct(cur.residue_burned_t, p.residue_burned_t)),
        "tw_gw_after_m3": round(p.gw_draft_m3),
        "tw_income_after": round(p.income_inr),
        "tw_co2_after_t": round(p.co2e_t, 2),
        "tw_drawdown_after_m": round(p.depletion_m, 2),
        "tw_pump_years_after": round(yrs_new, 1),
        "tw_pump_years_gained": round(yrs_new - yrs_cur, 1),
        "tw_plan_mix": ", ".join(f"{CROPS[k].label} {v:.0%}"
                                 for k, v in r.best.allocation.items()),
        "tw_plan_irrigation": r.best.irrigation,
        "tw_plan_residue": r.best.residue,

        # --- the crop economics -------------------------------------------
        "tw_paddy_water": int(CROPS["paddy"].water_m3_ha),
        "tw_maize_water": int(CROPS["maize"].water_m3_ha),
        "tw_paddy_rupee_m3": round(CROPS["paddy"].water_productivity_inr_m3, 1),
        "tw_maize_rupee_m3": round(CROPS["maize"].water_productivity_inr_m3, 1),
        "tw_paddy_price": int(CROPS["paddy"].residue_price_inr_per_t),
        "tw_wheat_price": int(CROPS["wheat"].residue_price_inr_per_t),
        # per hectare OF PADDY, not the area-weighted rotation average
        "tw_ch4_t": round(crop_carbon("paddy", residue_path="burn",
                                      irrigation_mode="flood").rice_ch4_t_ha, 2),
        "tw_ch4_awd_t": round(crop_carbon("paddy", residue_path="burn",
                                          irrigation_mode="controlled").rice_ch4_t_ha, 2),

        # --- finance -------------------------------------------------------
        "tw_capex": round(pkg.gross_capex_inr),
        "tw_subsidy": round(pkg.subsidy_inr),
        "tw_loan": round(pkg.loan_inr),
        "tw_own": round(pkg.own_contribution_inr),
        "tw_emi": round(pkg.emi_inr),
        "tw_payback": round(pkg.payback_years, 1),
        "tw_equity_payback": round(pkg.equity_payback_years, 1),
        "tw_carbon_t": round(pkg.annual_co2e_reduction_t, 2),
        "tw_carbon_verified_t": round(pkg.verified_co2e_t, 2),
        "tw_carbon_rev": round(pkg.carbon_revenue_inr),

        # --- verification --------------------------------------------------
        "tw_mc_draws": rep.draws,
        "tw_mc_income_lo": round(e_inc.lo),
        "tw_mc_income_hi": round(e_inc.hi),
        "tw_mc_water_lo": round(e_wat.lo),
        "tw_mc_water_hi": round(e_wat.hi),
    }

    data = json.loads(FACTS.read_text()) if FACTS.exists() else {}
    data.update(facts)
    FACTS.write_text(json.dumps(data, indent=2))
    print(json.dumps(facts, indent=2))
    print(f"\n→ merged into {FACTS} ({len(data)} keys)")


if __name__ == "__main__":
    main()
