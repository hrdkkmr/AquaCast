"""
Impact verification (MRV)
=========================

The last step of the chain — FORECAST -> SIMULATE -> OPTIMIZE -> FINANCE ->
**VERIFY** — and the one most climate platforms hand-wave.

Why uncertainty is shown instead of a single number
---------------------------------------------------
Every figure in this platform is a model output. Reporting "you will save
11,000 m³" invites a farmer to plan on it and a lender to price it, when the
honest statement is a range built from how wrong the inputs could be. This
module propagates input uncertainty through the whole twin by Monte Carlo and
reports a central estimate with a 90 % interval.

Uncertainty assumptions (stated, not hidden)
--------------------------------------------
============================  ==================  =======================
Parameter                     Spread (±)          Why
============================  ==================  =======================
Net return per crop           15 %                Price and yield variance
Crop water requirement        12 %                Seasonal ET variance
Emission factors              20 %                IPCC factor ranges
Residue price                 30 %                Thin, local markets
Specific yield                15 %                Aquifer heterogeneity
============================  ==================  =======================

These are judgement calls, and they are printed next to every result so they
can be argued with.

What this is not
----------------
This is not a validated carbon credit, and nothing here registers, issues, or
sells one. Real MRV under a registered methodology needs a baseline
determination, a validation body, meter- or remote-sensing-based monitoring,
and a verification period. The module states exactly which of those this
prototype performs (the modelling) and which it does not (all of the rest).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .crop_library import CROPS
from .farm_twin import FarmScenario, evaluate

UNCERTAINTY = {
    "net_return": 0.15,
    "water": 0.12,
    "emissions": 0.20,
    "residue_price": 0.30,
    "specific_yield": 0.15,
}

#: Which of the real MRV steps this prototype performs. The honest answer is
#: one of five, and it is shown in the UI rather than buried here.
MRV_STEPS = [
    dict(step="Baseline determination", status="modelled",
         detail="Computed from the calibrated crop library and the farm's own "
                "stated plan — not from measured historical input data."),
    dict(step="Additionality assessment", status="partial",
         detail="Compared against the paddy–wheat baseline, which is the "
                "regional default. A real assessment needs a counterfactual "
                "specific to this holding."),
    dict(step="Monitoring", status="not performed",
         detail="Production: flow meters on the tubewell (or a metered feeder "
                "for electric pumps), plus Sentinel-2 for residue and area."),
    dict(step="Third-party validation", status="not performed",
         detail="Requires an accredited VVB under a registered methodology."),
    dict(step="Issuance / registry", status="not performed",
         detail="No credit is issued, registered or sold anywhere in this "
                "prototype. Figures are estimates for planning only."),
]


@dataclass
class ImpactEstimate:
    metric: str
    unit: str
    central: float
    lo: float
    hi: float
    modelled: float

    def label(self, fmt: str = "{:,.0f}") -> str:
        return f"{fmt.format(self.central)} ({fmt.format(self.lo)} – {fmt.format(self.hi)})"


@dataclass
class ImpactReport:
    estimates: list
    draws: int
    verified_share: float
    mrv: list = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(
            Metric=e.metric, **{"Central estimate": round(e.central, 2),
                                "90% range": f"{e.lo:,.0f} – {e.hi:,.0f}",
                                "Unit": e.unit,
                                "Modelled (no uncertainty)": round(e.modelled, 2)})
            for e in self.estimates])


# ---------------------------------------------------------------------------
def _draw_shocks(rng, crop_keys) -> dict:
    """
    One set of perturbations shared by BOTH plans in a draw.

    The same rupee-per-quintal error applies to paddy whether the plan grows it
    or drops it, so drawing independently per plan would inflate the variance
    of the difference and make every conclusion look less certain than it is.
    """
    sd = lambda u: float(rng.normal(1.0, u / 1.645))
    return {
        "net": {k: sd(UNCERTAINTY["net_return"]) for k in crop_keys},
        "water": {k: sd(UNCERTAINTY["water"]) for k in crop_keys},
        "emissions": sd(UNCERTAINTY["emissions"]),
        "residue_price": sd(UNCERTAINTY["residue_price"]),
        "sy": sd(UNCERTAINTY["specific_yield"]),
    }


def _perturbed_outcome(scn: FarmScenario, shocks: dict) -> dict:
    """
    Re-run the twin's arithmetic with one set of perturbed inputs.

    Note the irrigation factors are carried over from the unperturbed run as a
    single ratio, so the perturbation applies to crop water *requirements* and
    not to the method's efficiency — those are different uncertainties and
    only the first is being sampled here.
    """
    from .config import get_prior
    from .carbon import crop_carbon, residue_economics
    import src.farm_twin as ft

    prior = get_prior(scn.district)
    base = evaluate(scn)
    areas = scn.areas_ha()
    lift = float(getattr(prior, "base_depth_2021_mbgl", 35.0))

    # --- water: perturb per-crop requirement, keep the method's factor -----
    raw = sum(CROPS[k].water_m3_ha * ha for k, ha in areas.items())
    method_factor = (base.gw_draft_m3 / raw) if raw > 1e-9 else 1.0
    raw_p = sum(CROPS[k].water_m3_ha * shocks["water"].get(k, 1.0) * ha
                for k, ha in areas.items())
    gw = raw_p * method_factor

    # --- income -----------------------------------------------------------
    income = sum(CROPS[k].net_return_inr_ha * shocks["net"].get(k, 1.0) * ha
                 for k, ha in areas.items())
    econ = residue_economics(areas, scn.residue)
    income += econ["value_inr"] * shocks["residue_price"] - econ["cost_inr"]

    # --- carbon -----------------------------------------------------------
    co2 = sum(crop_carbon(k, lift_m=lift, residue_path=scn.residue,
                          irrigation_mode=scn.irrigation).net_t_ha * ha
              for k, ha in areas.items()) * shocks["emissions"]

    # --- depletion --------------------------------------------------------
    sy = max(float(getattr(prior, "specific_yield", 0.12)) * shocks["sy"], 0.02)
    recharge = ft._recharge_m3(scn)
    from .dataset_generator import RETURN_FLOW_FRACTION
    depletion = (gw * (1.0 - RETURN_FLOW_FRACTION) - recharge) / max(
        scn.hectares * 1e4 * sy, 1.0)

    return dict(income=income, water=raw_p, gw=gw, co2=co2, depletion=depletion)


def verify(current: FarmScenario, proposed: FarmScenario,
           draws: int = 2000, seed: int = 11,
           verification_haircut: float = 0.55) -> ImpactReport:
    """
    Monte Carlo the difference between two plans.

    Both plans are perturbed with the *same* draw where the parameter is shared,
    so correlated errors cancel rather than double-counting — comparing a
    perturbed paddy farm against an unperturbed maize farm would wildly
    overstate the uncertainty in the difference.
    """
    rng = np.random.default_rng(seed)
    cur, prop = evaluate(current), evaluate(proposed)
    crop_keys = sorted(set(current.allocation) | set(proposed.allocation))

    d_income, d_water, d_co2, d_dep = [], [], [], []
    for _ in range(draws):
        shocks = _draw_shocks(rng, crop_keys)
        a = _perturbed_outcome(current, shocks)
        b = _perturbed_outcome(proposed, shocks)
        d_income.append(b["income"] - a["income"])
        d_water.append(a["gw"] - b["gw"])            # positive = saved
        d_co2.append(a["co2"] - b["co2"])            # positive = avoided
        d_dep.append(a["depletion"] - b["depletion"])

    def est(metric, unit, samples, modelled, fmt_scale=1.0):
        arr = np.asarray(samples, dtype=float) * fmt_scale
        lo, hi = np.nanpercentile(arr, [5, 95])
        return ImpactEstimate(metric, unit, float(np.nanmedian(arr)),
                              float(lo), float(hi), modelled * fmt_scale)

    estimates = [
        est("Groundwater saved", "m³/yr", d_water, cur.gw_draft_m3 - prop.gw_draft_m3),
        est("Net income change", "₹/yr", d_income, prop.income_inr - cur.income_inr),
        est("CO₂e avoided", "t/yr", d_co2, cur.co2e_t - prop.co2e_t),
        est("Drawdown avoided", "m/yr", d_dep, cur.depletion_m - prop.depletion_m),
    ]
    return ImpactReport(estimates=estimates, draws=draws,
                        verified_share=verification_haircut,
                        mrv=MRV_STEPS)


def verified_carbon(report: ImpactReport) -> dict:
    """Apply the verification haircut and state plainly what remains."""
    c = next(e for e in report.estimates if e.metric == "CO₂e avoided")
    verified = max(c.central, 0.0) * report.verified_share
    return dict(modelled_t=round(c.central, 2),
                verified_t=round(verified, 2),
                haircut=report.verified_share,
                note=f"{c.central:.2f} t CO₂e/yr modelled → {verified:.2f} t "
                     f"carried forward after a "
                     f"{int((1-report.verified_share)*100)} % haircut for "
                     f"additionality, leakage and measurement error. This is an "
                     f"estimate for planning — not an issued or validated credit.")


def self_check() -> None:
    print("\n[impact] verification (MRV)")
    print("-" * 78)
    from .farm_twin import baseline
    from .optimizer import optimize
    cur = baseline("Sangrur", 2.0)
    prop = optimize(cur, "balanced").best
    r = verify(cur, prop, draws=1500)
    print(f"  {r.draws:,} Monte Carlo draws, correlated between the two plans")
    print(r.frame().to_string(index=False))
    print()
    v = verified_carbon(r)
    print(f"  carbon: {v['modelled_t']} t modelled -> {v['verified_t']} t "
          f"after {int(v['haircut']*100)}% verification share")
    print()
    print("  MRV chain:")
    for s in MRV_STEPS:
        print(f"    {s['step']:<26s} {s['status']}")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
