"""
Multi-objective optimiser
=========================

Finds the crop plans a farmer could actually choose between, and shows the
trade-off rather than pretending there is one right answer.

Why a Pareto front
------------------
"Should I move out of paddy?" has no single answer, because the objectives
fight. The plan that maximises income is not the plan that saves the most
water, and telling a farmer otherwise is how advisories lose credibility.
What is genuinely useful is the **efficient frontier** — the set of plans where
you cannot get more income without using more water, or save more water
without giving up income. Everything off that line is dominated: strictly worse
and therefore not worth discussing. Showing the frontier lets the farmer pick a
point on it, which is a decision, instead of being handed one.

How it searches
---------------
Every objective the twin reports is linear in planted area once the dials are
fixed — only diversity is not, and that vectorises cleanly. So the whole search
runs as a handful of matrix multiplies over ~32,000 candidate plans (crop mix ×
irrigation method × residue pathway) in well under a second, instead of calling
``evaluate()`` thirty-two thousand times.

Replacing it in production
--------------------------
This is deliberately a scored enumeration, not a solver. The interface is
``optimize(scenario, mode) -> OptimizerResult``, so swapping the search for an
OR-Tools CP-SAT / MILP formulation is a change inside this one module: the
dashboards import ``optimize`` and nothing else. A real solver would add the
constraints this prototype only approximates — machinery and labour calendars,
canal delivery (warabandi) rotations, contract offtake, and a credit constraint
on the transition year. See README §"Prototype vs Production Architecture".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .crop_library import CROPS
from .farm_twin import (FarmScenario, evaluate, linear_coefficients, risk_band,
                        SEASON_ORDER, season_of, ZAID_MAX_SHARE)
from .carbon import IRRIGATION_LABELS, RESIDUE_PATHS

CROP_KEYS = list(CROPS.keys())

OBJECTIVES = {
    "income": ("Maximise income", "💰", "Highest net income for the farm"),
    "water": ("Save the most water", "💧", "Lowest groundwater draft"),
    "climate": ("Lowest climate impact", "🌍", "Lowest net CO₂e"),
    "balanced": ("Balanced", "⚖️",
                 "Best compromise across income, water and carbon"),
}

#: Weights on the normalised objectives. Income and water are the two that
#: actually decide the argument in Punjab; carbon is the co-benefit.
WEIGHTS = {
    "income":   dict(income=1.00, water=0.05, climate=0.05),
    "water":    dict(income=0.05, water=1.00, climate=0.30),
    "climate":  dict(income=0.10, water=0.30, climate=1.00),
    "balanced": dict(income=0.45, water=0.35, climate=0.20),
}

GRID_STEP = 0.10

#: Which irrigation methods are actually available to which crop. Drip on
#: paddy is not a practice that exists at scale in Punjab — rice is either
#: flooded or grown under alternate wetting and drying — so letting the search
#: pair them would produce a plan no farmer could execute.
#: Capital cost of converting one hectare, ₹. This is the barrier the optimiser
#: keeps running into — it recommends drip on almost every plan, and drip costs
#: real money up front. Rather than hide that, the optimiser reports the
#: investment and the payback, which is precisely what hands the plan over to
#: the climate-finance module.
IRRIGATION_CAPEX_INR_PER_HA = {
    "flood": 0.0, "controlled": 9_000.0,   # laser levelling + AWD training
    "sprinkler": 28_000.0, "drip": 78_000.0,
}
RESIDUE_CAPEX_INR_PER_HA = {
    "burn": 0.0, "incorporate": 16_000.0,  # happy seeder share (often hired)
    "mulch": 6_000.0, "collect": 0.0, "bioenergy": 0.0,
}

ALLOWED_IRRIGATION = {
    "paddy": ("flood", "controlled"),
    "wheat": ("flood", "controlled", "sprinkler", "drip"),
    "maize": ("flood", "controlled", "sprinkler", "drip"),
    "cotton": ("flood", "controlled", "drip"),
    "pulses": ("flood", "controlled", "sprinkler", "drip"),
    "oilseeds": ("flood", "controlled", "sprinkler", "drip"),
}


# ---------------------------------------------------------------------------
def _compositions(n_parts: int, total_units: int) -> np.ndarray:
    """All ways to split ``total_units`` into ``n_parts`` non-negative ints."""
    if n_parts == 1:
        return np.array([[total_units]], dtype=float)
    out = []
    for i in range(total_units + 1):
        for rest in _compositions(n_parts - 1, total_units - i):
            out.append([i, *rest])
    return np.array(out, dtype=float)


def candidate_allocations(step: float = GRID_STEP) -> tuple[np.ndarray, list]:
    """
    Every feasible crop mix on the grid, as a (n_plans × n_crops) share matrix.

    Shares are normalised **within season**, and a season may be left fallow.
    Zaid is capped at half the farm because moong is a rotating catch crop, not
    a third season (see ``ZAID_MAX_SHARE``).
    """
    units = int(round(1.0 / step))
    by_season: dict[str, list] = {}
    for season in SEASON_ORDER:
        keys = [k for k in CROP_KEYS if season_of(k) == season]
        if not keys:
            continue
        blocks = [_compositions(len(keys), units)]                 # planted
        blocks.append(np.zeros((1, len(keys)), dtype=float))        # fallow
        grid = np.vstack(blocks) / float(units)
        by_season[season] = (keys, grid)

    nh = len(by_season.get("Kharif", ([], np.zeros((1, 0))))[1])
    nr = len(by_season.get("Rabi", ([], np.zeros((1, 0))))[1])
    nz = len(by_season.get("Zaid", ([], np.zeros((1, 0))))[1])

    rows = []
    for i in range(nh):
        for j in range(nr):
            for l in range(nz):
                shares = {k: 0.0 for k in CROP_KEYS}
                for season, (keys, grid) in by_season.items():
                    row = grid[i if season == "Kharif" else j if season == "Rabi" else l]
                    for idx, k in enumerate(keys):
                        shares[k] = float(row[idx])
                rows.append(shares)
    mat = np.array([[r[k] for k in CROP_KEYS] for r in rows], dtype=float)
    return mat, rows


# ---------------------------------------------------------------------------
def _score_block(mat: np.ndarray, hectares: float, coef: dict) -> dict:
    """Vectorised objectives for every row of ``mat``."""
    per_ha = coef["per_ha"]
    A = {m: np.array([per_ha[k][m] for k in CROP_KEYS], dtype=float)
         for m in ("water_m3", "gw_m3", "depletion_m", "yield_t",
                   "income_inr", "co2e_t", "residue_burned_t")}
    # Zaid cap: shares above the cap contribute nothing extra
    zaid = np.array([season_of(k) == "Zaid" for k in CROP_KEYS], dtype=float)
    shares = mat.copy()
    zsum = shares[:, zaid.astype(bool)].sum(axis=1, keepdims=True)
    over = np.where(zsum > ZAID_MAX_SHARE, ZAID_MAX_SHARE / np.maximum(zsum, 1e-9), 1.0)
    shares[:, zaid.astype(bool)] *= over

    # area in hectares per crop = share * hectares (shares already per-season)
    area = shares * hectares

    out = {m: area @ A[m] for m in A}
    out["depletion_m"] = out["depletion_m"] + coef["depletion_const_m"]
    out["cropped_ha"] = area.sum(axis=1)
    # Shannon evenness, vectorised
    tot = area.sum(axis=1, keepdims=True)
    p = area / np.maximum(tot, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.where(p > 1e-12, np.log(np.maximum(p, 1e-12)), 0.0)
    H = -(p * ln).sum(axis=1)
    n = (area > 1e-6).sum(axis=1)
    out["diversity"] = np.where(n > 1, H / np.log(np.maximum(n, 2)), 0.0)
    out["n_crops"] = n
    return out


def _risk(scores: dict, rainfall: str) -> np.ndarray:
    from .farm_twin import RAINFALL_SCENARIOS
    dev = abs(RAINFALL_SCENARIOS.get(rainfall, 1.0) - 1.0)
    gw = np.clip(scores["depletion_m"] / 3.0, 0, 1) * 40.0
    rain = np.clip(dev / 0.25, 0, 1) * 20.0
    mono = (1.0 - scores["diversity"]) * 25.0
    burnt = scores["residue_burned_t"] / np.maximum(scores["cropped_ha"], 1e-9)
    burn = np.clip(burnt / 0.45, 0, 1) * 15.0
    return np.clip(gw + rain + mono + burn, 0, 100)


# ---------------------------------------------------------------------------
@dataclass
class OptimizerResult:
    mode: str
    plans: pd.DataFrame
    best: FarmScenario
    best_outcome: object
    pareto: pd.DataFrame
    baseline_outcome: object
    n_evaluated: int
    investment_inr: float = 0.0
    payback_years: float = float("nan")

    @property
    def investment_label(self) -> str:
        return (f"₹{self.investment_inr:,.0f} one-time conversion cost")

    def headline(self) -> str:
        b, p = self.baseline_outcome, self.best_outcome
        def pct(x, y):
            return (100.0 * (y - x) / x) if abs(x) > 1e-9 else 0.0
        return (f"{pct(b.gw_draft_m3, p.gw_draft_m3):+.0f}% groundwater · "
                f"{pct(b.income_inr, p.income_inr):+.1f}% income · "
                f"{pct(b.co2e_t, p.co2e_t):+.0f}% CO₂e")


def optimize(base: FarmScenario, mode: str = "balanced",
             grid_step: float = GRID_STEP,
             irrigations: tuple = ("flood", "controlled", "drip"),
             residues: tuple = ("burn", "incorporate", "mulch", "collect"),
             top_n: int = 400) -> OptimizerResult:
    """
    Search crop mix × irrigation × residue for the best plan under ``mode``.

    Returns every evaluated plan, the winner, and the Pareto frontier of income
    against groundwater — the chart that makes the trade-off legible.
    """
    mat, share_rows = candidate_allocations(grid_step)
    frame = []
    feasible_irr = []

    # ---- feasibility ---------------------------------------------------
    # (a) an empty farm "saves" all the water and emits nothing; it is not a
    #     farm. Punjab is a double-cropped system, so require Kharif and Rabi.
    kh = [k for k in CROP_KEYS if season_of(k) == "Kharif"]
    rb = [k for k in CROP_KEYS if season_of(k) == "Rabi"]
    kharif_planted = mat[:, [CROP_KEYS.index(k) for k in kh]].sum(axis=1) > 1e-6
    rabi_planted = mat[:, [CROP_KEYS.index(k) for k in rb]].sum(axis=1) > 1e-6
    feasible = kharif_planted & rabi_planted
    mat_f = mat[feasible]
    share_rows = [r for r, f in zip(share_rows, feasible) if f]

    # (b) irrigation x crop compatibility
    mask = {}
    for irr in irrigations:
        m = np.array([irr not in ALLOWED_IRRIGATION.get(k, (irr,))
                      for k in CROP_KEYS], dtype=float)
        mask[irr] = mat_f @ m < 1e-9          # no crop planted outside its set


    for irr in irrigations:
        if mat_f.shape[0] == 0:
            continue
        for res in residues:
            scn = FarmScenario(name=base.name, district=base.district,
                               hectares=base.hectares, allocation={},
                               irrigation=irr, residue=res, canal=base.canal,
                               rainfall=base.rainfall,
                               transplant_shift_days=base.transplant_shift_days)
            coef = linear_coefficients(scn)
            s = _score_block(mat_f, base.hectares, coef)
            s["risk"] = _risk(s, base.rainfall)
            ok = mask[irr]
            for m, v in s.items():
                v = np.where(ok, v, np.nan)
                frame.append(pd.DataFrame({"irrigation": irr, "residue": res,
                                           "idx": np.arange(len(mat_f)), m: v}))
            feasible_irr.append(ok)

    plans = pd.concat(frame, ignore_index=True)
    # pivot the per-(irr,res,idx) rows back into one row per plan
    keys = ["irrigation", "residue", "idx"]
    value_cols = [c for c in plans.columns if c not in keys]
    plans = plans.groupby(keys, sort=False)[value_cols].first().reset_index()
    plans = plans.dropna(subset=["income_inr", "gw_m3"]).reset_index(drop=True)
    if plans.empty:                       # pragma: no cover - defensive
        raise RuntimeError("No feasible plan survived the constraints.")

    # ---- normalise and scalarise --------------------------------------
    def norm(col, invert=False):
        v = plans[col].to_numpy(dtype=float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        x = (v - lo) / max(hi - lo, 1e-9)
        return (1.0 - x) if invert else x

    n_income = norm("income_inr")
    n_water = norm("gw_m3", invert=True)
    n_clim = norm("co2e_t", invert=True)
    w = WEIGHTS.get(mode, WEIGHTS["balanced"])
    plans["score"] = (w["income"] * n_income + w["water"] * n_water
                      + w["climate"] * n_clim)

    plans["allocation"] = [dict((k, round(float(share_rows[int(i)][k]), 3))
                                for k in CROP_KEYS
                                if share_rows[int(i)][k] > 1e-6)
                           for i in plans["idx"]]
    plans["Irrigation"] = plans["irrigation"].map(IRRIGATION_LABELS)
    plans["Residue"] = plans["residue"].map(
        lambda r: RESIDUE_PATHS.get(r, {}).get("label", r))
    plans = plans.sort_values("score", ascending=False).reset_index(drop=True)

    # ---- Pareto frontier: maximise income, minimise groundwater --------
    pareto = _pareto(plans)

    # ---- rebuild the winning scenario ---------------------------------
    top = plans.iloc[0]
    best = FarmScenario(
        name=OBJECTIVES.get(mode, ("Best plan",))[0], district=base.district,
        hectares=base.hectares, allocation=dict(top["allocation"]),
        irrigation=str(top["irrigation"]), residue=str(top["residue"]),
        canal=base.canal, rainfall=base.rainfall,
        transplant_shift_days=base.transplant_shift_days)
    base_scn = FarmScenario(name="Current farm", district=base.district,
                            hectares=base.hectares,
                            allocation=dict(base.allocation),
                            irrigation=base.irrigation, residue=base.residue,
                            canal=base.canal, rainfall=base.rainfall,
                            transplant_shift_days=base.transplant_shift_days)

    best_out = evaluate(best)
    base_out = evaluate(base_scn)
    invested = (IRRIGATION_CAPEX_INR_PER_HA.get(str(top["irrigation"]), 0.0)
                * base.hectares
                + RESIDUE_CAPEX_INR_PER_HA.get(str(top["residue"]), 0.0)
                * base.hectares)
    gain = best_out.income_inr - base_out.income_inr
    payback = (invested / gain) if gain > 1e-6 else float("nan")
    return OptimizerResult(mode=mode, plans=plans.head(top_n).copy(),
                           best=best, best_outcome=best_out, pareto=pareto,
                           baseline_outcome=base_out,
                           n_evaluated=int(len(plans)),
                           investment_inr=round(invested),
                           payback_years=payback)


def _pareto(plans: pd.DataFrame, max_points: int = 300) -> pd.DataFrame:
    """
    Efficient frontier on (maximise income, minimise groundwater).

    Scans income descending and keeps any plan that beats the best water figure
    seen so far — an O(n log n) sweep that returns the true non-dominated set
    without the O(n²) pairwise comparison.
    """
    d = plans[["income_inr", "gw_m3", "co2e_t", "risk", "diversity",
               "Irrigation", "Residue", "allocation", "irrigation", "residue"]
              ].copy()
    d = d.sort_values(["income_inr", "gw_m3"], ascending=[False, True])
    best_water = np.inf
    keep = []
    for i, row in enumerate(d.itertuples(index=False)):
        if row.gw_m3 < best_water - 1e-9:
            best_water = row.gw_m3
            keep.append(i)
            if len(keep) >= max_points:
                break
    return d.iloc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
def self_check() -> None:
    print("\n[optimizer] multi-objective search")
    print("-" * 78)
    from .farm_twin import baseline
    base = baseline("Sangrur", 2.0)
    for mode in OBJECTIVES:
        r = optimize(base, mode)
        b, p = r.baseline_outcome, r.best_outcome
        alloc = ", ".join(f"{CROPS[k].label} {v:.0%}"
                          for k, v in r.best.allocation.items())
        print(f"  {OBJECTIVES[mode][1]} {OBJECTIVES[mode][0]:<22s} "
              f"{r.n_evaluated:6d} plans evaluated")
        print(f"      mix      : {alloc}")
        print(f"      method   : {IRRIGATION_LABELS[r.best.irrigation]}, "
              f"{RESIDUE_PATHS[r.best.residue]['label']}")
        print(f"      income   : ₹{b.income_inr:,.0f} -> ₹{p.income_inr:,.0f}")
        print(f"      water    : {b.gw_draft_m3:,.0f} -> {p.gw_draft_m3:,.0f} m³")
        print(f"      CO₂e     : {b.co2e_t:.2f} -> {p.co2e_t:.2f} t")
        print(f"      risk     : {b.climate_risk:.0f} -> {p.climate_risk:.0f}")
        print(f"      {r.headline()}")
        print(f"      capex    : {r.investment_label}, payback "
              f"{r.payback_years:.1f} yr" if np.isfinite(r.payback_years)
              else f"      capex    : {r.investment_label}")
        print()
    r = optimize(base, "balanced")
    print(f"  Pareto frontier: {len(r.pareto)} non-dominated plans of "
          f"{r.n_evaluated}")
    print(f"  income range on frontier: ₹{r.pareto.income_inr.min():,.0f} – "
          f"₹{r.pareto.income_inr.max():,.0f}")
    print(f"  water range on frontier : {r.pareto.gw_m3.min():,.0f} – "
          f"{r.pareto.gw_m3.max():,.0f} m³")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
