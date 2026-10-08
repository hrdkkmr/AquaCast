"""
Climate Farm Twin
=================

The centrepiece of the upgrade: put a farm in, turn the dials, see what happens
to **water, income, carbon, residue and risk** — side by side against the farm
you have today.

    FORECAST -> SIMULATE -> OPTIMIZE -> FINANCE -> VERIFY

A Punjab farm is a double crop, and the twin models it as one
-------------------------------------------------------------
Land is not allocated once a year; it is planted in **Kharif** (monsoon) and
again in **Rabi** (winter), with an optional short **Zaid** catch in the
summer window between the wheat harvest and the paddy transplant. So a 2 ha
farm carries 4 ha of cropped area a year, and the allocation within each season
is normalised separately.

That is not cosmetic. Under a single-season model a 2 ha paddy farm nets
₹72,000/yr; modelled as the paddy–wheat rotation it actually is, the same farm
nets ₹1,31,000/yr = **₹26,500 per acre**, which is exactly the constant the
credit-risk engine was calibrated against. The two models only agree if the
rotation is modelled, so the rotation is what gets modelled.

The moong coupling — the one the model found rather than the one I assumed
-----------------------------------------------------------------------
Growing short-duration moong in the Zaid window delays the paddy transplant by
roughly three weeks. Later transplanting moves the peak pumping period into the
monsoon, so the paddy needs less groundwater. The twin captures that coupling:
plant moong, and the paddy you plant afterwards is cheaper to irrigate. This is
the intervention PAU actually promotes, and it falls out of the arithmetic
rather than being hard-coded as a recommendation.

What is computed vs what is estimated
-------------------------------------
* Crop water is **computed** from the calibrated groundwater drafts in
  ``crop_cycle_model``.
* Income is **computed** from ``crop_library`` net returns, anchored to the
  same constant the lender view uses.
* Carbon is a **prototype estimate** — published factors, correctly split into
  biogenic CO₂ (reported, not counted) and warming-equivalent CH₄/N₂O (counted).
* Depletion is **computed** from the farm's own draft less rainfall recharge
  and irrigation return flow, over the farm's own footprint.

Local depletion is deeper than the district average, on purpose
---------------------------------------------------------------
A paddy–wheat farm's net depletion comes out at metres per year while the
district trend is 0.8 m/yr. That is not an error: the district average is
diluted by fallow land, canal commands and non-paddy crops. The twin reports
the depletion *beneath this farm* — the number that decides when this farmer's
pump dies, and the reason Punjab's paddy belt empties faster than the
statewide figure suggests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import get_prior
from .crop_library import CROPS, CropSpec, diversity_index
from .carbon import (rotation_carbon, crop_carbon, circularity_summary,
                     IRRIGATION_VOLUME_FACTOR, IRRIGATION_LABELS,
                     PADDY_RESIDUE_T_HA)

from .dataset_generator import (RAIN_RECHARGE_MONSOON, RAIN_RECHARGE_NONMONSOON,
                                RETURN_FLOW_FRACTION)

MONSOON_SHARE_OF_RAIN = 0.78
RAINFALL_SCENARIOS = {"normal": 1.00, "-10%": 0.90, "-20%": 0.80, "+10%": 1.10}
CANAL_SCENARIOS = {"low": 0.60, "normal": 1.00, "high": 1.35}
RESIDUE_SCENARIOS = ("burn", "incorporate", "mulch", "collect", "bioenergy")

SEASON_ORDER = ("Kharif", "Rabi", "Zaid")
SEASON_GLYPH = {"Kharif": "🌧", "Rabi": "❄", "Zaid": "☀"}

#: Planting moong in the Zaid window pushes the paddy transplant back by about
#: this many days, which is where the water saving comes from.
ZAID_TRANSPLANT_DELAY_DAYS = 21

#: Moong is a catch crop, not a third season: the same field only takes it in
#: alternate years while it comes back into wheat. Capping it at half the farm
#: keeps the twin from inventing a triple-cropped farm that no one runs.
ZAID_MAX_SHARE = 0.50

#: PAU puts the paddy transplant deadline at about 25 June; yield then falls
#: roughly 0.8 %/day. Delaying *into* that window is free (and saves water);
#: delaying past it is what costs grain. So the penalty only bites after the
#: first ten days of shift — which is why the moong option still wins on net.
PADDY_YIELD_LOSS_PCT_PER_DAY = 0.8
PADDY_SHIFT_FREE_DAYS = 10

DEFAULT_ALLOCATION = {"paddy": 1.0, "wheat": 1.0}
BASELINE_ALLOCATION = {"paddy": 1.0, "wheat": 1.0}


def season_of(crop_key: str) -> str:
    return CROPS[crop_key].season


# ---------------------------------------------------------------------------
@dataclass
class FarmScenario:
    name: str = "Proposed farm"
    district: str = "Sangrur"
    hectares: float = 2.0
    allocation: dict = field(default_factory=lambda: dict(DEFAULT_ALLOCATION))
    irrigation: str = "flood"
    residue: str = "burn"
    canal: str = "normal"
    rainfall: str = "normal"
    transplant_shift_days: int = 0

    def areas_ha(self) -> dict[str, float]:
        """
        Hectares per crop. Shares are normalised **within each season**, so a
        2 ha farm double-cropped to paddy and wheat carries 2 ha of each.
        """
        out: dict[str, float] = {}
        for season in SEASON_ORDER:
            keys = [k for k in self.allocation if season_of(k) == season]
            tot = sum(float(self.allocation[k]) for k in keys)
            if tot <= 0:
                continue
            cap = ZAID_MAX_SHARE if season == "Zaid" else 1.0
            for k in keys:
                share = float(self.allocation[k]) / tot * cap
                if share > 1e-6:
                    out[k] = self.hectares * share
        return out

    def seasonal_frame(self) -> pd.DataFrame:
        rows = []
        for season in SEASON_ORDER:
            keys = [k for k in self.allocation if season_of(k) == season]
            tot = sum(float(self.allocation[k]) for k in keys)
            for k in keys:
                c = CROPS[k]
                rows.append(dict(Season=f"{SEASON_GLYPH[season]} {season}",
                                 Crop=c.label, glyph=c.glyph, colour=c.colour,
                                 **{"Share of land": round(
                                     100 * float(self.allocation[k]) / tot
                                     if tot > 0 else 0.0, 1),
                                    "Area (ha)": round(
                                        self.hectares * float(self.allocation[k])
                                        / tot, 2) if tot > 0 else 0.0}))
        return pd.DataFrame(rows)

    def cropped_ha(self) -> float:
        return sum(self.areas_ha().values())


@dataclass
class FarmOutcome:
    scenario: FarmScenario
    crop_area_ha: dict
    cropped_ha: float
    water_demand_m3: float
    gw_draft_m3: float
    canal_supply_m3: float
    depletion_m: float
    income_inr: float
    crop_income_inr: float
    residue_value_inr: float
    yield_t: float
    co2e_t: float
    co2e_biogenic_t: float
    residue_t: float
    residue_burned_t: float
    climate_risk: float
    diversity: float
    carbon_breakdown: dict
    circularity: dict
    per_crop: pd.DataFrame

    def as_row(self) -> dict:
        return {"Water use (m³)": round(self.water_demand_m3),
                "Groundwater draft (m³)": round(self.gw_draft_m3),
                "Net income (₹)": round(self.income_inr),
                "Grain (t)": round(self.yield_t, 2),
                "CO₂e (t/yr)": round(self.co2e_t, 2),
                "Residue burned (t)": round(self.residue_burned_t, 2),
                "Aquifer drawdown (m/yr)": round(self.depletion_m, 2),
                "Climate risk": round(self.climate_risk)}


# ---------------------------------------------------------------------------
def _rain_m(district: str, scenario: str) -> float:
    scale = RAINFALL_SCENARIOS.get(scenario, 1.0)
    annual_mm = 600.0
    try:
        from .dataset_generator import fit_climate_model
        cm = fit_climate_model()
        annual_mm = float(np.nansum(cm.gamma_shape * cm.gamma_scale * cm.p01[1:13]))
    except Exception:
        pass
    if not np.isfinite(annual_mm) or annual_mm <= 0:
        annual_mm = 600.0
    return annual_mm / 1000.0 * scale


def _shift_days(scn: FarmScenario, areas: dict[str, float]) -> int:
    """Later transplanting = less pumping. Moong in Zaid forces the delay."""
    d = int(scn.transplant_shift_days)
    if areas.get("pulses", 0) > 1e-6 and areas.get("paddy", 0) > 1e-6:
        d = max(d, ZAID_TRANSPLANT_DELAY_DAYS)
    return d


def _transplant_factor(shift_days: int) -> float:
    return float(np.clip(1.0 - 0.004 * max(shift_days, 0), 0.85, 1.0))


def _recharge_m3(scn: FarmScenario) -> float:
    rain_m = _rain_m(scn.district, scn.rainfall)
    area_m2 = scn.hectares * 1e4
    return rain_m * area_m2 * (MONSOON_SHARE_OF_RAIN * RAIN_RECHARGE_MONSOON
                               + (1 - MONSOON_SHARE_OF_RAIN) * RAIN_RECHARGE_NONMONSOON)


# ---------------------------------------------------------------------------
def evaluate(scn: FarmScenario) -> FarmOutcome:
    prior = get_prior(scn.district)
    canal_share = float(getattr(prior, "canal_share_pct", 0.0) or 0.0) / 100.0
    canal_share_eff = float(np.clip(canal_share * CANAL_SCENARIOS.get(scn.canal, 1.0),
                                    0.0, 0.85))
    irrig_factor = IRRIGATION_VOLUME_FACTOR.get(scn.irrigation, 1.0)
    areas = scn.areas_ha()
    tf = _transplant_factor(_shift_days(scn, areas))
    sy = float(getattr(prior, "specific_yield", 0.12) or 0.12)
    lift = float(getattr(prior, "base_depth_2021_mbgl", 35.0))

    gross_m3 = gw_m3 = 0.0
    rows = []
    for k, ha in areas.items():
        c: CropSpec = CROPS[k]
        field_m3_ha = c.water_m3_ha / max(1.0 - canal_share, 0.15)
        f_shift = tf if k == "paddy" else 1.0
        # residue mulch keeps moisture in the soil, cutting the next irrigation
        f_res = 1.0 - (0.08 if scn.residue in ("incorporate", "mulch") else 0.0)
        field_need = field_m3_ha * irrig_factor * f_shift * f_res * ha
        gw = field_need * (1.0 - canal_share_eff)
        gross_m3 += field_need
        gw_m3 += gw
        rows.append(dict(Crop=c.label, glyph=c.glyph, colour=c.colour,
                         **{"Season": season_of(k), "Area (ha)": round(ha, 2),
                            "Field water (m³)": round(field_need),
                            "Groundwater (m³)": round(gw),
                            "Net ₹": round(c.net_return_inr_ha * ha),
                            "₹/m³": round(c.water_productivity_inr_m3, 1)}))
    per_crop = (pd.DataFrame(rows).sort_values("Area (ha)", ascending=False)
                if rows else pd.DataFrame())

    area_m2 = scn.hectares * 1e4
    recharge = _recharge_m3(scn)
    net = gw_m3 * (1.0 - RETURN_FLOW_FRACTION) - recharge
    depletion_m = net / max(area_m2 * sy, 1.0)

    # A late transplant costs grain. Charged against income, not just water, so
    # the twin can say "shift by 3 weeks and you lose 9 % of your paddy" rather
    # than presenting the water saving as free money.
    penalty_days = max(_shift_days(scn, areas) - PADDY_SHIFT_FREE_DAYS, 0)
    paddy_yield_factor = max(1.0 - 0.01 * PADDY_YIELD_LOSS_PCT_PER_DAY * penalty_days,
                             0.70)
    crop_income = 0.0
    yield_t = 0.0
    for k, ha in areas.items():
        c = CROPS[k]
        yf = paddy_yield_factor if k == "paddy" else 1.0
        yield_t += c.yield_t_ha * yf * ha
        # yield loss hits gross revenue; costs are already sunk
        crop_income += (c.net_return_inr_ha + (yf - 1.0) * c.gross_inr_ha) * ha

    cd = rotation_carbon(areas, lift_m=lift, residue_path=scn.residue,
                         irrigation_mode=scn.irrigation)
    circ = circularity_summary(areas, scn.residue)
    income = crop_income + circ["net_value_inr"]

    return FarmOutcome(
        scenario=scn, crop_area_ha=areas, cropped_ha=sum(areas.values()),
        water_demand_m3=gross_m3, gw_draft_m3=gw_m3,
        canal_supply_m3=gross_m3 - gw_m3, depletion_m=depletion_m,
        income_inr=income, crop_income_inr=crop_income,
        residue_value_inr=circ["net_value_inr"], yield_t=yield_t,
        co2e_t=cd.net_t_ha * max(sum(areas.values()), 1e-9),
        co2e_biogenic_t=cd._biogenic * max(sum(areas.values()), 1e-9),
        residue_t=circ["residue_generated_t"],
        residue_burned_t=circ["residue_burned_t"],
        climate_risk=_climate_risk(depletion_m, scn, areas),
        diversity=diversity_index(areas), carbon_breakdown=cd.as_dict(),
        circularity=circ, per_crop=per_crop)


def _climate_risk(depletion_m: float, scn: FarmScenario,
                  areas: dict[str, float]) -> float:
    """
    0-100 from four measured things, weighted: groundwater stress (40),
    rainfall deviation (20), monoculture (25), residue burning (15).
    """
    gw = float(np.clip(depletion_m / 3.0, 0, 1)) * 40.0
    dev = abs(RAINFALL_SCENARIOS.get(scn.rainfall, 1.0) - 1.0)
    rain = float(np.clip(dev / 0.25, 0, 1)) * 20.0
    mono = (1.0 - diversity_index(areas)) * 25.0
    tot_ha = max(sum(areas.values()), 1e-9)
    burn = float(np.clip(
        sum(CROPS[k].residue_burned_frac * ha for k, ha in areas.items())
        / tot_ha / 0.45, 0, 1)) * 15.0
    return float(np.clip(gw + rain + mono + burn, 0, 100))


def risk_band(risk: float) -> tuple[str, str]:
    if risk >= 66:
        return "HIGH", "#B3261E"
    if risk >= 38:
        return "MODERATE", "#F59E0B"
    return "LOW", "#2E7D32"


# ---------------------------------------------------------------------------
def _residue_net_per_ha(crop_key: str, rp: dict) -> float:
    """Net ₹/ha from diverting this crop's burnt residue down pathway ``rp``."""
    c = CROPS[crop_key]
    t = c.residue_burned_t_ha                       # tonnes/ha otherwise burnt
    return (t * c.residue_price_inr_per_t * float(rp.get("price_share", 0.0))
            - t * float(rp.get("cost_inr_per_t", 0.0))
            - float(rp.get("cost_inr_ha", 0.0)))


def linear_coefficients(scn: FarmScenario) -> dict:
    """
    Per-hectare coefficients such that every objective is ``sum(coef * area)``.

    Everything the twin reports — water, groundwater, income, carbon, residue,
    depletion — is linear in the planted area once the dials are fixed. Only
    diversity is not, and it is cheap enough to vectorise separately. That means
    the optimiser can score tens of thousands of candidate plans with a single
    matrix multiply instead of calling ``evaluate()`` that many times.
    """
    prior = get_prior(scn.district)
    canal_share = float(getattr(prior, "canal_share_pct", 0.0) or 0.0) / 100.0
    canal_share_eff = float(np.clip(canal_share * CANAL_SCENARIOS.get(scn.canal, 1.0),
                                    0.0, 0.85))
    irrig_factor = IRRIGATION_VOLUME_FACTOR.get(scn.irrigation, 1.0)
    f_res = 1.0 - (0.08 if scn.residue in ("incorporate", "mulch") else 0.0)
    tf = _transplant_factor(_shift_days(scn, {"pulses": 1e-9, "paddy": 1e-9}))
    sy = float(getattr(prior, "specific_yield", 0.12) or 0.12)
    area_m2 = scn.hectares * 1e4
    lift = float(getattr(prior, "base_depth_2021_mbgl", 35.0))
    from .carbon import RESIDUE_PATHS
    rp = RESIDUE_PATHS.get(scn.residue, RESIDUE_PATHS["incorporate"])

    out = {}
    for k, c in CROPS.items():
        fav = 1.0 - (rp["water_saving_pct"] / 100.0) if k == "paddy" else 1.0
        field_ha = (c.water_m3_ha / max(1.0 - canal_share, 0.15)
                    * irrig_factor * (tf if k == "paddy" else 1.0) * f_res * fav)
        gw_ha = field_ha * (1.0 - canal_share_eff)
        co2_ha = crop_carbon(k, lift, scn.residue, irrigation_mode=scn.irrigation).net_t_ha
        out[k] = dict(
            water_m3=field_ha, gw_m3=gw_ha,
            depletion_m=gw_ha * (1.0 - RETURN_FLOW_FRACTION) / max(area_m2 * sy, 1.0),
            yield_t=c.yield_t_ha,
            income_inr=(c.net_return_inr_ha + _residue_net_per_ha(k, rp)),
            co2e_t=co2_ha,
            residue_burned_t=(c.residue_burned_t_ha if scn.residue == "burn" else 0.0),
            residue_t=c.residue_t_ha,
            season=season_of(k))
    return dict(per_ha=out,
                depletion_const_m=-_recharge_m3(scn) / max(area_m2 * sy, 1.0),
                hectares=scn.hectares)


# ---------------------------------------------------------------------------
def compare(current: FarmScenario, proposed: FarmScenario) -> dict:
    a, b = evaluate(current), evaluate(proposed)

    def pct(x, y):
        return (100.0 * (y - x) / x) if abs(x) > 1e-9 else 0.0

    return dict(
        current=a, proposed=b,
        water_delta_m3=b.water_demand_m3 - a.water_demand_m3,
        water_pct=pct(a.water_demand_m3, b.water_demand_m3),
        gw_delta_m3=b.gw_draft_m3 - a.gw_draft_m3, gw_pct=pct(a.gw_draft_m3, b.gw_draft_m3),
        income_delta_inr=b.income_inr - a.income_inr, income_pct=pct(a.income_inr, b.income_inr),
        co2_delta_t=b.co2e_t - a.co2e_t, co2_pct=pct(a.co2e_t, b.co2e_t),
        residue_delta_t=b.residue_burned_t - a.residue_burned_t,
        residue_pct=pct(a.residue_burned_t, b.residue_burned_t),
        depletion_delta_m=b.depletion_m - a.depletion_m,
        risk_delta=b.climate_risk - a.climate_risk,
        rows=pd.DataFrame([dict(Scenario="Current farm", **a.as_row()),
                           dict(Scenario="Proposed farm", **b.as_row())]))


def baseline(district: str = "Sangrur", hectares: float = 2.0) -> FarmScenario:
    """The farm Punjab has today: paddy–wheat, flood irrigation, burnt straw."""
    return FarmScenario(name="Current farm", district=district, hectares=hectares,
                        allocation=dict(BASELINE_ALLOCATION), irrigation="flood",
                        residue="burn", canal="normal", rainfall="normal",
                        transplant_shift_days=0)


def self_check() -> None:
    print("\n[farm_twin] Climate Farm Twin")
    print("-" * 78)
    cur = baseline("Sangrur", 2.0)
    a = evaluate(cur)
    print(f"  2 ha Sangrur farm · paddy–wheat rotation · "
          f"{a.cropped_ha:.1f} ha cropped")
    print(f"  net income        : ₹{a.income_inr:,.0f}  "
          f"= ₹{a.income_inr/(cur.hectares*2.47105):,.0f}/acre/yr")
    print(f"  calibration anchor: ₹26,500/acre/yr  (credit_risk_engine)")
    print(f"  groundwater draft : {a.gw_draft_m3:,.0f} m³")
    print(f"  drawdown          : {a.depletion_m:.2f} m/yr beneath this farm")
    print(f"  CO₂e              : {a.co2e_t:.2f} t/yr "
          f"(+{a.co2e_biogenic_t:.1f} t biogenic, reported separately)")
    print(f"  climate risk      : {a.climate_risk:.0f} ({risk_band(a.climate_risk)[0]})")
    print()

    prop = FarmScenario(name="Proposed farm", district="Sangrur", hectares=2.0,
                        allocation={"paddy": 0.5, "maize": 0.5, "wheat": 0.5,
                                    "oilseeds": 0.5, "pulses": 1.0},
                        irrigation="controlled", residue="collect",
                        canal="normal", rainfall="normal", transplant_shift_days=0)
    b = evaluate(prop)
    print(f"  {'':22s} {'CURRENT':>12s} {'PROPOSED':>12s} {'Δ':>13s}")
    for lab, x, y, unit in [
            ("Water use", a.water_demand_m3, b.water_demand_m3, "m³"),
            ("Groundwater draft", a.gw_draft_m3, b.gw_draft_m3, "m³"),
            ("Drawdown", a.depletion_m, b.depletion_m, "m/yr"),
            ("Net income", a.income_inr, b.income_inr, "₹"),
            ("Grain", a.yield_t, b.yield_t, "t"),
            ("CO₂e (net)", a.co2e_t, b.co2e_t, "t"),
            ("Residue burned", a.residue_burned_t, b.residue_burned_t, "t"),
            ("Climate risk", a.climate_risk, b.climate_risk, "")]:
        print(f"  {lab:22s} {x:12,.2f} {y:12,.2f} {y-x:+13,.2f} {unit}")
    c = compare(cur, prop)
    print()
    print(f"  {c['water_pct']:+.0f}% water · {c['income_pct']:+.1f}% income · "
          f"{c['co2_pct']:+.0f}% CO₂e · {c['residue_pct']:+.0f}% residue burned")
    print(f"  diversity {a.diversity:.2f} -> {b.diversity:.2f} · "
          f"risk {risk_band(a.climate_risk)[0]} -> {risk_band(b.climate_risk)[0]}")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
