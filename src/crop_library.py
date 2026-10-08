"""
Crop library — the decision layer
=================================

Everything the Crop Intelligence cards, the Climate Farm Twin, Circularity,
Carbon and Climate Finance need to know about a Punjab crop, in one place.

Calibration contract
--------------------
Net returns are anchored to ``credit_risk_engine.WHEAT_PADDY_GROSS_MARGIN_INR_PER_ACRE_YR``
so the twin and the lender view can never disagree about what a hectare earns.
Paddy + wheat net return therefore sums to that constant by construction.

Net return is quoted **over total cost** (including imputed family labour and
land rent), which is the conservative reading and the one a lender should use.
Gross-return figures quoted at MSP would be roughly twice as large and would
make every intervention look twice as attractive.

Why non-paddy crops carry a ``market_realisation`` below 1.0
------------------------------------------------------------
Paddy and wheat are effectively procured at MSP in Punjab — the farmer who grows
them gets the announced price. Maize, pulses and oilseeds nominally have an MSP
but procurement is thin, so the realised farm-gate price is materially below it.
That gap, not ignorance, is the main reason farmers stay in the paddy-wheat
rotation. Modelling it is what stops the tool from naively concluding
"just grow maize, it uses less water", which is the advice farmers have been
given for fifteen years and have declined.

Carbon and residue coefficients are literature values (IPCC 2019 refinement,
NAAS/PAU residue ratios) and every figure derived from them is labelled a
**prototype estimate** at the point of display.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .crop_cycle_model import PADDY, WHEAT, ZAID, MAIZE, COTTON

HA_PER_ACRE = 1.0 / 2.47105


def _margin_constant() -> float:
    """Net return of the wheat+paddy rotation, ₹/ha/yr — single source of truth."""
    from .credit_risk_engine import (WHEAT_PADDY_GROSS_MARGIN_INR_PER_ACRE_YR,
                                     TYPICAL_OPERATED_ACRES)
    return float(WHEAT_PADDY_GROSS_MARGIN_INR_PER_ACRE_YR) / HA_PER_ACRE


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CropSpec:
    key: str
    label: str
    season: str                 # Rabi | Zaid | Kharif | Annual
    gw_draft_mm: float          # seasonal groundwater draft (calibrated)
    duration_days: int
    yield_t_ha: float           # Punjab average
    msp_inr_per_t: float        # MSP 2025-26, ₹/tonne
    market_realisation: float   # share of MSP actually realised at the farm gate
    net_return_inr_ha: float    # over total cost — see module docstring
    residue_t_ha: float         # dry residue generated
    residue_burned_frac: float  # share typically burnt in Punjab
    n_kg_ha: float              # fertiliser nitrogen applied
    diesel_l_ha: float          # machinery + threshing
    colour: str
    glyph: str                  # emoji used as a lightweight crop asset
    notes: str = ""
    #: Farm-gate price of baled residue, ₹/tonne. This is the whole reason
    #: paddy straw is burnt and wheat straw is not: paddy straw is poor fodder
    #: and trades near ₹900/t, while wheat bhusa is prime dry fodder worth
    #: several times that. A flat per-hectare residue value would have hidden
    #: the single most important fact about Punjab's burning problem.
    residue_price_inr_per_t: float = 0.0

    # ---- derived -------------------------------------------------------
    @property
    def water_m3_ha(self) -> float:
        """Groundwater volume per hectare: 1 mm over 1 ha = 10 m³."""
        return self.gw_draft_mm * 10.0

    @property
    def water_productivity_inr_m3(self) -> float:
        """Net rupees per cubic metre of groundwater — the metric that decides."""
        return self.net_return_inr_ha / max(self.water_m3_ha, 1.0)

    @property
    def net_inr_acre(self) -> float:
        return self.net_return_inr_ha * HA_PER_ACRE

    @property
    def residue_burned_t_ha(self) -> float:
        return self.residue_t_ha * self.residue_burned_frac

    @property
    def gross_inr_ha(self) -> float:
        return self.yield_t_ha * self.msp_inr_per_t * self.market_realisation


# ---------------------------------------------------------------------------
# Calibrated so PADDY + WHEAT == _margin_constant()
# ---------------------------------------------------------------------------
def _build() -> dict[str, CropSpec]:
    M = _margin_constant()
    paddy_net = M * 0.55
    wheat_net = M * 0.45

    specs = [
        CropSpec("paddy", "Paddy", "Kharif", gw_draft_mm=550.0, duration_days=128,
                 yield_t_ha=6.8, msp_inr_per_t=23_200.0, market_realisation=1.00,
                 net_return_inr_ha=paddy_net, residue_t_ha=7.5,
                 residue_price_inr_per_t=900.0, residue_burned_frac=0.48, n_kg_ha=150.0, diesel_l_ha=95.0,
                 colour="#2E7D5B", glyph="🌾",
                 notes="Assured procurement, but the single largest driver of "
                       "Punjab's groundwater deficit and of straw burning."),

        CropSpec("wheat", "Wheat", "Rabi", gw_draft_mm=350.0, duration_days=160,
                 yield_t_ha=5.1, msp_inr_per_t=24_250.0, market_realisation=1.00,
                 net_return_inr_ha=wheat_net, residue_t_ha=6.0,
                 residue_price_inr_per_t=3_200.0, residue_burned_frac=0.12, n_kg_ha=140.0, diesel_l_ha=70.0,
                 colour="#C9A227", glyph="🌿",
                 notes="Sown into an aquifer the monsoon has only partly "
                       "refilled. Straw is mostly baled, not burnt."),

        CropSpec("maize", "Maize", "Kharif", gw_draft_mm=209.0, duration_days=93,
                 yield_t_ha=5.6, msp_inr_per_t=22_250.0, market_realisation=0.86,
                 net_return_inr_ha=paddy_net * 0.94, residue_t_ha=6.2,
                 residue_price_inr_per_t=1_600.0, residue_burned_frac=0.15, n_kg_ha=120.0, diesel_l_ha=80.0,
                 colour="#8E6E9E", glyph="🌽",
                 notes="62% less groundwater than paddy for ~6% less income — "
                       "the strongest diversification candidate on water "
                       "productivity. Held back by thin procurement."),

        CropSpec("cotton", "Cotton", "Kharif", gw_draft_mm=262.0, duration_days=170,
                 yield_t_ha=2.4, msp_inr_per_t=77_100.0, market_realisation=0.92,
                 net_return_inr_ha=paddy_net * 1.05, residue_t_ha=4.8,
                 residue_price_inr_per_t=700.0, residue_burned_frac=0.30, n_kg_ha=105.0, diesel_l_ha=110.0,
                 colour="#B0A8B9", glyph="🌱",
                 notes="Highest net return per hectare, but a long season, "
                       "pest risk and price volatility suited only to the "
                       "south-west districts."),

        CropSpec("pulses", "Moong / pulses", "Zaid", gw_draft_mm=311.0,
                 duration_days=70, yield_t_ha=1.1, msp_inr_per_t=86_820.0,
                 market_realisation=0.80, net_return_inr_ha=paddy_net * 0.50,
                 residue_t_ha=2.4, residue_price_inr_per_t=1_400.0, residue_burned_frac=0.05, n_kg_ha=20.0,
                 diesel_l_ha=45.0, colour="#4C9A2A", glyph="🫘",
                 notes="Fits the empty Apr-Jun window and fixes nitrogen, but "
                       "the short season and low absolute return keep it a "
                       "minority choice."),

        CropSpec("oilseeds", "Mustard / sunflower", "Rabi", gw_draft_mm=198.0,
                 duration_days=140, yield_t_ha=1.6, msp_inr_per_t=59_500.0,
                 market_realisation=0.85, net_return_inr_ha=wheat_net * 0.72,
                 residue_t_ha=3.2, residue_price_inr_per_t=1_700.0, residue_burned_frac=0.08, n_kg_ha=60.0,
                 diesel_l_ha=55.0, colour="#D9A441", glyph="🌻",
                 notes="A Rabi wheat alternative using ~43% less groundwater; "
                       "the lowest-risk way to take land out of wheat."),
    ]
    return {c.key: c for c in specs}


CROPS: dict[str, CropSpec] = _build()
KHARIF_CROPS = ("paddy", "maize", "cotton")
RABI_CROPS = ("wheat", "oilseeds")
ZAID_CROPS = ("pulses",)


# ---------------------------------------------------------------------------
def crop_frame() -> pd.DataFrame:
    """Tidy comparison table — powers the Crop Intelligence cards and compare."""
    rows = []
    for c in CROPS.values():
        rows.append(dict(
            key=c.key, Crop=c.label, Season=c.season,
            **{"Duration (d)": c.duration_days,
               "Yield (t/ha)": c.yield_t_ha,
               "GW draft (mm)": c.gw_draft_mm,
               "Water (m³/ha)": round(c.water_m3_ha),
               "Net ₹/ha": round(c.net_return_inr_ha),
               "Net ₹/acre": round(c.net_inr_acre),
               "₹/m³ water": round(c.water_productivity_inr_m3, 1),
               "Residue (t/ha)": c.residue_t_ha,
               "Burned (t/ha)": round(c.residue_burned_t_ha, 2),
               "N (kg/ha)": c.n_kg_ha,
               "MSP ₹/t": round(c.msp_inr_per_t),
               "Price realisation": f"{c.market_realisation:.0%}",
               "colour": c.colour, "glyph": c.glyph, "notes": c.notes}))
    return pd.DataFrame(rows).sort_values("₹/m³ water", ascending=False)


def pressure_label(gw_mm: float) -> tuple[str, str]:
    """Groundwater pressure band for a crop's seasonal draft."""
    if gw_mm >= 450:
        return "VERY HIGH", "#B3261E"
    if gw_mm >= 300:
        return "HIGH", "#EF4444"
    if gw_mm >= 200:
        return "MODERATE", "#F59E0B"
    if gw_mm >= 120:
        return "LOW", "#22C55E"
    return "VERY LOW", "#15803D"


def compare_crops(keys: list[str]) -> pd.DataFrame:
    df = crop_frame()
    return df[df["key"].isin(keys)].copy()


def best_by_water_productivity(season: str | None = None) -> CropSpec:
    pool = [c for c in CROPS.values() if season is None or c.season == season]
    return max(pool, key=lambda c: c.water_productivity_inr_m3)


def diversity_index(allocation: dict[str, float]) -> float:
    """
    Shannon evenness (0-1) of a crop allocation.

    1.0 = perfectly even mix, 0 = monoculture. Monoculture is the measurable
    form of the risk Punjab carries: one crop, one pest, one price, one
    aquifer.
    """
    a = np.array([v for v in allocation.values() if v > 0], dtype=float)
    if a.size <= 1:
        return 0.0
    p = a / a.sum()
    return float(-(p * np.log(p)).sum() / np.log(a.size))


def self_check() -> None:
    M = _margin_constant()
    print("\n[crop_library] Crop decision layer")
    print("-" * 78)
    print(f"  calibration anchor       : wheat+paddy net = ₹{M:,.0f}/ha/yr "
          f"(₹{M*HA_PER_ACRE:,.0f}/acre)")
    tot = CROPS["paddy"].net_return_inr_ha + CROPS["wheat"].net_return_inr_ha
    print(f"  paddy + wheat in library : ₹{tot:,.0f}/ha  "
          f"{'(consistent)' if abs(tot - M) < 1 else '(MISMATCH)'}")
    print()
    df = crop_frame()
    cols = ["Crop", "Season", "Water (m³/ha)", "Net ₹/ha", "₹/m³ water",
            "Burned (t/ha)"]
    print(df[cols].to_string(index=False))
    print()
    p, m = CROPS["paddy"], CROPS["maize"]
    print(f"  paddy -> maize : {100*(1-m.water_m3_ha/p.water_m3_ha):.0f}% less water "
          f"for {100*(1-m.net_return_inr_ha/p.net_return_inr_ha):.1f}% less income")
    print(f"  water productivity: paddy ₹{p.water_productivity_inr_m3:.1f}/m³ vs "
          f"maize ₹{m.water_productivity_inr_m3:.1f}/m³ "
          f"({m.water_productivity_inr_m3/p.water_productivity_inr_m3:.1f}x)")
    print(f"  monoculture diversity index: {diversity_index({'paddy':1.0}):.2f} · "
          f"even 3-crop mix: {diversity_index({'paddy':.4,'maize':.3,'pulses':.3}):.2f}")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
