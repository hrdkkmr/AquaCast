"""
Carbon & circularity
====================

Two things the brief asks for that the original system never touched:

1. **CO₂e accounting** — residue burning, irrigation energy, fertiliser,
   machinery, and the soil-carbon credit from putting residue back.
2. **Circularity** — what else the residue could become, and what each pathway
   is worth in water, carbon, rupees and soil.

Every number here is a **prototype estimate** built from published emission
factors, and every surface that displays one says so. This is not a validated
carbon credit calculation and must never be presented as one.

Emission factors (documented, all in one place)
-----------------------------------------------
``GRID_KG_CO2_PER_KWH``  0.71  — India grid average, CEA CO₂ baseline database
``DIESEL_KG_CO2_PER_L``  2.68  — IPCC/DEFRA diesel combustion, CO₂e
``N_MANUFACTURE``        5.90  — kg CO₂e per kg N, urea manufacture + transport
``N2O_DIRECT``           4.29  — kg CO₂e per kg N: 1% of applied N emitted as
                                 N₂O-N (IPCC 2019 Tier 1), GWP₁₀₀ = 273
``RESIDUE_BURN_BIOGENIC``1.30  — kg CO₂ per kg dry residue burnt. Biogenic:
                                 the carbon was fixed last season, so it is
                                 reported but NOT added to the GWP total.
``RESIDUE_BURN_GWP``     0.16  — kg CO₂e per kg dry residue burnt — the part
                                 that actually warms: CH₄ (3.4 g/kg, GWP 28)
                                 plus N₂O (0.17 g/kg, GWP 265). IPCC 2019.
``RICE_CH4_FLOOD``      30.0   — kg CH₄/ha/season from a continuously flooded
                                 paddy; GWP₁₀₀ = 28. This is the single largest
                                 GHG source in Punjab paddy and omitting it
                                 would have understated the crop's footprint by
                                 roughly a quarter.
``AWD_CH4_FACTOR``       0.50  — alternate wetting and drying roughly halves it
``SOIL_C_SEQUESTRATION`` 0.15  — t C/ha/yr additional SOC from incorporation,
                                 mid-range of long-term Indian trials

Pumping energy is computed, not assumed: E = ρ g V H / η, with η the same
wire-to-water efficiency the credit engine uses, so the energy figure and the
money figure cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .crop_library import CROPS, CropSpec

# ---------------------------------------------------------------------------
GRID_KG_CO2_PER_KWH = 0.71
DIESEL_KG_CO2_PER_L = 2.68
N_MANUFACTURE_KG_CO2_PER_KG_N = 5.90
N2O_DIRECT_KG_CO2_PER_KG_N = 4.29
RESIDUE_BURN_BIOGENIC_KG_CO2_PER_KG = 1.30   # biogenic — reported, not counted
RESIDUE_BURN_GWP_KG_CO2E_PER_KG = 0.16       # CH4 + N2O only — this is counted
RICE_CH4_KG_HA_FLOOD = 30.0
CH4_GWP100 = 28.0
AWD_CH4_FACTOR = 0.50
SOIL_C_SEQUESTRATION_T_C_HA_YR = 0.15
NITROGEN_FIXATION_KG_N_HA_PULSES = 40.0      # credit for a legume in the rotation

#: Volume pumped per unit of crop water by irrigation method, relative to
#: flood. Drip and sprinkler cut the volume *and* the energy; controlled
#: (alternate wetting and drying) cuts paddy's methane as well.
IRRIGATION_VOLUME_FACTOR = {"flood": 1.00, "controlled": 0.82,
                            "sprinkler": 0.75, "drip": 0.62}
IRRIGATION_LABELS = {"flood": "Flood irrigation", "controlled": "Controlled (AWD)",
                     "sprinkler": "Sprinkler", "drip": "Drip / micro-irrigation"}

WATER_DENSITY = 1000.0
GRAVITY = 9.81
KWH_PER_JOULE = 1.0 / 3.6e6


def pumping_energy_kwh(volume_m3_ha: float, lift_m: float,
                       efficiency: float = 0.55) -> float:
    """Electrical energy to lift ``volume`` through ``lift`` at ``efficiency``."""
    e_j = WATER_DENSITY * GRAVITY * float(volume_m3_ha) * float(lift_m)
    return e_j * KWH_PER_JOULE / max(float(efficiency), 0.05)


# ---------------------------------------------------------------------------
@dataclass
class CarbonBreakdown:
    residue_t_ha: float
    rice_ch4_t_ha: float
    irrigation_t_ha: float
    fertiliser_t_ha: float
    machinery_t_ha: float
    soil_credit_t_ha: float
    legume_credit_t_ha: float

    @property
    def gross_t_ha(self) -> float:
        return (self.residue_t_ha + self.rice_ch4_t_ha + self.irrigation_t_ha
                + self.fertiliser_t_ha + self.machinery_t_ha)

    @property
    def biogenic_t_ha(self) -> float:
        """CO₂ from burning last season's stubble — climate-neutral, reported."""
        return self._biogenic

    @property
    def net_t_ha(self) -> float:
        return max(self.gross_t_ha - self.soil_credit_t_ha
                   - self.legume_credit_t_ha, 0.0)

    _biogenic: float = 0.0

    def as_dict(self) -> dict:
        return dict(residue=self.residue_t_ha, rice_ch4=self.rice_ch4_t_ha,
                    irrigation=self.irrigation_t_ha,
                    fertiliser=self.fertiliser_t_ha, machinery=self.machinery_t_ha,
                    soil_credit=-self.soil_credit_t_ha,
                    legume_credit=-self.legume_credit_t_ha,
                    gross=self.gross_t_ha, net=self.net_t_ha)


def crop_carbon(crop_key: str, lift_m: float = 40.0,
                residue_path: str = "burn",
                efficiency: float = 0.55,
                irrigation_mode: str = "flood") -> CarbonBreakdown:
    """
    Per-hectare CO₂e for one crop under one residue and irrigation pathway.

    ``residue_path``    ∈ {burn, incorporate, mulch, collect, bioenergy}
    ``irrigation_mode`` ∈ {flood, controlled, drip, sprinkler}

    Only ``burn`` emits; the others sequester or displace. Irrigation mode
    drives both the volume pumped *and* paddy's methane: flooding a rice field
    is what makes it anaerobic, so moving to controlled irrigation cuts CH₄
    about in half on top of the energy saving.
    """
    c: CropSpec = CROPS[crop_key]
    vol = c.water_m3_ha * IRRIGATION_VOLUME_FACTOR.get(irrigation_mode, 1.0)
    kwh = pumping_energy_kwh(vol, lift_m, efficiency)

    residue = 0.0
    biogenic = 0.0
    ch4 = 0.0
    soil = 0.0
    if residue_path == "burn":
        # tonnes of residue x tonnes CO2e per tonne (the factor is kg/kg, and
        # kg CO2 per kg residue == tonnes CO2 per tonne residue)
        residue = c.residue_burned_t_ha * RESIDUE_BURN_GWP_KG_CO2E_PER_KG
        biogenic = c.residue_burned_t_ha * RESIDUE_BURN_BIOGENIC_KG_CO2_PER_KG
    elif residue_path in ("incorporate", "mulch"):
        soil = (SOIL_C_SEQUESTRATION_T_C_HA_YR * 44.0 / 12.0)
    elif residue_path == "collect":
        # removed from field: no in-field emission, no soil credit here
        soil = 0.0
    elif residue_path == "bioenergy":
        # displaces coal power; credit booked on the residue actually recovered
        recovered = c.residue_t_ha * (1.0 - c.residue_burned_frac)
        soil = recovered * 0.55 * 0.5

    if crop_key == "paddy":
        f = AWD_CH4_FACTOR if irrigation_mode in ("controlled", "drip",
                                                  "sprinkler") else 1.0
        ch4 = RICE_CH4_KG_HA_FLOOD * f * CH4_GWP100 / 1000.0

    n_credit = 0.0
    if crop_key in ("pulses",):
        n_credit = (NITROGEN_FIXATION_KG_N_HA_PULSES
                    * N_MANUFACTURE_KG_CO2_PER_KG_N / 1000.0)

    return CarbonBreakdown(
        residue_t_ha=residue,
        rice_ch4_t_ha=ch4,
        irrigation_t_ha=kwh * GRID_KG_CO2_PER_KWH / 1000.0,
        fertiliser_t_ha=(c.n_kg_ha * (N_MANUFACTURE_KG_CO2_PER_KG_N
                                      + N2O_DIRECT_KG_CO2_PER_KG_N) / 1000.0),
        machinery_t_ha=c.diesel_l_ha * DIESEL_KG_CO2_PER_L / 1000.0,
        soil_credit_t_ha=soil,
        legume_credit_t_ha=n_credit,
        _biogenic=biogenic)


def rotation_carbon(allocation_ha: dict[str, float], lift_m: float = 40.0,
                    residue_path: str = "burn",
                    efficiency: float = 0.55,
                    irrigation_mode: str = "flood") -> CarbonBreakdown:
    """Area-weighted carbon for a whole rotation (dict of crop → hectares)."""
    total_ha = sum(allocation_ha.values()) or 1.0
    parts = [crop_carbon(k, lift_m, residue_path, efficiency, irrigation_mode)
             for k in allocation_ha]
    def s(attr):
        return sum(getattr(p, attr) * (allocation_ha[k] / total_ha)
                   for k, p in zip(allocation_ha, parts))
    return CarbonBreakdown(
        residue_t_ha=s("residue_t_ha"), rice_ch4_t_ha=s("rice_ch4_t_ha"),
        irrigation_t_ha=s("irrigation_t_ha"),
        fertiliser_t_ha=s("fertiliser_t_ha"), machinery_t_ha=s("machinery_t_ha"),
        soil_credit_t_ha=s("soil_credit_t_ha"),
        legume_credit_t_ha=s("legume_credit_t_ha"),
        _biogenic=s("_biogenic"))


# ---------------------------------------------------------------------------
# Circularity — residue pathways
# ---------------------------------------------------------------------------
#: Residue tonnage a paddy crop leaves, used as the reference against which
#: baling revenue is scaled. Other crops carry proportionally less straw.
PADDY_RESIDUE_T_HA = CROPS["paddy"].residue_t_ha


def residue_economics(allocation_ha: dict[str, float], pathway: str) -> dict:
    """
    Revenue, cost and net for diverting the *burnt* fraction of residue.

    The economics run on the tonnage a crop would otherwise have burnt, priced
    at that crop's own farm-gate rate. That is why the same dial is worth
    ₹6,800/ha on a paddy field and very little on a cotton field: cotton stalks
    are woody and nearly unsaleable, paddy straw at least has a market.
    """
    p = RESIDUE_PATHS.get(pathway, RESIDUE_PATHS["incorporate"])
    burnt = value = cost = 0.0
    for k, ha in allocation_ha.items():
        c = CROPS[k]
        t = c.residue_burned_t_ha * ha
        burnt += t
        value += t * c.residue_price_inr_per_t * float(p.get("price_share", 0.0))
        cost += t * float(p.get("cost_inr_per_t", 0.0))
    cost += float(p.get("cost_inr_ha", 0.0)) * (sum(allocation_ha.values()) or 1.0)
    return dict(burnt_t=burnt, value_inr=value, cost_inr=cost,
                net_value_inr=value - cost)

RESIDUE_PATHS = {
    "burn": dict(
        label="Burn in field", glyph="🔥", colour="#B3261E",
        co2_t_ha=None,          # computed
        water_saving_pct=0.0,
        cost_inr_ha=0.0, cost_inr_per_t=0.0, price_share=0.0,
        soil_impact="Nutrient loss, organic matter decline, PM2.5 episode",
        note="Fastest and cheapest for the farmer, and the reason Punjab's "
             "October air is what it is. Not a recommendation — the baseline."),
    "incorporate": dict(
        label="Incorporate into soil", glyph="🚜", colour="#7CB342",
        water_saving_pct=8.0,
        cost_inr_ha=4_500.0,     # happy seeder / rotavator pass
        cost_inr_per_t=0.0, price_share=0.0,
        soil_impact="+0.15 t C/ha/yr, better structure and water holding",
        note="Requires a Happy Seeder or super-straw management system. "
             "Saves one irrigation by keeping moisture in."),
    "mulch": dict(
        label="Surface mulch", glyph="🌾", colour="#8D6E63",
        water_saving_pct=14.0,
        cost_inr_ha=3_000.0,
        cost_inr_per_t=0.0, price_share=0.0,
        soil_impact="Lower evaporation, weed suppression, slow C build",
        note="Strongest water saving per rupee of any residue option."),
    "collect": dict(
        label="Bale & sell (biomass)", glyph="📦", colour="#0288D1",
        water_saving_pct=4.0,
        cost_inr_ha=0.0, cost_inr_per_t=400.0, price_share=1.0,
        soil_impact="Carbon removed from field — needs a nutrient top-up",
        note="Converts a disposal cost into revenue, but exports soil carbon. "
             "Works where a biomass plant or power station is within haulage."),
    "bioenergy": dict(
        label="Bioenergy / industrial", glyph="⚡", colour="#F9A825",
        water_saving_pct=4.0,
        cost_inr_ha=0.0, cost_inr_per_t=650.0, price_share=0.75,
        soil_impact="Displaces coal generation; ash returned as potash",
        note="Highest system-level climate benefit, lowest farmer margin — "
             "needs a supply chain, not just a machine."),
}


def residue_pathways(residue_t_ha: float, burned_frac: float,
                     price_inr_per_t: float = None) -> pd.DataFrame:
    """
    The alternatives to burning, priced per hectare.

    ``price_inr_per_t`` is the farm-gate rate for that crop's residue; it
    defaults to paddy straw, the crop the decision is actually about.
    """
    if price_inr_per_t is None:
        price_inr_per_t = CROPS["paddy"].residue_price_inr_per_t
    rows = []
    total = float(residue_t_ha)
    burned = total * float(burned_frac)
    recoverable = total - burned
    for key, p in RESIDUE_PATHS.items():
        co2 = (burned * RESIDUE_BURN_GWP_KG_CO2E_PER_KG) if key == "burn" else 0.0
        soil = 0.0
        if key in ("incorporate", "mulch"):
            soil = SOIL_C_SEQUESTRATION_T_C_HA_YR * 44.0 / 12.0
        elif key == "bioenergy":
            soil = recoverable * 0.55 * 0.5
        value = burned * float(p.get("price_share", 0.0)) * price_inr_per_t
        cost = (burned * float(p.get("cost_inr_per_t", 0.0))
                + float(p.get("cost_inr_ha", 0.0)))
        rows.append(dict(
            key=key, Pathway=p["label"], glyph=p["glyph"], colour=p["colour"],
            **{"CO₂e (t/ha)": round(co2 - soil, 3),
               "Water saving": f"{p['water_saving_pct']:.0f}%",
               "Cost ₹/ha": round(cost),
               "Revenue ₹/ha": round(value),
               "Net ₹/ha": round(value - cost),
               "Soil impact": p["soil_impact"],
               "note": p["note"]}))
    return pd.DataFrame(rows)


def circularity_summary(allocation_ha: dict[str, float],
                        residue_path: str = "incorporate") -> dict:
    """Residue mass balance + the value of diverting it from the field."""
    total_ha = sum(allocation_ha.values()) or 1.0
    gen = burnt = 0.0
    for k, ha in allocation_ha.items():
        c = CROPS[k]
        gen += c.residue_t_ha * ha
        burnt += c.residue_burned_t_ha * ha
    p = RESIDUE_PATHS.get(residue_path, RESIDUE_PATHS["incorporate"])
    econ = residue_economics(allocation_ha, residue_path)
    # Diverting residue means it is NOT burnt. Reporting a burnt tonnage next
    # to a "bale and sell" plan was simply wrong.
    actually_burnt = burnt if residue_path == "burn" else 0.0
    burned_path_co2 = burnt * RESIDUE_BURN_GWP_KG_CO2E_PER_KG
    biogenic_co2 = burnt * RESIDUE_BURN_BIOGENIC_KG_CO2_PER_KG
    alt_co2 = 0.0
    if residue_path in ("incorporate", "mulch"):
        alt_co2 = -SOIL_C_SEQUESTRATION_T_C_HA_YR * 44.0 / 12.0 * total_ha
    elif residue_path == "bioenergy":
        alt_co2 = -(gen - burnt) * 0.55 * 0.5
    t_ha = gen / total_ha
    return dict(
        residue_value_scale=round(float(np.clip(t_ha / PADDY_RESIDUE_T_HA, 0, 1.5)), 2),
        residue_generated_t=round(gen, 2),
        residue_burned_t=round(actually_burnt, 2),
        residue_diverted_t=round(burnt - actually_burnt, 2),
        residue_recoverable_t=round(gen - burnt, 2),
        burned_share_pct=round(100 * actually_burnt / max(gen, 1e-9), 1),
        co2_if_burned_t=round(burned_path_co2, 2),
        biogenic_co2_t=round(biogenic_co2, 2),
        co2_with_path_t=round(alt_co2, 2),
        co2_avoided_t=round(burned_path_co2 - alt_co2, 2),
        value_inr=round(econ["value_inr"]),
        cost_inr=round(econ["cost_inr"]),
        net_value_inr=round(econ["net_value_inr"]),
        water_saving_pct=p["water_saving_pct"],
        pathway=p["label"], pathway_glyph=p["glyph"], pathway_colour=p["colour"],
        hectares=round(total_ha, 2))


def self_check() -> None:
    print("\n[carbon] CO₂e & circularity  (prototype estimates)")
    print("-" * 78)
    print(f"  {'crop':<20s} {'gross t/ha':>10s} {'net t/ha':>9s}  "
          f"{'CH4':>6s} {'water':>6s} {'fertiliser':>10s} {'residue':>7s}")
    for k in ("paddy", "wheat", "maize", "cotton", "pulses", "oilseeds"):
        c = crop_carbon(k, lift_m=40.0)
        print(f"  {CROPS[k].label:<20s} {c.gross_t_ha:10.3f} {c.net_t_ha:9.3f}  "
              f"{c.rice_ch4_t_ha:6.3f} {c.irrigation_t_ha:6.3f} "
              f"{c.fertiliser_t_ha:10.3f} {c.residue_t_ha:7.3f}")
    print()
    cb = crop_carbon("paddy", 40.0, residue_path="burn")
    ci = crop_carbon("paddy", 40.0, residue_path="incorporate")
    print(f"  paddy burn -> incorporate : {cb.net_t_ha:.2f} -> {ci.net_t_ha:.2f} "
          f"t CO₂e/ha ({100*(1-ci.net_t_ha/max(cb.net_t_ha,1e-9)):.0f}% lower)")
    cf = crop_carbon("paddy", 40.0, "burn", irrigation_mode="flood")
    ca = crop_carbon("paddy", 40.0, "burn", irrigation_mode="controlled")
    print(f"  paddy flood -> AWD        : {cf.net_t_ha:.2f} -> {ca.net_t_ha:.2f} "
          f"t CO₂e/ha — methane is {cf.rice_ch4_t_ha:.2f} -> "
          f"{ca.rice_ch4_t_ha:.2f} t")
    print(f"  biogenic CO2 from burning : {cf.biogenic_t_ha:.2f} t/ha "
          f"(reported separately, NOT added to GWP)")
    print()
    print(residue_pathways(7.5, 0.48)[
        ["Pathway", "CO₂e (t/ha)", "Water saving", "Net ₹/ha"]].to_string(index=False))
    print()
    s = circularity_summary({"paddy": 2.0, "wheat": 2.0}, "incorporate")
    print(f"  4 ha paddy+wheat: {s['residue_generated_t']} t generated, "
          f"{s['residue_burned_t']} t burned, {s['co2_avoided_t']} t CO₂e avoided, "
          f"net ₹{s['net_value_inr']:,}")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
