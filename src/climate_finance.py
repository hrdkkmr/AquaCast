"""
Climate finance
===============

Turns an optimised farm plan into something a farmer can actually pay for, and
something a lender can actually underwrite.

The problem it solves
---------------------
The optimiser keeps recommending drip irrigation, and drip costs ₹78,000 per
hectare. A plan that raises income 20 % over five years is worthless to a
farmer who does not have ₹78,000 this season. So the finance module takes the
capital requirement, stacks the real subsidies against it, sizes the loan for
whatever is left, and shows the payback — on the farmer's cash flow, not on the
project's lifetime total.

Public scheme figures used here
-------------------------------
These are published rates, not invented ones. They are the reason the numbers
below look better than a plain commercial loan:

=============  =========================================================
Scheme         Support
=============  =========================================================
PMKSY-PDMC     55 % micro-irrigation subsidy (small/marginal), 45 % other
PM-KUSUM       60 % subsidy on stand-alone solar pumps (30 % central +
               30 % state); 50 % on grid-connected
SMAM           40-50 % subsidy on crop-residue machinery (Happy Seeder etc.)
KCC             Crop loan at 7 %, less 3 % prompt-repayment subvention →
               effective 4 % p.a.
NABARD          Concessional refinance / climate adaptation windows
=============  =========================================================

Prototype boundaries — read before quoting any number
-----------------------------------------------------
* **No approval is implied.** Nothing here is a sanction, an eligibility
  determination or an offer of credit. The lender view is the same
  ``credit_risk_engine`` the platform already used, re-scored on the proposed
  plan.
* **Carbon revenue is an estimate, not a credit.** The tonnages are computed
  from published factors; they are *not* validated, registered, issued, or
  sellable. Real issuance requires a registered methodology, a validation body
  and a monitoring period. The module says so on every screen that shows it.
* **Lender APIs, subsidy disbursement and KYC are mocked.** See README
  §"Prototype vs Production Architecture".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .crop_library import HA_PER_ACRE
from .farm_twin import FarmScenario, evaluate
from .optimizer import IRRIGATION_CAPEX_INR_PER_HA, RESIDUE_CAPEX_INR_PER_HA
from .carbon import GRID_KG_CO2_PER_KWH, pumping_energy_kwh

# ---------------------------------------------------------------------------
# Published scheme parameters
# ---------------------------------------------------------------------------
MICRO_IRRIGATION_SUBSIDY = {
    "marginal": 0.55, "small": 0.55, "medium": 0.45, "large": 0.45,
}
RESIDUE_MACHINERY_SUBSIDY = 0.50
SOLAR_PUMP_SUBSIDY = 0.60

KCC_RATE_PCT = 7.0                 # Kisan Credit Card headline rate
KCC_SUBVENTION_PCT = 3.0           # prompt repayment incentive
EFFECTIVE_KCC_RATE = (KCC_RATE_PCT - KCC_SUBVENTION_PCT) / 100.0
MAX_TENURE_YEARS = 7

#: Voluntary carbon market, ₹/t CO₂e. Quoted as a range because it is one —
#: this is an assumption, and it is shown as an assumption everywhere it is used.
#: How far below the 2021 baseline a typical Punjab submersible intake sits.
PUMP_INTAKE_BELOW_BASE_M = 6.0

CARBON_PRICE_INR_PER_T = 2_500.0
CARBON_PRICE_RANGE = (1_600.0, 4_100.0)

#: Share of a modelled reduction that a real MRV process would actually verify.
#: Additionality, leakage and measurement error all bite. Carrying the modelled
#: number straight through to revenue would be the single easiest way to
#: overclaim in this whole build.
VERIFICATION_HAIRCUT = 0.55


# ---------------------------------------------------------------------------
@dataclass
class FinancePackage:
    gross_capex_inr: float
    subsidy_inr: float
    loan_inr: float
    own_contribution_inr: float
    tenure_years: int
    rate_pct: float
    emi_inr: float
    annual_income_gain_inr: float
    annual_water_saving_m3: float
    annual_co2e_reduction_t: float
    verified_co2e_t: float
    carbon_revenue_inr: float
    net_annual_benefit_inr: float
    gross_annual_benefit_inr: float
    payback_years: float
    equity_payback_years: float
    npv_inr: float
    irr_pct: float
    schemes: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """One column, one dtype — a mixed int/str column fails Arrow."""
        def money(x):
            return ("—" if not np.isfinite(x) else f"₹{x:,.0f}")
        rows = [
            ("Capital required", money(self.gross_capex_inr)),
            ("Government subsidy", money(self.subsidy_inr)),
            ("Loan required", money(self.loan_inr)),
            ("Farmer's own contribution", money(self.own_contribution_inr)),
            ("Annual EMI", money(self.emi_inr) + " / month"),
            ("Added farm income", money(self.annual_income_gain_inr) + " / yr"),
            ("Carbon revenue (est.)", money(self.carbon_revenue_inr) + " / yr"),
            ("Net benefit after EMI", money(self.net_annual_benefit_inr) + " / yr"),
            ("Payback on total capital",
             "—" if not np.isfinite(self.payback_years)
             else f"{self.payback_years:.1f} yr"),
            ("Payback on farmer's own cash",
             "—" if not np.isfinite(self.equity_payback_years)
             else f"{self.equity_payback_years:.1f} yr"),
        ]
        return pd.DataFrame(rows, columns=["Item", "Amount"])


def _emi(principal: float, annual_rate: float, years: int) -> float:
    if principal <= 0 or years <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = int(years * 12)
    if r <= 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def _irr(cashflows: list[float], lo: float = -0.9, hi: float = 3.0) -> float:
    """Bisection IRR. Returns NaN when the sign never changes."""
    f = lambda r: sum(c / (1 + r) ** i for i, c in enumerate(cashflows))
    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return float("nan")
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
def build_package(current: FarmScenario, proposed: FarmScenario,
                  farm_size_class: str = "small",
                  tenure_years: int = 5,
                  include_solar: bool = False,
                  solar_capex_inr: float = 0.0,
                  own_contribution_pct: float = 0.10,
                  lift_m: float = 35.0,
                  efficiency: float = 0.55) -> FinancePackage:
    """
    Finance the gap between two farm plans.

    Stacks the subsidies first, asks the farmer for a modest contribution so
    there is skin in the game, and loans the rest on KCC terms against the
    additional cash flow the plan generates.
    """
    cur, prop = evaluate(current), evaluate(proposed)

    capex = (IRRIGATION_CAPEX_INR_PER_HA.get(proposed.irrigation, 0.0)
             * proposed.hectares
             + RESIDUE_CAPEX_INR_PER_HA.get(proposed.residue, 0.0)
             * proposed.hectares)
    schemes = []
    if IRRIGATION_CAPEX_INR_PER_HA.get(proposed.irrigation, 0.0) > 0:
        schemes.append(dict(scheme="PMKSY — Per Drop More Crop",
                            basis=f"Micro-irrigation ({proposed.irrigation})",
                            rate=MICRO_IRRIGATION_SUBSIDY.get(farm_size_class, 0.45),
                            note="55 % for small & marginal holdings, 45 % others"))
    if RESIDUE_CAPEX_INR_PER_HA.get(proposed.residue, 0.0) > 0:
        schemes.append(dict(scheme="SMAM — crop residue machinery",
                            basis=f"Residue management ({proposed.residue})",
                            rate=RESIDUE_MACHINERY_SUBSIDY,
                            note="Happy Seeder / super SMS, 50 % of cost"))
    if include_solar and solar_capex_inr > 0:
        capex += solar_capex_inr
        schemes.append(dict(scheme="PM-KUSUM — solar pump",
                            basis="Stand-alone solar pump", rate=SOLAR_PUMP_SUBSIDY,
                            note="60 % subsidy (30 % central + 30 % state)"))

    subsidy = sum(s["rate"] * _scheme_basis(s, proposed, solar_capex_inr)
                  for s in schemes)
    subsidy = min(subsidy, capex * 0.75)          # schemes never cover everything

    own = max(capex * own_contribution_pct, 0.0)
    loan = max(capex - subsidy - own, 0.0)

    emi = _emi(loan, EFFECTIVE_KCC_RATE, tenure_years)
    gain = prop.income_inr - cur.income_inr
    water = cur.gw_draft_m3 - prop.gw_draft_m3
    co2 = cur.co2e_t - prop.co2e_t

    # Carbon revenue: modelled reduction, haircut for what MRV would actually
    # verify, priced at a stated assumption. Never presented as a credit.
    verified = max(co2, 0.0) * VERIFICATION_HAIRCUT
    carbon_rev = verified * CARBON_PRICE_INR_PER_T

    net_annual = gain + carbon_rev - emi * 12.0
    payback = (capex / (gain + carbon_rev)) if (gain + carbon_rev) > 1e-6 else float("nan")

    # Equity cash flow: the farmer puts in their own contribution at t=0 and
    # services the loan out of the extra income. Charging the full loan at t=0
    # *and* as an EMI every year would double-count the same money.
    gross_benefit = gain + carbon_rev
    flows = [-own] + [gross_benefit - emi * 12.0] * tenure_years
    npv = sum(c / (1.09) ** i for i, c in enumerate(flows))
    irr = _irr(flows) * 100.0
    equity_payback = (own / max(gross_benefit - emi * 12.0, 1e-9)
                      if gross_benefit - emi * 12.0 > 1e-9 else float("nan"))

    notes = []
    if gain < 0:
        notes.append("This plan lowers farm income before finance costs — it is "
                     "a water and carbon play, not a profit play, and needs a "
                     "subsidy or a payment for the water saved to make sense.")
    if net_annual <= 0:
        notes.append("Loan service exceeds the added annual benefit at this "
                     "tenure. Extend the tenure or increase the subsidy share.")
    if co2 > 0:
        notes.append(f"Modelled reduction is {co2:.1f} t CO₂e/yr; only "
                     f"{verified:.1f} t is carried into revenue after a "
                     f"{int((1-VERIFICATION_HAIRCUT)*100)} % haircut for "
                     "additionality and measurement. Not a validated credit.")
    notes.append(f"Carbon priced at ₹{CARBON_PRICE_INR_PER_T:,.0f}/t CO₂e, an "
                 f"assumption within the observed ₹{CARBON_PRICE_RANGE[0]:,.0f}–"
                 f"₹{CARBON_PRICE_RANGE[1]:,.0f} voluntary-market range.")

    return FinancePackage(
        gross_capex_inr=capex, subsidy_inr=subsidy, loan_inr=loan,
        own_contribution_inr=own, tenure_years=tenure_years,
        rate_pct=EFFECTIVE_KCC_RATE * 100.0, emi_inr=emi,
        annual_income_gain_inr=gain, annual_water_saving_m3=water,
        annual_co2e_reduction_t=co2, verified_co2e_t=verified,
        carbon_revenue_inr=carbon_rev, net_annual_benefit_inr=net_annual,
        gross_annual_benefit_inr=gain + carbon_rev,
        payback_years=payback, equity_payback_years=equity_payback, npv_inr=npv, irr_pct=irr,
        schemes=schemes, notes=notes)


def _scheme_basis(s: dict, scn: FarmScenario, solar_capex: float) -> float:
    if s["scheme"].startswith("PM-KUSUM"):
        return solar_capex
    if s["scheme"].startswith("PMKSY"):
        return IRRIGATION_CAPEX_INR_PER_HA.get(scn.irrigation, 0.0) * scn.hectares
    return RESIDUE_CAPEX_INR_PER_HA.get(scn.residue, 0.0) * scn.hectares


# ---------------------------------------------------------------------------
def years_to_pump_failure(scn: FarmScenario, pump_depth_m: float | None = None
                          ) -> float:
    """
    Years until the water table passes the pump intake.

    The credit engine's drawdown score saturates once a pump is already
    failing, so two very different plans can score identically. This is the
    number that separates them, and the one a farmer actually feels.
    Returns ``inf`` when the farm is not drawing the aquifer down at all.
    """
    from .config import get_prior
    out = evaluate(scn)
    prior = get_prior(scn.district)
    base = float(prior.base_depth_2021_mbgl)
    pump = float(pump_depth_m if pump_depth_m is not None
                 else base + PUMP_INTAKE_BELOW_BASE_M)
    if out.depletion_m <= 1e-9:
        return float("inf")
    return max((pump - base) / out.depletion_m, 0.0)


def lender_view(current: FarmScenario, proposed: FarmScenario,
                horizon_years: int = 5) -> dict:
    """
    Re-score the farm through the existing credit engine on the proposed plan.

    The linkage is the aquifer. A plan that cuts groundwater draft slows the
    farm's own drawdown, which is a direct input to the underwriting score the
    lender already uses — so the same ``credit_risk_engine.score()`` that prices
    well-deepening capex here prices a crop change. Deliberately reusing the
    one engine means the farmer view and the lender view can never disagree.
    """
    from .credit_risk_engine import score as credit_score
    from .config import get_prior
    rows = []
    for label, scn in (("Current plan", current), ("Proposed plan", proposed)):
        out = evaluate(scn)
        prior = get_prior(scn.district)
        base = float(prior.base_depth_2021_mbgl)
        predicted = base + out.depletion_m * horizon_years
        pump = base + PUMP_INTAKE_BELOW_BASE_M
        r = credit_score(district=scn.district,
                         predicted_level_mbgl=predicted,
                         pump_set_depth_mbgl=pump,
                         current_level_mbgl=base,
                         decline_m_per_yr=out.depletion_m,
                         annual_net_income_inr=out.income_inr)
        yrs = years_to_pump_failure(scn, pump)
        saturated = bool(r.components["drawdown"].get("already_failing"))
        rows.append(dict(Plan=label,
                         **{"Drawdown (m/yr)": round(out.depletion_m, 2),
                            "Level in 5 yr (m)": round(predicted, 1),
                            "Pump fails in": ("—" if np.isinf(yrs)
                                              else f"{yrs:.1f} yr"),
                            "Risk score": round(float(r.score), 1)}))
    df = pd.DataFrame(rows)
    delta = float(df.loc[1, "Risk score"] - df.loc[0, "Risk score"])
    both_sat = all(evaluate(x).depletion_m > 0 for x in (current, proposed))
    return dict(table=df, risk_delta=delta,
                years_gained=(years_to_pump_failure(proposed)
                              - years_to_pump_failure(current)),
                drawdown_delta=float(df.loc[1, "Drawdown (m/yr)"]
                                     - df.loc[0, "Drawdown (m/yr)"]),
                note=("Same credit engine used across the platform. Note that "
                      "the drawdown component saturates once a pump is already "
                      "failing, so both plans can score alike on a 5-year view "
                      "while buying very different amounts of time — which is "
                      "why 'pump fails in' is shown alongside."
                      if both_sat else
                      "Same credit engine used across the platform."))


# ---------------------------------------------------------------------------
def self_check() -> None:
    print("\n[climate_finance] finance the transition")
    print("-" * 78)
    from .farm_twin import baseline
    from .optimizer import optimize
    cur = baseline("Sangrur", 2.0)
    prop = optimize(cur, "balanced").best
    pkg = build_package(cur, prop, tenure_years=5)
    print(f"  capital required      : ₹{pkg.gross_capex_inr:,.0f}")
    for s in pkg.schemes:
        print(f"    {s['scheme']:<34s} {s['rate']:.0%}  {s['note']}")
    print(f"  subsidy               : ₹{pkg.subsidy_inr:,.0f}")
    print(f"  farmer contributes    : ₹{pkg.own_contribution_inr:,.0f}")
    print(f"  loan                  : ₹{pkg.loan_inr:,.0f} "
          f"@ {pkg.rate_pct:.0f}% over {pkg.tenure_years} yr")
    print(f"  EMI                   : ₹{pkg.emi_inr:,.0f}/month")
    print(f"  added income          : ₹{pkg.annual_income_gain_inr:,.0f}/yr")
    print(f"  CO₂e reduction        : {pkg.annual_co2e_reduction_t:.2f} t modelled "
          f"-> {pkg.verified_co2e_t:.2f} t after haircut")
    print(f"  carbon revenue        : ₹{pkg.carbon_revenue_inr:,.0f}/yr (estimate)")
    print(f"  net after EMI         : ₹{pkg.net_annual_benefit_inr:,.0f}/yr")
    print(f"  payback on capital    : {pkg.payback_years:.1f} yr")
    print(f"  payback on own cash   : {pkg.equity_payback_years:.1f} yr")
    print(f"  NPV ₹{pkg.npv_inr:,.0f} · IRR {pkg.irr_pct:.1f}%")
    lv = lender_view(cur, prop)
    print()
    print(lv["table"].to_string(index=False))
    print(f"  risk {lv['risk_delta']:+.1f} pts · drawdown "
          f"{lv['drawdown_delta']:+.2f} m/yr · pump life "
          f"{lv['years_gained']:+.1f} yr gained")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
