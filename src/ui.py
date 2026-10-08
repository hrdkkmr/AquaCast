"""
Shared design system
====================

One place for the visual language, so eight tabs in four dashboards look like
one product rather than four different apps.

Rules this file enforces
------------------------
* **No decorative statistics.** Every number rendered through ``kpi()`` or
  ``so_what()`` must carry an interpretation. A figure with no decision
  attached is clutter, and the brief was explicit about that.
* **Simulated things are labelled where they are shown,** not in a footnote at
  the bottom of a page nobody scrolls to. ``prototype_badge()`` sits next to the
  thing it qualifies.
* **One accent, one warning colour, one severity ramp.** Charts share
  ``PLOTLY_TEMPLATE`` so a legend in the Admin dashboard means the same thing
  as the same legend on the farmer's phone.
* **Hedges are structural.** ``ESTIMATE`` / ``PROJECTED`` prefixes live here so
  a claim about carbon or credit can never slip out unqualified.
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
INK = "#0F1B2D"
MUTED = "#5A6B82"
LINE = "#E3E8EF"
SURFACE = "#FFFFFF"
CANVAS = "#F6F8FB"

ACCENT = "#0B6E4F"        # deep green — water / the product's own colour
ACCENT_SOFT = "#E6F2EE"
BLUE = "#0B63CE"
AMBER = "#C77B0B"
RED = "#B3261E"

SEVERITY = {"low": "#2E7D32", "moderate": "#F59E0B", "high": "#B3261E",
            "severe": "#8E1B14"}

ESTIMATE = "est."
PROJECTED = "projected"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif",
                  size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=28, b=8),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE,
                        font=dict(size=12, color=INK)),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED, size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
    ))

CROP_SERIES = ["#0B6E4F", "#C77B0B", "#0B63CE", "#7A4E9E", "#B3261E", "#4E7A8E"]


def install() -> None:
    """Register the template and the page chrome. Call once, at app start."""
    pio.templates["aquacast"] = PLOTLY_TEMPLATE
    pio.templates.default = "aquacast"


# ---------------------------------------------------------------------------
CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{
      font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  }}
  .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }}
  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.015em; }}

  .ac-card {{
      background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px;
      padding: 18px 20px; margin-bottom: 14px;
  }}
  .ac-card.ac-flat {{ background: transparent; border: none; padding: 0; }}

  .ac-kpi .ac-label {{
      font-size: 11px; font-weight: 600; letter-spacing: .07em;
      text-transform: uppercase; color: {MUTED};
  }}
  .ac-kpi .ac-value {{
      font-size: 30px; font-weight: 700; line-height: 1.05; color: {INK};
      margin-top: 4px; letter-spacing: -0.02em;
  }}
  .ac-kpi .ac-sub {{ font-size: 12px; color: {MUTED}; margin-top: 5px; }}
  .ac-delta-pos {{ color: {ACCENT}; font-weight: 600; }}
  .ac-delta-neg {{ color: {RED}; font-weight: 600; }}

  .ac-sowhat {{
      border-left: 3px solid {ACCENT}; background: {ACCENT_SOFT};
      border-radius: 0 10px 10px 0; padding: 12px 16px; margin: 10px 0 16px 0;
      font-size: 13.5px; color: {INK};
  }}
  .ac-sowhat b {{ color: {ACCENT}; }}
  .ac-sowhat.ac-warn {{ border-left-color: {AMBER}; background: #FDF6E7; }}
  .ac-sowhat.ac-warn b {{ color: {AMBER}; }}

  .ac-badge {{
      display: inline-block; font-size: 10.5px; font-weight: 600;
      letter-spacing: .04em; text-transform: uppercase;
      background: #EEF1F6; color: {MUTED}; border-radius: 999px;
      padding: 3px 10px; margin-left: 6px; vertical-align: middle;
  }}
  .ac-badge.ac-live {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
  .ac-badge.ac-sim {{ background: #FDF6E7; color: {AMBER}; }}

  .ac-rule {{ border: none; border-top: 1px solid {LINE}; margin: 22px 0 14px; }}

  .ac-rec {{
      border: 1px solid {LINE}; border-left-width: 3px; border-radius: 0 12px 12px 0;
      padding: 13px 16px; margin-bottom: 10px; background: {SURFACE};
  }}
  .ac-rec h4 {{ margin: 0 0 4px 0; font-size: 14.5px; }}
  .ac-rec p {{ margin: 0 0 5px 0; font-size: 12.8px; color: {MUTED}; }}
  .ac-rec .ac-impact {{ font-size: 12.5px; color: {INK}; font-weight: 500; }}

  div[data-testid="stTabs"] button {{ font-size: 13.5px; font-weight: 500; }}
  div[data-testid="stMetricValue"] {{ font-size: 24px; }}
  .ac-muted {{ color: {MUTED}; font-size: 12.5px; }}
</style>
"""


def css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def badge(text: str = "Prototype simulation — live integration planned for production",
          kind: str = "sim") -> str:
    """Inline label marking a simulated or estimated feature."""
    cls = "ac-badge ac-live" if kind == "live" else "ac-badge ac-sim"
    return f'<span class="{cls}">{text}</span>'


def kpi(label: str, value: str, sub: str = "", delta: str = "",
        positive: bool | None = None) -> None:
    """
    A single figure with its interpretation. ``sub`` is mandatory in practice —
    an unexplained number violates the "so what?" rule.
    """
    d = ""
    if delta:
        cls = "ac-delta-pos" if positive else "ac-delta-neg"
        d = f'<span class="{cls}">{delta}</span> '
    st.markdown(
        f'<div class="ac-card ac-kpi"><div class="ac-label">{label}</div>'
        f'<div class="ac-value">{value}</div>'
        f'<div class="ac-sub">{d}{sub}</div></div>', unsafe_allow_html=True)


def so_what(finding: str, action: str, warn: bool = False) -> None:
    """The platform's core pattern: a measured fact, then what to do about it."""
    cls = "ac-sowhat ac-warn" if warn else "ac-sowhat"
    st.markdown(f'<div class="{cls}"><b>So what?</b> {finding} &nbsp;→&nbsp; '
                f'<b>{action}</b></div>', unsafe_allow_html=True)


def rec(priority: str, title: str, action: str, impact: str,
        colour: str | None = None) -> None:
    c = colour or {"urgent": RED, "high": AMBER, "medium": BLUE,
                   "opportunity": ACCENT}.get(priority, MUTED)
    st.markdown(
        f'<div class="ac-rec" style="border-left-color:{c}">'
        f'<h4>{title}</h4><p>{action}</p>'
        f'<div class="ac-impact">📈 {impact}</div></div>', unsafe_allow_html=True)


def rule() -> None:
    st.markdown('<hr class="ac-rule">', unsafe_allow_html=True)


def section(title: str, sub: str = "", simulated: bool = False) -> None:
    b = badge() if simulated else ""
    st.markdown(f"### {title}{b}")
    if sub:
        st.markdown(f'<div class="ac-muted">{sub}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int = 340, title: str | None = None,
          legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, showlegend=legend,
                      title=dict(text=title, font=dict(size=13, color=MUTED))
                      if title else None)
    return fig


def fmt_inr(x: float) -> str:
    a = abs(x)
    if a >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if a >= 1e5:
        return f"₹{x/1e5:.2f} L"
    if a >= 1e3:
        return f"₹{x/1e3:.1f}K"
    return f"₹{x:,.0f}"


def fmt_num(x: float, dp: int = 0) -> str:
    return f"{x:,.{dp}f}"


def delta_pct(before: float, after: float) -> tuple[str, bool]:
    if abs(before) < 1e-9:
        return ("—", True)
    p = 100.0 * (after - before) / before
    return (f"{p:+.0f}%", p >= 0)
