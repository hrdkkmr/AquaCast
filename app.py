"""
AquaCast — AI-powered Climate-Smart Farm Digital Twin
====================================================

    FORECAST -> SIMULATE -> OPTIMIZE -> FINANCE -> VERIFY

Run with::

    streamlit run app.py

This file is only the router. It picks a destination from the sidebar and hands
off; every screen lives in ``pages/``.

===============  =========================================================
Destination      What it is
===============  =========================================================
Farmer           The planning console: eight tabs from overview to advisor
Farmer Phone     The same advice as a WhatsApp message on a real handset
Climate Ops      The lender / administrator console across 23 districts
Presentation     Stage view: architecture, impact, roadmap, deck
Forecast Lab     Modules A-F — the original AquaCast forecasting platform
===============  =========================================================

Zero credentials
----------------
Every screen here runs with no API keys, no database and no network. Anything
that would need one in production is mocked and labelled at the point of use —
see README §"Prototype vs Production Architecture".
"""
from __future__ import annotations

import warnings
from pathlib import Path

import streamlit as st

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src import ui                                    # noqa: E402

ROLE_HELP = {
    "Farmer": "One farm, seen whole — water, crop, carbon, money.",
    "Farmer Phone": "The same advice, as it reaches a farmer who has no laptop.",
    "Climate Ops": "The portfolio view: where to intervene, and what it is worth.",
    "Presentation": "Stage mode for the demo — jump to any section.",
    "Forecast Lab": "Modules A-F: the LSTM forecast and the aquifer model.",
}

st.set_page_config(page_title="AquaCast — Climate-Smart Farm Digital Twin",
                   page_icon="💧", layout="wide",
                   initial_sidebar_state="expanded")
ui.install()
ui.css()

pages = {
    "Farmer": st.Page("pages/1_🌾_Farmer.py", title="Farmer", icon="🌾",
                      default=True),
    "Farmer Phone": st.Page("pages/2_📱_Farmer_Phone.py", title="Farmer Phone",
                            icon="📱"),
    "Climate Ops": st.Page("pages/3_🏛_Climate_Operations.py",
                           title="Climate Operations", icon="🏛️"),
    "Presentation": st.Page("pages/4_🎤_Presentation.py", title="Presentation",
                            icon="🎤"),
    "Forecast Lab": st.Page("pages/5_📊_Forecast_Lab.py", title="Forecast Lab",
                            icon="📊"),
}

# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="padding:6px 0 2px">'
        f'<div style="font-size:19px;font-weight:800;letter-spacing:-.02em;'
        f'color:{ui.INK}">💧 AquaCast</div>'
        f'<div style="font-size:11.5px;color:{ui.MUTED};margin-top:2px">'
        f'Climate-Smart Farm Digital Twin</div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:10.5px;letter-spacing:.05em;color:{ui.ACCENT};'
        f'font-weight:700;margin:10px 0 8px">'
        f'FORECAST · SIMULATE · OPTIMIZE · FINANCE · VERIFY</div>',
        unsafe_allow_html=True)

    role = st.radio("Dashboard", list(pages), index=0, label_visibility="collapsed",
                    help="Switch the whole console.")
    st.caption(ROLE_HELP[role])
    st.markdown('<hr class="ac-rule">', unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:10.5px;color:{ui.MUTED};line-height:1.5">'
        f'<b>SANKALP · Climate Edition</b><br>Satin Finserv<br><br>'
        f'Runs with zero credentials. Simulated features are labelled '
        f'where they appear.</div>', unsafe_allow_html=True)

# st.navigation takes a sequence; we drive the destination from our own radio,
# so hand it the single chosen page.
st.navigation([pages[role]]).run()
