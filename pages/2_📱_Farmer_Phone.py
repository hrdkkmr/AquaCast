"""
Farmer Phone
============

The same advice, on the device a farmer actually has.

This is deliberately **a rendered handset, not a responsive layout.** A
shrinking dashboard is not the same thing as a phone experience, and pretending
otherwise is how "mobile-first" products end up unusable in a field. The frame
below is a device: status bar, notch, home indicator, app grid and a WhatsApp
thread — so a reviewer can see what reaches the farmer, not what a desktop
window looks like when you narrow it.

What is real and what is mocked
-------------------------------
**Real:** the advice content. Every number in the message is computed from the
same twin the dashboards use; nothing on this screen is typed in by hand.

**Mocked:** the message transport. There is no Twilio / Gupshup / WhatsApp
Business API call, no SMS gateway and no IVR. The send button stages the
message locally and shows the delivery states that a real integration would
produce. See README §"Prototype vs Production Architecture".
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui                                              # noqa: E402
from src.crop_library import CROPS, HA_PER_ACRE                  # noqa: E402
from src.farm_twin import baseline, evaluate                     # noqa: E402
from src.optimizer import optimize                               # noqa: E402
from src.advisor import recommend, llm_narrate                   # noqa: E402
from src.climate_finance import years_to_pump_failure            # noqa: E402
from src.config import DISTRICTS                                 # noqa: E402

# ---------------------------------------------------------------------------
PHONE_CSS = """
<style>
  .ac-stage{ display:flex; gap:46px; align-items:flex-start; flex-wrap:wrap;
             justify-content:center; padding:8px 0 4px; }

  .dev{
      width:330px; height:690px; background:#10151c; border-radius:44px;
      padding:11px; box-shadow:0 26px 60px rgba(15,27,45,.34),
      0 0 0 1px rgba(255,255,255,.06) inset; position:relative; flex:none;
  }
  .dev .screen{
      width:100%; height:100%; border-radius:34px; overflow:hidden;
      background:linear-gradient(175deg,#0b3b2e 0%,#075e4b 48%,#04453a 100%);
      display:flex; flex-direction:column; position:relative;
  }
  .dev .notch{
      position:absolute; top:9px; left:50%; transform:translateX(-50%);
      width:112px; height:25px; background:#080c11; border-radius:16px; z-index:6;
  }
  .dev .status{
      display:flex; justify-content:space-between; align-items:center;
      padding:12px 17px 5px; font-size:11px; font-weight:600; color:#EAF3F0;
      letter-spacing:.02em; z-index:5;
  }
  .dev .wtop{
      background:#075E54; padding:7px 13px 9px; display:flex; align-items:center;
      gap:10px; z-index:5;
  }
  .dev .wtop .back{ color:#CFE8E2; font-size:15px; }
  .dev .wtop .av{
      width:33px;height:33px;border-radius:50%;background:linear-gradient(145deg,#25D366,#0FA958);
      display:flex;align-items:center;justify-content:center;font-size:18px;flex:none;
  }
  .dev .wtop .who{ font-size:13px; font-weight:600; color:#fff; line-height:1.15; }
  .dev .wtop .st{ font-size:10.5px; color:#B9DED1; }
  .dev .wtop .ic{ margin-left:auto; color:#CFE8E2; font-size:14px; letter-spacing:5px; }

  .dev .body{
      flex:1; overflow-y:auto; padding:12px 11px 8px;
      background:#E9E5DD;
      background-image:radial-gradient(rgba(0,0,0,.05) 1px, transparent 1px);
      background-size:19px 19px;
  }
  .dev .sysline{
      text-align:center; font-size:9.5px; color:#5E6B67; background:#DDE7E3;
      border-radius:6px; padding:3px 8px; margin:0 auto 9px; display:table;
      letter-spacing:.04em; text-transform:uppercase; font-weight:700;
  }
  .dev .msg{
      max-width:85%; border-radius:12px; padding:8px 10px 5px; margin-bottom:8px;
      font-size:11.6px; line-height:1.5; color:#12242C; position:relative;
      box-shadow:0 1px 1.5px rgba(0,0,0,.13); white-space:pre-wrap;
      font-family:'Inter',system-ui,sans-serif;
  }
  .dev .msg.in{ background:#fff; border-top-left-radius:3px; margin-right:auto; }
  .dev .msg.out{
      background:#DCF8C6; border-top-right-radius:3px; margin-left:auto;
  }
  .dev .msg .tick{
      float:right; font-size:9px; color:#4FA96B; margin-left:6px; margin-top:5px;
  }
  .dev .msg .t{ float:right; font-size:9px; color:#8A9A95; margin-left:6px; margin-top:5px; }
  .dev .msg b{ font-weight:700; }
  .dev .msg .card{
      background:#F3F7F5; border-left:3px solid #0B6E4F; border-radius:7px;
      padding:6px 8px; margin:6px 0 2px; font-size:11px;
  }
  .dev .chips{ display:flex; flex-wrap:wrap; gap:5px; margin-top:7px; }
  .dev .chip{
      background:#EEF4F1; border:1px solid #CFE0D8; border-radius:999px;
      padding:3px 9px; font-size:10.2px; color:#0B6E4F; font-weight:600;
  }
  .dev .input{
      background:#F0F0F0; padding:8px 11px; display:flex; align-items:center;
      gap:7px; z-index:5;
  }
  .dev .input .fld{
      flex:1; background:#fff; border-radius:19px; padding:7px 13px;
      font-size:11.3px; color:#8A9A95;
  }
  .dev .input .snd{
      width:32px;height:32px;border-radius:50%;background:#0FA958;color:#fff;
      display:flex;align-items:center;justify-content:center;font-size:15px;flex:none;
  }
  .dev .home{
      height:17px; display:flex; align-items:center; justify-content:center;
      background:#F0F0F0; z-index:5;
  }
  .dev .home i{ width:104px; height:4px; border-radius:3px; background:#1F2A33;
                display:block; opacity:.75; }

  .apps{ display:grid; grid-template-columns:repeat(4,1fr); gap:13px 9px;
         padding:16px 13px; }
  .app{ text-align:center; }
  .app .ic{
      width:50px;height:50px;border-radius:13px;margin:0 auto 5px;
      display:flex;align-items:center;justify-content:center;font-size:23px;
      color:#fff; box-shadow:0 3px 8px rgba(0,0,0,.22);
  }
  .app .lb{ font-size:9.6px; color:#EAF3F0; font-weight:500; }
  .dev .dock{
      background:rgba(0,0,0,.22); margin:11px 13px 15px; border-radius:19px;
      display:flex; justify-content:space-around; padding:11px 8px;
  }
  .dev .dock .ic{ width:45px;height:45px;border-radius:12px;display:flex;
      align-items:center;justify-content:center;font-size:21px; }

  .spec{ flex:1; min-width:290px; max-width:430px; }
</style>
"""

APPS = [
    ("🌾", "AquaCast", "linear-gradient(145deg,#0B6E4F,#128C7E)"),
    ("💧", "Tubewell", "linear-gradient(145deg,#0B63CE,#3B82F6)"),
    ("🌦️", "Monsoon", "linear-gradient(145deg,#5B8DEF,#93C5FD)"),
    ("🏦", "Kisan Card", "linear-gradient(145deg,#B45309,#F59E0B)"),
    ("📈", "Mandi", "linear-gradient(145deg,#7A4E9E,#A855F7)"),
    ("📷", "Crop Doc", "linear-gradient(145deg,#2E7D32,#4ADE80)"),
    ("📞", "Sarkar", "linear-gradient(145deg,#B3261E,#EF4444)"),
    ("⚙️", "Settings", "linear-gradient(145deg,#475569,#94A3B8)"),
]


# ---------------------------------------------------------------------------
def _message(farmer: str, district: str, hectares: float, lang: str) -> str:
    """Build the WhatsApp payload from the same engine the dashboards use."""
    base = baseline(district, hectares)
    cur = evaluate(base)
    ropt = optimize(base, "balanced")
    recs = recommend(base, proposed=ropt.best)
    yrs = years_to_pump_failure(base)

    if lang == "pa":
        head = f"ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ {farmer} ਜੀ।"
    elif lang == "hi":
        head = f"नमस्ते {farmer} जी।"
    else:
        head = f"Hello {farmer} ji."

    top = recs[0] if recs else None
    alt = ", ".join(f"{CROPS[k].label} {v:.0%}"
                    for k, v in ropt.best.allocation.items())

    body = (
        f"{head}\n\n"
        f"*AquaCast — {district}, {hectares:.1f} ha ({hectares/HA_PER_ACRE:.1f} acre)*\n\n"
        f"{top.title if top else 'Your weekly plan'}\n"
        f"{top.action if top else ''}\n\n"
        f"*What changes*\n"
        f"• Groundwater {cur.gw_draft_m3:,.0f} → {ropt.best_outcome.gw_draft_m3:,.0f} m³\n"
        f"• Income {ui.fmt_inr(cur.income_inr)} → {ui.fmt_inr(ropt.best_outcome.income_inr)}\n"
        f"• Pump intake has {yrs:.1f} yr of water left\n\n"
        f"*Suggested mix*\n{alt}\n"
        f"{ropt.best.irrigation.replace('_',' ').title()} · "
        f"{ropt.best.residue.title()}\n\n"
        f"Reply 1 for the loan options, 2 for a call back."
    )
    return body


def _esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _render_phone(msg: str, screen: str, time_str: str = "9:41") -> str:
    if screen == "WhatsApp":
        body = f"""
      <div class="wtop">
        <span class="back">‹</span>
        <div class="av">🌾</div>
        <div><div class="who">AquaCast Kisan</div>
             <div class="st">online · Punjabi, हिंदी, English</div></div>
        <span class="ic">📞 ⋮</span>
      </div>
      <div class="body">
        <div class="sysline">Today</div>
        <div class="msg in">{_esc(msg)}<span class="t">9:41 AM ✓✓</span></div>
        <div class="msg out">1<span class="tick">✓✓</span></div>
        <div class="msg in">
          <b>ਕਰਜ਼ਾ / लोन / Loan</b>
          <div class="card">Drip + residue kit · ₹1,56,000<br>
            Subsidy ₹85,800 (55%)<br>
            Loan ₹54,600 @ 4% · EMI ₹1,006/mo<br>
            आपका हिस्सा / your share ₹15,600</div>
          <div class="chips"><span class="chip">Apply</span>
            <span class="chip">Call me</span>
            <span class="chip">More detail</span></div>
          <span class="t">9:42 AM ✓✓</span>
        </div>
      </div>
      <div class="input"><div class="fld">Message</div>
        <div class="snd">➤</div></div>
      <div class="home"><i></i></div>"""
    else:
        apps = "".join(
            f'<div class="app"><div class="ic" style="background:{bg}">{g}</div>'
            f'<div class="lb">{n}</div></div>' for g, n, bg in APPS)
        body = f"""
      <div class="apps">{apps}</div>
      <div class="dock">
        <div class="ic" style="background:#075E54">📞</div>
        <div class="ic" style="background:#128C7E">💬</div>
        <div class="ic" style="background:#0B63CE">🌐</div>
        <div class="ic" style="background:#B45309">📷</div>
      </div>
      <div class="home"><i></i></div>"""

    return f"""
<div class="ac-stage">
  <div class="dev">
    <div class="screen">
      <div class="notch"></div>
      <div class="status"><span>{time_str}</span>
        <span>●●●● 5G ▮ 78%</span></div>
      {body}
    </div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
st.markdown("## 📱 Farmer Phone")
st.markdown(
    f'<div class="ac-muted">What reaches a farmer who has a smartphone and no '
    f'laptop. Rendered as a device — status bar, notch, home indicator, app '
    f'grid, WhatsApp thread — because a narrowed dashboard is not a phone '
    f'experience. Every number in the message comes from the same twin as the '
    f'dashboards; only the transport is simulated.</div>' + ui.badge(
        "Message delivery mocked — Twilio / WhatsApp Business in production"),
    unsafe_allow_html=True)

st.markdown(PHONE_CSS, unsafe_allow_html=True)

c = st.columns([1, 1, 1, 1])
with c[0]:
    district = st.selectbox("District", DISTRICTS,
                            index=DISTRICTS.index("Sangrur")
                            if "Sangrur" in DISTRICTS else 0)
with c[1]:
    hectares = st.slider("Land (ha)", 0.5, 10.0, 2.0, 0.5)
with c[2]:
    lang = st.selectbox("Language", ["en", "hi", "pa"], index=2,
                        format_func=lambda k: {"en": "English", "hi": "हिंदी",
                                               "pa": "ਪੰਜਾਬੀ"}[k])
with c[3]:
    farmer = st.text_input("Farmer name", "Baldev Singh")

screen = st.radio("Screen", ["WhatsApp", "Home"], horizontal=True)
st.markdown('<hr class="ac-rule">', unsafe_allow_html=True)

with st.spinner("Computing advisory…"):
    msg = _message(farmer, district, hectares, lang)

l, r = st.columns([1.05, 1], gap="large")
with l:
    st.markdown(_render_phone(msg, screen), unsafe_allow_html=True)
with r:
    st.markdown("#### Why it looks like this")
    st.markdown(
        f'<div class="ac-card"><div style="font-weight:600;font-size:14px">'
        f'One message, one decision</div><div class="ac-muted" style="margin-top:5px">'
        f'A farmer will not open eight tabs. The message leads with the single '
        f'highest-priority action, then gives the two numbers that justify it, '
        f'then offers two replies. Everything else stays in the dashboards for '
        f'the field officer.</div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ac-card"><div style="font-weight:600;font-size:14px">'
        f'Zero data, zero app install</div><div class="ac-muted" '
        f'style="margin-top:5px">WhatsApp is already on the phone and already '
        f'understood. No app to download, no login, works on 2G, and the '
        f'message is readable months later in the thread.</div></div>',
        unsafe_allow_html=True)

    st.markdown("#### Message payload")
    st.code(msg, language="markdown")

    st.markdown("#### Planned production integrations")
    for k, v in [
        ("WhatsApp Business API", "Template messages, opt-in, delivery receipts"),
        ("Twilio / Gupshup", "SMS fallback for feature phones (~38% of Punjab)"),
        ("IVR voice", "Punjabi TTS for low-literacy users"),
        ("Two-way parsing", "Replies like '1' become callbacks in the CRM"),
    ]:
        st.markdown(f"- **{k}** — {v}")
    st.caption("None of these are wired up in this prototype. The send button "
               "stages the message locally only.")
