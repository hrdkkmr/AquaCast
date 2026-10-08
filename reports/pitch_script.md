# AquaCast-Punjab — 3-minute pitch script

**Total: 412 words ≈ 2 min 45 s speaking, leaving ~15 s for demo clicks.**
Speak at a deliberate ~150 words/minute. The pauses marked `[pause]` are where
the point lands — do not fill them.

Legend: **[SLIDE n]** = advance deck · **[CLICK]** = interact with the app

---

## 0:00 – 0:18 · The problem  (48 words)  **[SLIDE 2]**

> Punjab extracts **164 percent** of the groundwater it recharges. A hundred
> and eleven of its 153 blocks are over-exploited.
>
> `[pause]`
>
> That number is on every slide deck in this competition. What is *not* on any
> of them is what a farmer should actually do about it on Monday morning.

---

## 0:18 – 0:40 · What we built  (41 words)  **[SLIDE 21]**

> AquaCast forecasts the water table thirty, sixty and ninety days out —
> district by district, well by well.
>
> And then it does the part nobody else does: it converts that forecast into an
> **instruction**. Not a chart. An instruction.

---

## 0:40 – 1:35 · Demo 1 — the farmer  (72 words)  **[CLICK: View as → 🌾 Kisan]**

> This is what the farmer sees. Three hundred and seventy-five pixels, Punjabi
> or English, one dial.
>
> It does not say "your aquifer is stressed." It says **run your pump six hours
> today, between ten at night and four in the morning.**
>
> `[pause]`
>
> It knows the crop stage. If crown-root initiation falls in the next thirty
> days, it warns you — because missing *that* irrigation cannot be recovered
> later in the season. And it pushes as a WhatsApp message, in Gurmukhi.

---

## 1:35 – 2:05 · Demo 2 — the lender  (74 words)  **[CLICK: 🏛️ Administrator → 🗺️ Network map]**

> Here is what the lender sees. Every dot is a tubewell. Every circle is a real
> CGWB monitoring piezometer. Every red line is a dependency — this bore is
> pulling on *that* piezometer.
>
> Click one.
>
> `[CLICK a piezometer]`
>
> One bore moves its neighbour by **four centimetres**. Invisible.
>
> A village of forty-six bores moves it by **nearly a metre**.
>
> `[pause]`
>
> Nobody's pump is the problem — and everybody's is.

---

## 2:05 – 2:40 · The proof, and the caveat  (66 words)  **[SLIDE 8]**

> Accuracy: **eleven point seven centimetres** RMSE, out of sample. That is
> sixty-seven percent better than assuming tomorrow looks like today, and it
> beats a GRU and a linear probe on identical splits.
>
> Real data throughout — 31 in-situ CGWB readings, 305 bore logs, 122 wells,
> eight satellite rasters.
>
> And one honest caveat: the daily curve is a **physics reconstruction**, not
> continuous measurement, because only 31 readings exist. We say that before
> you have to ask.

---

## 2:40 – 3:00 · The money, and the close  (91 words)  **[SLIDE 24]**

> Last thing. Ask anyone what to finance a struggling farmer, and they'll say
> deepen the bore. So we priced it.
>
> Deepening costs **two lakh rupees** — and its benefit doesn't start for
> **twenty-one years**. Drip on the existing grid connection costs one lakh,
> pays back in under four years, and adds **thirteen years** of bore life.
>
> `[pause]`
>
> And the solar pump? Never pays. Because Punjab's farm power is already free.
>
> `[SLIDE 25]`
>
> Every other team will show you a groundwater chart.
>
> We show a farmer **which hours to pump** — and a bank **which loan is about
> to go dry**.
>
> Thank you.

---

# Delivery notes

**Timing discipline.** If you are over at the 2:40 mark, cut the "305 bore
logs, 122 wells" clause — keep the 31 readings, because that is what the honest
caveat hangs on. Never cut the caveat.

**Before you start — pre-warm the demo.** The network map is 2.7 MB and takes a
few seconds to render. Open the app, switch to the Administrator view, and load
the map tab *before* you begin speaking. Then flip between the Kisan and Admin
views live. If you switch cold, you will burn 10 seconds of silence.

**Backup if the demo dies.** Have `reports/network_map.html` open in a second
browser tab. It is the same map, standalone, no Streamlit needed. If the live
app fails, say "I have it exported here" and switch tabs — it reads as
preparedness, not failure.

**The two pauses are the pitch.** At 0:18 and at 2:40. Do not rush them. The
first sets up that you are different; the second is the number they will quote
back to you.

---

# If they stop you with questions

**"Only 31 water level readings — how is that a model?"**
31 is what CGWB publishes for these districts. We calibrate the physics
reconstruction against them, then measure skill against persistence and
seasonal-naive baselines, not against nothing. The honest framing: we are
forecasting a reconstruction, and we say so on the Model Lab tab.

**"Why not GRACE satellite data?"**
A GRACE cell is ten thousand square kilometres. Sangrur is three thousand six
hundred. It cannot see a village, it cannot separate groundwater from soil
moisture, and it arrives two months late. We use it as context, not as input.

**"Is the tubewell network real?"**
The bore *positions* are synthetic — seeded, inside the real district boundary,
on the real wheat mask. Every *property* is measured. And we report it as a
0.8 percent sample of the 92,000 bores actually in the ground, so our
interference numbers are a lower bound, not an exaggeration.

**"Why doesn't solar pay?"**
Because Punjab subsidises agricultural power to roughly zero. A solar pump
displaces electricity nobody was paying for. The water saving from drip is
worth real money; the energy saving is not.

**"What's the ask?"**
A pilot with a regional rural bank on one block, and access to CGWB's quarterly
well census to replace the static 31 readings with a live feed.
