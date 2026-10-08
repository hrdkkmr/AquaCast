# 💧 AquaCast-Punjab

### AI-Powered Aquifer Depletion & Irrigation Advisory System
**Sankalp — Climate-Smart Agriculture · Water Conservation · Rural Credit Risk Mitigation**

> Punjab grows wheat on 3.5 M hectares every Rabi and pumps the aquifer dry to do it.
> AquaCast forecasts the water table **30 / 60 / 90 days ahead**, district by district and
> well by well, then converts that forecast into a pumping quota, a bilingual WhatsApp
> advisory, and a credit-risk score a lender can actually price.

---

## 0. Executive summary

Punjab extracts **more groundwater than the state receives in recharge** — a stage
of groundwater extraction of **163.76 %** in the 2023 CGWB assessment (reported in
the Rajya Sabha, December 2024: 27.8 bcm extracted against 16.98 bcm extractable
[1](https://www.tribuneindia.com/news/haryana/groundwater-extraction-stage-reaches-136-in-haryana-164-in-punjab/)),
easing to **152.22 %** in the 2026 assessment [2](https://theprint.in/india/cgwb-report-shows-decline-in-extraction-better-groundwater-levels-punjab-minister/3030157/).
**111 of 153 assessment blocks remain over-exploited** [3](https://www.royalpatiala.in/drain-strain-punjabs-111-blocks-overexploited-second-highest-in-india/).

That number is known. What is missing is the last mile: nobody tells the farmer
*which hours to pump this week*, and nobody tells the lender *which loan is about
to go dry*.

AquaCast closes that gap. It forecasts the water table **30 / 60 / 90 days ahead**
with a PyTorch LSTM (Sangrur out-of-sample **RMSE 0.117 m, R² 0.935, +67.4 % skill
over persistence**), and converts the forecast into three artefacts:

| For | Artefact | Ends in |
|---|---|---|
| **The kisan** | A 375 px bilingual card: one dial, one instruction | *"Run the pump 6 h today, 22:00–04:00"* |
| **The farmer** | Water budget, neighbour-interference calculator, rotation simulator, drilling finance | *Shift 2 000 ha out of paddy → save 6.8 MCM* |
| **Satin Finserv** | Block stress heatmap, portfolio at risk, deepening-vs-drip capex | *Drip on grid: 3.7 yr payback. Solar: never pays.* |

**The one-line pitch:** every other team will show you a groundwater chart. We show
a farmer which hours to pump, and a bank which loan is about to go dry.

---

## 1. What makes this different

Most hackathon prototypes generate a fake CSV, fit a model, and stop. AquaCast is
**grounded in real measurements at every layer**:

| Layer | Real or modelled? | Source shipped in `data/` |
|---|---|---|
| Rainfall (daily, 3 gauges, 1 540 days) | 🟢 **Measured** | India-WRIS hourly telemetry, hourly *increments* series |
| Air temperature (hourly → daily Tmax/Tmin) | 🟢 **Measured** | Punjab SW telemetry, 5 stations, 62 339 rows |
| Water level | 🟢 **Measured (sparse)** | 31 in-situ CGWB readings, 10 wells, 2021-2024 |
| District / block polygons | 🟢 **Real** | India-WRIS District Boundary (22 Punjab districts) |
| Observation-well network | 🟢 **Real** | 122 CGWB + Punjab-GW wells inside Sangrur |
| Aquifer thickness, depth-to-first-aquifer | 🟢 **Real** | CGWB rasters, reprojected to a 440 m Punjab grid |
| Soil texture / depth / slope / productivity / erosion | 🟢 **Real** | NBSS&LUP rasters |
| Rabi wheat mask 2023-24 | 🟢 **Real** | Satellite-derived crop mask |
| Daily water-level *trajectory* | 🟡 **Physics + ML** | FAO-56 water balance, calibrated against the 31 CGWB readings |

Every row of the training frame carries a `*_src` provenance column (`telemetry` vs
`generated`), and Tab 4 of the dashboard prints the full ledger.

---

## 2. Headline results

Out-of-sample **2024-02-29 → 2026-07-03** (held out by a chronological split; the
network never sees it). Identical split, identical features, identical epochs for every
row of the comparison.

### Per-district accuracy

| District | t+30 | t+60 | t+90 | **overall RMSE** | MAE | R² | Trend acc | Skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| **Sangrur** | 0.125 m | 0.122 m | 0.104 m | **0.117 m** | 0.092 m | **0.935** | 91.7 % | **+67.3 %** |
| **Ludhiana** | 0.092 m | 0.098 m | 0.107 m | **0.099 m** | 0.080 m | **0.938** | 90.5 % | **+70.9 %** |
| **Moga** | 0.122 m | 0.115 m | 0.106 m | **0.114 m** | 0.089 m | **0.942** | 91.7 % | **+68.1 %** |

Baselines on the same split: persistence 0.341–0.359 m, seasonal-naive 0.461–0.594 m,
linear-trend 0.315–0.329 m. AquaCast beats all three on every district.

### Model bake-off (Sangrur test split)

| Model | RMSE | MAE | Trend acc | Params |
|---|---|---|---|---|
| **LSTM (AquaCast — shipped, pooled)** | **0.1173 m** | 0.0921 | **0.917** | 52 547 |
| Linear probe (flatten-the-window) | 0.1519 m | 0.1160 | 0.910 | 1 443 |
| LSTM, single-district, from scratch | 0.1589 m | 0.1313 | 0.865 | 52 547 |
| GRU (2-layer) | 0.1850 m | 0.1584 | 0.829 | 39 363 |
| Persistence (naive) | 0.3591 m | 0.3020 | 0.042 | 0 |

The recurrent model earns its place **only once the three districts are pooled** —
training the identical architecture on Sangrur alone costs 0.042 m of RMSE. That is the
whole argument for multi-task pooling, and we show the number rather than assert it.

### Two results that look wrong and aren't

* **Error *falls* with horizon** (0.125 m at t+30 → 0.104 m at t+90). The spread of the
  target grows with horizon (σ 0.21 m → 0.43 m) while the *unpredictable* part does not:
  at 30 days the change is dominated by individual rain events that a 60-day window
  physically cannot contain; at 90 days it is dominated by the Rabi-vs-monsoon calendar,
  which is perfectly predictable. Per-horizon R² therefore rises 0.65 → 0.94.
* **Residual bias +0.045 m** (the model very slightly over-predicts deepening). We report
  it rather than subtract it.

Physics-model seasonal cycle vs the 31 CGWB readings: **RMSE 0.50 m** after removing
each well's static lithology offset (per-well levels range 28 m → 45 m, so the offset
is a legitimate nuisance parameter, not a fudge).

### Operational forecast

The live advisory is issued from the newest observation in the record — not from the
last date with a realised 90-day target. As of **02 Oct 2026** the dashboard's
+90-day forecast is a **0.41–0.46 m monsoon recovery**, which is what the observed
October→December climatology has done in every year of the record.

---

## 3. Architecture

```mermaid
flowchart TB
    subgraph SPATIAL["SPATIAL INPUTS — real vectors & rasters (data/)"]
        direction LR
        S1["District Boundary<br/>District Headquarter"]
        S2["Groundwater Level Station<br/>Groundwater Station<br/>WIMS Station"]
        S3["Rainfall Station<br/>Litholog (305 bore logs)"]
        S4["Rasters 384 m<br/>aquifer thickness · depth to<br/>first aquifer · soil texture<br/>· Rabi wheat mask"]
    end

    subgraph TEMPORAL["HYDRO-TEMPORAL ENGINE"]
        direction TB
        T1["telemetry_ingest.py<br/>WRIS parser + QC + provenance"]
        T2["dataset_generator.py<br/>FAO-56 water balance<br/>ΔS = Recharge(R, infiltration)<br/>− Discharge(pumping, baseflow)<br/>gamma recharge lag kernel"]
        T3["model_lstm.py<br/>PyTorch LSTM 60 d × 8 ch<br/>MC-dropout → 30/60/90 d"]
        T1 --> T2 --> T3
    end

    subgraph STRATA["STRATIFIED & CROP MODEL"]
        direction TB
        C1["crop_cycle_model.py (B)<br/>Rabi wheat → Zaid → Kharif paddy<br/>Draft = Σ A×Depth (MCM)"]
        C2["aquifer_strata.py (C)<br/>L1 0–40 · L2 40–90 · L3 &gt;90 m<br/>₹/ft to deepen"]
        C3["spatial_network.py (A)<br/>Cooper–Jacob / Theis<br/>N×N station influence matrix"]
        C1 --- C2 --- C3
    end

    subgraph OUT["TRIPLE DASHBOARD (Module E)"]
        direction LR
        V1["📱 Kisan Mobile<br/>375 px · ਪੰਜਾਬੀ/English<br/>dial + WhatsApp push"]
        V2["🧑‍🌾 Farmer Desktop<br/>water budget · interference<br/>rotation · drilling ₹"]
        V3["🏛️ Admin & Satin Finserv<br/>network map · block heatmap<br/>PAR ₹ Cr · power optimiser"]
    end

    SPATIAL --> TEMPORAL
    SPATIAL --> STRATA
    TEMPORAL --> C1
    TEMPORAL --> C3
    STRATA --> OUT

    R1["credit_risk_engine.py (D)<br/>0–100 = w₁·Drawdown + w₂·Stage + w₃·Capex<br/>KCC bands · drip vs deepening NPV"] --> V3
    R2["benchmarks.py (F)<br/>vs GRACE · India-WRIS · PAU"] --> V3
    T3 --> R1

    style SPATIAL fill:#0d3b66,stroke:#0EA5E9,color:#E6F4FF
    style TEMPORAL fill:#123a2a,stroke:#22C55E,color:#E6F4FF
    style STRATA fill:#3a2a12,stroke:#F59E0B,color:#FDE9C8
    style OUT fill:#2a1230,stroke:#8B5CF6,color:#EDE4FF
    style R1 fill:#3a1212,stroke:#EF4444,color:#FFE4E4
    style R2 fill:#12283a,stroke:#38BDF8,color:#E4F4FF
```

The same pipeline, written as the module map it is compiled from:

```
run_pipeline.py
│
├─[0] pre-flight ............ verify / rebuild the Punjab spatial clips
│
├─[1] src/spatial_loader.py ............. MODULE 1  spatial ingestion + crop baseline
│     22 Punjab districts · 122 Sangrur wells · 8 raster covariates (384 m)
│     calibrated priors: 280 000 ha wheat · 350 mm · 980 MCM · Sy = 0.12
│
├─[2] src/telemetry_ingest.py ........... MODULE 2a real telemetry ingestion + QC
│     WRIS Excel parser · spike/stuck-sensor QC · provenance flags
│
├─[3] src/dataset_generator.py .......... MODULE 2b hybrid time-series synthesis
│     Richardson weather generator FITTED TO REAL TELEMETRY (not guessed)
│     + FAO-56 dual-Kc crop demand (Hargreaves-Samani ET0)
│     + lumped aquifer water balance with a GAMMA RECHARGE LAG KERNEL
│     + calibration against the 31 CGWB readings
│     → data/processed/processed_<district>_timeseries.csv
│
├─[4] src/model_lstm.py ................. MODULE 3  PyTorch LSTM
│     60 d × 8 channels → 2-layer LSTM(64, dropout .2) → 3 direct outputs
│     MC-dropout uncertainty · permutation importance · input saliency
│     → models/lstm_aquifer.pth
│
├─[5] src/crop_cycle_model.py ........... MODULE B  full-year rotation
│     Rabi wheat (Kc 1.15) → Zaid → Kharif paddy (Kc 1.20, ponded)
│     Draft = Σ A_crop × Depth_crop · calibrated to PADDY_GW_DRAFT_MM
│
├─[6] src/aquifer_strata.py ............. MODULE C  three-tier strata + ₹/ft
│     L1 shallow · L2 semi-confined · L3 deep · T, S, EC from real Lithologs
│
├─[7] src/spatial_network.py ............ MODULE A  tubewell interference
│     Cooper–Jacob/Theis · village clusters · N×N station influence matrix
│
├─[8] src/credit_risk_engine.py ......... MODULE D  0–100 underwriting score
│     w₁·Drawdown + w₂·CriticalStage + w₃·CapexRatio · drip vs deepening NPV
│
├─[9] src/benchmarks.py ................. MODULE F  GRACE · India-WRIS · PAU
│
└─[10] app.py + src/personas.py ......... MODULE E  triple dashboard (3 roles)
```

---

## 4. Six engineering decisions worth defending

1. **Physics-prior residual learning.** LSTMs extrapolate linear trends badly. The
   calibrated CGWB depletion rate is subtracted from the target and re-added
   analytically, so the network only has to model the *stationary seasonal residual* —
   the part that actually generalises.
2. **Cyclical calendar encoding.** Two extra channels (sin/cos of day-of-year) tell the
   network where it stands in the Rabi calendar. Ablation on the Sangrur test set:
   **RMSE 0.203 m → 0.113 m**.
3. **Multi-district pooling.** Training on Sangrur + Ludhiana + Moga triples the
   sequence count (571 → 1 713) and is the difference between an LSTM that memorises
   and one that generalises.
4. **A lagged recharge kernel.** Central Punjab's water table is 30-40 m deep, so the
   monsoon arrives months late. The CGWB hydrographs prove it — they are *deepest in
   August* and *shallowest in January*. A gamma transfer function with a calibrated
   ~150-day mean lag reproduces that phase exactly.
5. **MC-dropout confidence bands.** Dropout stays on at inference; 60 stochastic passes
   give a genuine epistemic σ. No fake ±5 % fan chart.
6. **Honest baselines.** Persistence, seasonal-naive, linear-trend, a GRU and a
   flatten-the-window *linear probe* are all trained and reported next to the model on
   the identical split. If the LSTM didn't earn its place, you'd see it.
7. **Adversarial self-auditing.** Every headline number is reproducible by
   `python3 run_pipeline.py`, and the two substantive bugs we caught in our own
   evaluation path are now regression-tested in `tests/smoke_test.py`:
   * *Stale forecast window.* `build_sequences` stops 90 days short of the record
     because every training window needs a realised target — so its last row issues a
     forecast from 90 days **before** today. `live_window()` builds the operational
     window that ends on the newest datum; the test asserts the issue date equals
     `df.date.max()` and that the 90-day delta agrees with the observed Oct→Dec
     monsoon recovery to within 0.30 m. Catching it changed the dashboard's live
     forecast from −0.01 m to −0.45 m.
   * *Trend-prior mismatch.* `predict()` re-adds the depletion prior (rate × horizon) on
     the way out, but the ground truth it was scored against did not. That single
     omission faked a +0.19 m bias and inflated the reported RMSE from 0.117 m to
     0.210 m. Fixing it is why every number in §2 is where it is.

---

## 5. The wow features

* **🧪 Digital Twin** — move a policy lever (monsoon anomaly, micro-irrigation adoption,
  paddy transplant delay, canal availability) and watch the aquifer trajectory bend in
  ~30 ms. It surfaces the counter-intuitive result that *saving irrigation water does
  not save the aquifer 1:1*, because flood-irrigation losses currently return as
  recharge — the aquifer only keeps the net difference.
* **🗺️ 122-well risk map** — the district LSTM forecast is downscaled to every CGWB
  observation well using measured per-well offsets (IDW interpolation), with pump
  installation depth estimated from the real depth-to-first-aquifer raster.
* **📲 Phone-notification advisory card** — a real WhatsApp-shaped preview in English
  *and* Gurmukhi Punjabi, with a 160-character SMS fallback and a print-ready
  advisory pack export.
* **🏦 Lender view** — six-component auditable risk score → PD → EAD × PD × LGD →
  portfolio PAR and the loss a 90-day warning can avert.
* **🔬 Model Lab** — live training curve, test-set validation, skill scatter,
  permutation importance, gradient saliency ("which of the last 60 days drove this
  forecast?") and a full data-provenance ledger.

---

## 6. Climate Digital Twin — the upgrade

The platform is now organised as one chain:

    FORECAST  ->  SIMULATE  ->  OPTIMIZE  ->  FINANCE  ->  VERIFY

| Step | Module | What it does |
|------|--------|--------------|
| Forecast | `model_lstm.py`, `dataset_generator.py` | PyTorch LSTM on real India-WRIS telemetry + FAO-56 + lumped aquifer balance |
| Simulate | **`farm_twin.py`** | Recomputes water, income, carbon, residue, drawdown and risk for any farm setup |
| Optimize | **`optimizer.py`** | Scores ~12,500 candidate plans against four objectives; returns the Pareto frontier |
| Finance | **`climate_finance.py`** | Stacks real subsidies, sizes the loan, computes payback; re-scores the lender's own model |
| Verify | **`impact.py`** | Monte Carlo uncertainty on every claim; states which MRV steps are performed |
| Decide | **`crop_library.py`, `carbon.py`, `advisor.py`, `satellite.py`** | Crop economics, emissions, recommendations, field observation |

### Four dashboards

| Dashboard | Tabs |
|-----------|------|
| **Farmer** | Overview · Water Intelligence · Crop Intelligence · Climate Twin · Circularity · Climate Finance · Impact · AI Advisor |
| **Farmer Phone** | A rendered handset — status bar, notch, home indicator, app grid, WhatsApp thread |
| **Climate Operations** | Regional Overview · Water Risk · Crop Patterns · Climate Impact · Interventions · Finance · Verification |
| **Presentation** | Title · problem · product · architecture · twin · impact · moat · roadmap · deck & video, with **Demo Mode** jump-to-section |

**Modules A–F are unchanged and preserved** as the *Forecast Lab* destination.

### Three findings the model produced

1. **A Punjab farm is a double crop, and modelling it as one understates it by half.** A 2 ha single-season paddy farm nets ₹72,000/yr. Modelled as the paddy–wheat rotation it actually is, the same farm nets **₹1,31,000/yr = ₹26,500/acre** — exactly the constant `credit_risk_engine` was calibrated against. The twin and the lender agree because they must.

2. **Paddy straw is burnt because it is worth ₹900/t; wheat straw is not burnt because it is worth ₹3,200/t.** The residue problem is a price problem. A flat per-hectare residue value hides this, so the module prices by crop.

3. **Diversification + drip + residue marketing beats paddy–wheat on all three axes at once:** **−47% groundwater, +20% income, −40% CO₂e.** That is not a trade-off — the current plan is simply *dominated*, sitting well inside the efficient frontier.

### Honesty constraints built into the model

These are deliberate. Removing any of them would make the numbers look better.

* **Paddy's methane is modelled.** Flooded rice emits ~30 kg CH₄/ha/season (0.84 t CO₂e) — the largest single item in paddy's footprint. Alternate wetting and drying halves it. Omitting this would have understated paddy by ~25%.
* **Biogenic CO₂ is reported but not counted.** Burning releases 4.68 t CO₂/ha of CO₂ that was fixed last season. Counting it would roughly double the apparent footprint; it is shown separately instead.
* **The moong window is capped at half the farm.** Short-duration moong is a rotating catch crop, not a third season. Uncapped, the optimiser invents a triple-cropped farm nobody runs.
* **Late transplanting costs grain.** Yield falls ~0.8%/day after 25 June. The water saving is not free money.
* **Drip is not offered on paddy.** It is not a practice that exists at scale in Punjab.
* **Carbon revenue carries a 45% haircut** for additionality, leakage and measurement error, and is labelled an estimate — never a credit.

---

## 7. Prototype vs Production Architecture

**This prototype runs with zero credentials.** No API keys, no database, no
network. Everything below is either computed locally from real data committed to
the repository, or explicitly mocked and labelled as such in the UI.

### What runs for real

| Component | Status | Source |
|-----------|--------|--------|
| Groundwater forecast | **Real** | PyTorch LSTM (2-layer, MC-dropout), 0.0722 m val RMSE, beats persistence/seasonal-naive/linear-trend |
| Crop water balance | **Real** | FAO-56 dual-Kc, calibrated to CGWB observations |
| Well network & aquifer strata | **Real** | CGWB station GeoJSON, litholog-derived strata |
| Soil, aquifer, crop-mask rasters | **Real** | GeoTIFFs in `data/rasters/`, sampled at real coordinates |
| Rainfall & temperature | **Real** | India-WRIS gauges + IMD station telemetry |
| Crop economics | **Computed** | Calibrated to a single constant shared with the credit engine |
| Emissions | **Estimated** | Published IPCC/NAAS factors, split biogenic vs warming-equivalent |
| Subsidy rates | **Published** | PMKSY-PDMC, PM-KUSUM, SMAM, KCC, NABARD |

### What is mocked, and where to swap it

Every row below is a single-module change. The dashboards import the function,
never the dependency.

| Production dependency | This prototype | Swap point |
|----------------------|----------------|------------|
| PostgreSQL + PostGIS | Local GeoJSON / GeoPackage in `data/` | `src/spatial_loader.py` |
| Google Earth Engine (Sentinel-2 NDVI) | Phenology model anchored to PAU crop calendar | `src/satellite.py :: field_timeseries()` |
| Twilio / WhatsApp Business API | Rendered handset; message staged locally | `pages/2_📱_Farmer_Phone.py` |
| IVR / Punjabi TTS | Not implemented; documented as planned | `src/advisor.py` |
| Lender KYC + credit bureau | Deterministic `credit_risk_engine` on public priors | `src/credit_risk_engine.py :: score()` |
| IoT soil-moisture probes | FAO-56 estimated from weather | `src/dataset_generator.py` |
| OR-Tools CP-SAT / MILP optimiser | Vectorised enumeration of 12,584 plans (~0.4 s) | `src/optimizer.py :: optimize()` |
| XGBoost / LightGBM crop ranker | Deterministic scored rule set | `src/advisor.py :: recommend()` |
| LLM (GPT/Claude) narration | Rule templates; disabled without a key | `src/advisor.py :: llm_narrate()` |
| Carbon registry & issuance | Estimates with a stated haircut; nothing issued | `src/impact.py` |
| Subsidy disbursement (DBT) | Published rates applied arithmetically | `src/climate_finance.py` |

### The zero-credentials guarantee

```bash
streamlit run app.py        # works, no .env, no network, no keys
```

Optional keys are read from the environment and degrade gracefully:

| Variable | Used by | Without it |
|----------|---------|------------|
| `AQUACAST_LLM_KEY` | `advisor.llm_narrate()` | Deterministic engine only; UI says so |
| `AQUACAST_GEE_SERVICE_ACCOUNT` | satellite NDVI | Phenology model; UI labels it simulated |
| `AQUACAST_DB_DSN` | spatial loading | Local `data/` files |
| `AQUACAST_TWILIO_SID` | message delivery | Local staging only |

See `.env.example`.

### What this prototype does not do

* It does not sanction, approve or offer credit. No KYC, no bureau pull.
* It does not issue, register or sell a carbon credit. Five MRV steps are
  labelled in-app; **one is performed** (modelling) and **four are not**.
* It does not verify anything by measurement. Every figure is modelled, with a
  Monte Carlo range shown alongside.
* It does not send any message. Delivery is simulated.

### A deliberate deviation from the original brief

The build brief specified a React/TypeScript frontend with a Node service. This
implementation **kept Python + Streamlit**, because the trained PyTorch LSTM,
the aquifer model and modules A–F are all Python — a React rewrite would have
required a Python backend anyway, so the rewrite would have added a network hop
and a second language without adding capability. The brief invited sensible
engineering decisions; this one is flagged here rather than hidden. The
separation that mattered is preserved: every module exposes a pure function, so
a React frontend could consume exactly the same interfaces over an API without
touching any model code.

---

## 8. Quick start

```bash
pip install -r requirements.txt          # CPU torch is enough (52 k parameters)
python run_pipeline.py                   # pipeline → metrics → streamlit
python run_pipeline.py --no-app          # pipeline only
python run_pipeline.py --retrain --epochs 60
python run_pipeline.py --force-data      # rebuild the hybrid time-series
python tests/smoke_test.py               # 25 headless checks (~30 s)
streamlit run app.py                     # the full console (5 destinations)

# rebuild the pitch deck from live artefacts (figures → facts → .pptx)
python scripts/make_figures.py           # 13 chart PNGs -> reports/figures/
python scripts/deck_facts.py             # every quoted number -> reports/deck_facts.json
python scripts/build_deck.py             # -> reports/AquaCast-Punjab.pptx (21 slides)
```
The deck contains no typed-in numbers: `deck_facts.py` computes them from the
fitted model and `make_figures.py` draws the charts from `data/`. Change the data,
re-run those three commands, and every figure in the deck updates.

The trained weights and generated datasets are already committed, so
`streamlit run app.py` works with zero setup time.

---

## 9. Data provenance & caveats

* **Raw WRIS exports** (`data/raw_telemetry/`) are India-WRIS / CGWB portal downloads,
  shared by the team. The three rainfall gauges (Babanpur, Bald Kothi, Bugra Head) sit
  inside Sangrur; the temperature telemetry comes from Ludhiana, Mansa, Muktsar,
  Ferozpur and Pathankot and is inverse-distance blended onto the district centroid.
* **Portal quirks handled:** the `Rainfall_*.xlsx` files contain *two* interleaved
  series — a running daily accumulator (double-counts ~12×) and the true hourly
  increments; `Temperature_Bugra head.xlsx` is mislabelled by the portal and actually
  contains a third rainfall gauge; several boundary GeoJSONs are stored `[lat, lon]`
  instead of `[lon, lat]`. All three are detected and handled in code.
* **Gap filling** uses a Richardson-type generator whose parameters (monthly Markov
  wet/dry transitions, gamma wet-day depths, harmonic temperature climatology, AR(1)
  residual noise) are fitted to the real Punjab telemetry — 1 540 of 1 828 rainfall
  days and 1 070 of 1 828 temperature days are measured, the rest are modelled.
* **Only 31 water-level labels exist.** The district trajectory is therefore a physics
  reconstruction, not an observation; its *shape* is validated against those 31 points,
  and the LSTM is trained on the reconstruction. This is stated plainly on the Model
  Lab tab — we would rather be honest than look better than we are.
* The national India-WRIS boundary/raster downloads (`data_raw/`) were clipped to
  Punjab to keep the repository at ~26 MB. Re-download them from the portal (or the
  team Drive folder) and run `python scripts/build_spatial_assets.py` to rebuild.

### 7.1 Data pedigree — what is real, what is modelled, what is a scenario knob

Every quantity in the system is one of four kinds. We label them everywhere they
appear, because a number that looks authoritative and is not is worse than no number.

| Kind | Meaning | Where it appears |
|---|---|---|
| 🟢 **Measured** | Comes from a shipped dataset, unmodified | Rainfall, temperature, 31 CGWB water-level readings, 122 well locations, 305 lithologs (T, S, EC), district polygons, all 8 rasters |
| 🔵 **Derived** | Computed from measured inputs by a stated formula | ET₀ (Hargreaves), recharge (gamma-lag routing), aquifer T = K·b, local bore density = `prior.tubewells / net sown km²`, GRACE cell area, ₹/ft deepening cost |
| 🟡 **Reconstructed** | Physics model calibrated against the measured points | The daily water-level trajectory. Only 31 real labels exist; the shape is validated against them and the LSTM trains on the reconstruction. Disclosed on the Model Lab tab. |
| 🟠 **Scenario knob** | An assumption the user can dial, never presented as fact | Zaid (summer) crop area share, credit-score weights, the saline-depth uplift |

**Modelled-but-labelled:** the synthetic village tubewell network in Module A. Bore
*positions* are synthetic (seeded rejection sampling inside the real district polygon,
kept on the real Rabi wheat mask, with cluster centres weighted toward the real
monitoring stations because CGWB sites piezometers where the pumps are). Every
property assigned to them — transmissivity, storativity, pumping rate — is derived
from real rasters and priors. The network is reported as a **0.7–0.8 % sample** of the
district's real tubewell count, so its interference figures are a lower bound.

### 7.2 Three places where the data contradicted the brief

We built what we were asked to build, and then reported what the data actually said.

1. **"Monsoon flooding depletes the aquifer before November wheat sowing."** It does
   not. The detrended seasonal cycle has its **deepest point in July** and its
   shallowest in **December**. The monsoon *refills* the aquifer; it is the Rabi wheat
   season that drains it into the summer trough. The honest version is stronger: the
   monsoon refills but does not restore — each November starts ~0.65 m deeper than the
   last. (`crop_cycle_model.seasonal_trough`)
2. **"Deeper water is saline."** Not in this dataset. EC correlates with drilled depth
   at **r = +0.02** across 96 real lithologs. 7 % of logs exceed 2 000 µS/cm, but they
   are not concentrated in deep bores. Deep salinity is flagged as a literature risk
   and labelled as such, never as a finding. (`aquifer_strata.salinity_risk`)
3. **10 m Sentinel-2 hyper-local resolution.** Our shipped rasters are **384 m**, not
   10 m. That is still a 269× linear improvement on a 1° GRACE cell, but we report the
   resolution we actually hold. (`benchmarks.our_raster_cell_m`)

---

## 10. Repository map

```
├── app.py                         Streamlit dashboard · role switcher → 3 views
├── run_pipeline.py               single entrypoint
├── requirements.txt
├── scripts/setup_env.sh          repeatable installer (torch CPU wheel via TMPDIR)
├── src/
│   ├── config.py                 paths, district priors, hyper-parameters, scenarios
│   ├── spatial_loader.py         MODULE 1  spatial ingestion & crop baseline
│   ├── telemetry_ingest.py       MODULE 2a WRIS/CGWB telemetry ingestion + QC
│   ├── dataset_generator.py      MODULE 2b hybrid time-series synthesis
│   ├── model_lstm.py             MODULE 3  PyTorch LSTM + baselines + explainability
│   ├── advisory_engine.py        MODULE 4  zones, quotas, dry-out, messaging
│   ├── risk_engine.py            micro-finance risk (1-100) → PD → expected loss
│   ├── digital_twin.py           what-if simulator
│   ├── well_network.py           district → 122 wells downscaling
│   ├── crop_cycle_model.py       MODULE B  full-year rotation (Rabi/Zaid/Kharif)
│   ├── aquifer_strata.py         MODULE C  3-tier strata, ₹/ft, cross-section
│   ├── spatial_network.py        MODULE A  Cooper–Jacob interference network
│   ├── network_map.py            MODULE A2 interactive Leaflet map (arcs + cones)
│   ├── credit_risk_engine.py     MODULE D  0-100 KCC score, drip vs deepening
│   ├── benchmarks.py             MODULE F  vs GRACE / India-WRIS / PAU
│   ├── personas.py               MODULE E  kisan / farmer / admin views
│   ├── i18n.py                   English / ਪੰਜਾਬੀ (Gurmukhi) copy
│   └── viz.py                    every Plotly figure
├── scripts/
│   ├── build_spatial_assets.py   clip national WRIS vectors/rasters down to Punjab
│   ├── make_figures.py           13 deck charts, drawn from live data
│   ├── deck_facts.py             every number the deck quotes, computed
│   └── build_deck.py             21-slide .pptx assembled from the two above
├── tests/smoke_test.py           25 assertions, incl. regressions for the audit fixes and the honesty checks
├── reports/
│   ├── AquaCast-Punjab.pptx      the pitch deck
│   ├── network_map.html          standalone Leaflet map — opens in any browser
│   ├── deck_facts.json           the numbers, as data
│   └── figures/                  the 13 charts
├── data/                         26 MB of Punjab-clipped WRIS/CGWB assets
│   ├── raw_telemetry/            the original portal exports
│   ├── processed/                generated daily time-series + fitted climate model
│   └── rasters/                  8 reprojected Punjab rasters (384 m)
└── models/lstm_aquifer.pth       trained weights (+ .json model card)
```

---

*Built for Sankalp. Every number on the dashboard is computed live from the shipped
data — no hardcoded strings, no mocked charts.*
