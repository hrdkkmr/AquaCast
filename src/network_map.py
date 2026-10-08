"""
Module A (part 2) — Interactive tubewell-network map
====================================================

A Leaflet map of the district's tubewell field, its real CGWB monitoring
piezometers, the village clusters, the real bore logs, and — the part that
matters — the **dependency arcs** joining each bore to the piezometer it
actually moves.

Click any feature and you get:

* **a tubewell** — its village cluster, block, discharge, transmissivity, the
  three drawdown-cone radii, and the piezometers it is answerable for;
* **a piezometer** — the superposed drawdown on it, how many bores are pulling
  on it, and the five that contribute most, with distances;
* **a litholog** — the measured transmissivity, storage coefficient, drilled
  depth and electrical conductivity from the real CGWB bore log.

Design notes
------------
**Arcs are directed and thresholded.** Drawing all 690 x 164 couplings would be
a hairball; we draw only the strongest, with opacity and width scaled by the
actual metres of drawdown. Every arc drawn is a real entry of the influence
matrix — nothing is decorative.

**The cone overlay is pre-computed, not drawn live.** Leaflet cannot run our
Cooper–Jacob solver, so the cone radii are computed here in Python from each
bore's own transmissivity and storativity and emitted as circles on a
toggleable layer for the most influential bores.

**Levels are downscaled, and labelled as such.** There is no per-piezometer
measurement in the shipped data. The level shown at a piezometer is the district
level adjusted by that point's deviation on the real depth-to-first-aquifer
raster, which is a spatial downscaling of a district quantity — not an
observation. The popup says so.
"""
from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd

# Cluster palette — distinct hues, all legible on the dark basemap.
CLUSTER_COLOURS = ["#38BDF8", "#A78BFA", "#34D399", "#FBBF24", "#FB7185",
                   "#22D3EE", "#F472B6", "#A3E635", "#FDBA74", "#818CF8",
                   "#2DD4BF", "#FCA5A5"]

EARTH_M_PER_DEG = 111_320.0


def _cluster_colour(cluster: str) -> str:
    try:
        idx = int(str(cluster).lstrip("C"))
    except Exception:
        idx = hash(str(cluster))
    return CLUSTER_COLOURS[idx % len(CLUSTER_COLOURS)]


def drawdown_colour(m: float) -> str:
    """Green -> amber -> red as superposed drawdown grows."""
    if m <= 0.05:
        return "#22C55E"
    if m <= 0.20:
        return "#84CC16"
    if m <= 0.40:
        return "#FBBF24"
    if m <= 0.70:
        return "#F97316"
    return "#EF4444"


# ---------------------------------------------------------------------------
def piezometer_levels(net, district: str, district_level_mbgl: float) -> np.ndarray:
    """
    Downscale the district water level onto each piezometer.

    Uses the real depth-to-first-aquifer raster: a piezometer sitting where the
    first aquifer is deeper than the district average is assigned a
    proportionally deeper water table. This is a **downscaling of a district
    quantity**, not a measurement — there is no per-station level in the
    shipped data, and the map says so rather than inventing one.
    """
    from .spatial_loader import sample_rasters

    st = net.stations
    if not len(st):
        return np.array([])
    try:
        import geopandas as gpd
        pts = gpd.GeoDataFrame(
            st[["lon", "lat"]].reset_index(drop=True),
            geometry=gpd.points_from_xy(st.lon.to_numpy(), st.lat.to_numpy()),
            crs="EPSG:4326")
        s = sample_rasters(pts, keys=["depth_first_aquifer_m"])
        depth = pd.to_numeric(s["depth_first_aquifer_m"], errors="coerce")
        depth = depth.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        if not np.isfinite(depth).any():
            depth = None
    except Exception:
        depth = None
    if depth is None:
        return np.full(len(st), float(district_level_mbgl))

    # Robust: the raster carries cells out to 216 m, and a single outlier would
    # otherwise swing a piezometer's level by tens of metres. Winsorise at the
    # 5th/95th percentile of this district's own stations first.
    lo, hi = np.nanpercentile(depth, [5, 95])
    depth = np.clip(depth, lo, hi)

    dev = depth - float(np.nanmean(depth))
    dev = np.nan_to_num(dev, nan=0.0)
    # Half the deviation propagates to the water table — the raster describes
    # the top of the saturated zone, and the water table inside it moves less
    # than the zone boundary does. Capped at +/-4 m so a downscaling of a
    # district value can never masquerade as a local measurement.
    dev = np.clip(0.5 * dev, -4.0, 4.0)
    return np.clip(float(district_level_mbgl) + dev, 1.0, None)


# ---------------------------------------------------------------------------
def build_network_map(net, district: str,
                      district_level_mbgl: float = 32.0,
                      decline_m_per_yr: float = 0.80,
                      n_arcs: int = 140,
                      n_cones: int = 45,
                      show_arcs: bool = True,
                      show_cones: bool = True,
                      show_lithology: bool = True,
                      show_heat: bool = True) -> "object":
    """Assemble the Leaflet map. Returns a ``folium.Map``."""
    import folium
    from folium.plugins import HeatMap

    wells, st = net.wells, net.stations
    if not len(st):
        raise ValueError("no monitoring piezometers for this district")

    levels = piezometer_levels(net, district, district_level_mbgl)
    if "level_mbgl" not in st.columns:
        st = st.copy()
    st["level_mbgl"] = levels

    clat = float(np.mean([wells.lat.mean(), st.lat.mean()]))
    clon = float(np.mean([wells.lon.mean(), st.lon.mean()]))

    fmap = folium.Map(location=[clat, clon], zoom_start=10, tiles=None,
                      control_scale=True, prefer_canvas=True)
    folium.TileLayer("CartoDB positron", name="Light basemap",
                     control=True).add_to(fmap)
    folium.TileLayer("CartoDB dark_matter", name="Dark basemap",
                     control=True).add_to(fmap)

    # ---- 1. district boundary ------------------------------------------
    try:
        from .spatial_loader import district_geojson
        gj = district_geojson(district)
        if gj:
            folium.GeoJson(
                gj, name=f"{district} boundary",
                style_function=lambda _f: dict(fillColor="#0EA5E9", color="#0369A1",
                                               weight=2.2, fillOpacity=0.06),
                tooltip=folium.Tooltip(f"{district} district boundary"),
            ).add_to(fmap)
    except Exception:
        pass

    # ---- 2. village tubewell clusters ----------------------------------
    wlayer = folium.FeatureGroup(name=f"Tubewells — {len(wells):,} bores, "
                                      f"{net.meta.get('n_clusters', 0)} villages",
                                 show=True)
    infl = net.influence
    for i, r in wells.iterrows():
        own = float(infl[i].max()) if i < infl.shape[0] else 0.0
        col = _cluster_colour(getattr(r, "cluster", "C0"))
        txt = _well_popup(r, net, i, own)
        folium.CircleMarker(
            [float(r.lat), float(r.lon)], radius=3.0, color=col, weight=1,
            fill=True, fill_color=col, fill_opacity=0.85,
            tooltip=folium.Tooltip(f"{getattr(r,'cluster','-')} · "
                                   f"{_html.escape(str(getattr(r,'block','-')))}"),
            popup=folium.Popup(txt, max_width=340),
        ).add_to(wlayer)
    wlayer.add_to(fmap)

    # ---- 3. real CGWB piezometers --------------------------------------
    slayer = folium.FeatureGroup(
        name=f"Piezometers — {len(st)} real CGWB stations", show=True)
    for j, r in st.iterrows():
        dd = float(r.get("cumulative_drawdown_m", 0.0) or 0.0)
        lvl = float(r.get("level_mbgl", np.nan))
        col = drawdown_colour(dd)
        rad = 4.0 + min(9.0, 3.2 * np.log1p(dd * 40.0))
        folium.CircleMarker(
            [float(r.lat), float(r.lon)], radius=rad, color="#0B1E30",
            weight=1.2, fill=True, fill_color=col, fill_opacity=0.92,
            tooltip=folium.Tooltip(f"{_html.escape(str(r.get('station','—')))} — "
                                   f"{dd:.2f} m superposed drawdown"),
            popup=folium.Popup(_station_popup(r, net, j, decline_m_per_yr),
                               max_width=380),
        ).add_to(slayer)
    slayer.add_to(fmap)

    # ---- 4. dependency arcs (directed, thresholded) ---------------------
    if show_arcs and infl.size:
        alayer = folium.FeatureGroup(name="Dependency arcs — bore → piezometer",
                                     show=True)
        try:
            deps = net.top_dependencies(k=int(n_arcs))
        except Exception:
            deps = pd.DataFrame()
        mx = float(deps["drawdown_m"].max()) if len(deps) else 1.0
        for _, d in deps.iterrows():
            wi, si = int(d["wi"]), int(d["si"])
            if wi >= len(wells) or si >= len(st):
                continue
            w, s = wells.iloc[wi], st.iloc[si]
            v = float(d["drawdown_m"])
            frac = v / max(mx, 1e-9)
            folium.PolyLine(
                [(float(w.lat), float(w.lon)), (float(s.lat), float(s.lon))],
                color="#EF4444", weight=0.8 + 2.6 * frac,
                opacity=0.16 + 0.52 * frac, dash_array=None,
                tooltip=folium.Tooltip(f"{v:.3f} m · {float(d['dist_m']):,.0f} m apart"),
            ).add_to(alayer)
        alayer.add_to(fmap)

    # ---- 5. drawdown cone overlay ---------------------------------------
    if show_cones and infl.size and len(wells):
        clayer = folium.FeatureGroup(
            name="Drawdown cones — top bores by influence", show=False)
        strongest = np.asarray(infl.max(axis=1)).ravel()
        order = np.argsort(strongest)[::-1][:int(n_cones)]
        for wi in order:
            r = wells.iloc[int(wi)]
            try:
                from .spatial_network import single_well_cone
                cone = single_well_cone(district, float(r.lat), float(r.lon),
                                        float(r.get("Q_m3_day", 200.0)))
                radii = cone["radii_m"]
            except Exception:
                radii = {}
            for label, rm in list(radii.items())[:3]:
                if not np.isfinite(rm) or rm <= 0:
                    continue
                folium.Circle(
                    [float(r.lat), float(r.lon)], radius=float(rm),
                    color="#B3261E", weight=1.0, fill=True,
                    fill_color="#EF4444", fill_opacity=0.05,
                    tooltip=folium.Tooltip(f"{label} cone ≈ {rm:,.0f} m"),
                ).add_to(clayer)
            folium.CircleMarker(
                [float(r.lat), float(r.lon)], radius=6.5, color="#B3261E",
                weight=1.6, fill=True, fill_color="#EF4444", fill_opacity=0.35,
                tooltip=folium.Tooltip(
                    f"cone centre · {float(strongest[int(wi)]):.3f} m peak drawdown"),
            ).add_to(clayer)
        clayer.add_to(fmap)

    # ---- 6. real lithologs ----------------------------------------------
    if show_lithology:
        try:
            from .aquifer_strata import _litholog_raw
            lg = _litholog_raw()
            lg = lg[pd.to_numeric(lg.get("Latitude"), errors="coerce").notna()]
            lg = lg[lg["District"].astype(str).str.strip().str.lower()
                    == district.lower()] if len(lg) else lg
            if len(lg):
                llayer = folium.FeatureGroup(
                    name=f"Lithologs — {len(lg)} real CGWB bore logs", show=True)
                for _, r in lg.iterrows():
                    folium.Marker(
                        [float(r["Latitude"]), float(r["Longitude"])],
                        icon=folium.Icon(color="purple", icon="info-sign"),
                        popup=folium.Popup(_lith_popup(r), max_width=340),
                        tooltip=folium.Tooltip("CGWB litholog"),
                    ).add_to(llayer)
                llayer.add_to(fmap)
        except Exception:
            pass

    # ---- 7. drawdown heat layer ------------------------------------------
    if show_heat and len(st):
        try:
            hlayer = folium.FeatureGroup(name="Drawdown heatmap", show=False)
            pts = [[float(r.lat), float(r.lon),
                    float(r.get("cumulative_drawdown_m", 0.0))]
                   for _, r in st.iterrows()]
            mx = max(p[2] for p in pts) or 1.0
            HeatMap([[p[0], p[1], p[2] / mx] for p in pts],
                    radius=26, blur=20, max_zoom=12,
                    gradient={0.2: "#38BDF8", 0.5: "#FBBF24", 0.8: "#F97316",
                              1.0: "#EF4444"}).add_to(hlayer)
            hlayer.add_to(fmap)
        except Exception:
            pass

    folium.LayerControl(collapsed=False).add_to(fmap)
    _legend(fmap, net)
    return fmap


# ---------------------------------------------------------------------------
def _well_popup(r, net, wi: int, own_max: float) -> str:
    from .spatial_network import WELL_RADIUS_M
    try:
        from .spatial_network import single_well_cone
        cone = single_well_cone(net.district, float(r.lat), float(r.lon),
                                float(r.get("Q_m3_day", 200.0)))
        radii = "".join(
            f"<tr><td>{k}</td><td style='text-align:right'>{v:,.0f} m</td></tr>"
            for k, v in list(cone["radii_m"].items())[:3])
        self_dd = f"{cone['self_drawdown_m']:.2f} m"
    except Exception:
        radii, self_dd = "", "—"
    infl = net.influence
    top = []
    if infl.size and wi < infl.shape[0]:
        row = infl[wi]
        for si in np.argsort(row)[::-1][:3]:
            if row[si] > 1e-4 and si < len(net.stations):
                s = net.stations.iloc[si]
                dkm = np.hypot((s.lat - r.lat) * 111.0, (s.lon - r.lon) * 103.0)
                top.append(f"<tr><td>{_html.escape(str(s.get('station','—')))}"
                           f"</td><td style='text-align:right'>{row[si]:.3f} m"
                           f"</td><td style='text-align:right'>{dkm*1000:,.0f} m</td></tr>")
    return f"""
<div style="font-family:system-ui;width:300px">
<b style="font-size:13px">Tubewell · {_html.escape(str(getattr(r,'cluster','—')))}</b><br>
<span style="color:#64748B;font-size:11px">
{_html.escape(str(getattr(r,'block','—')))} block · position is synthetic,
properties are measured</span>
<hr style="margin:6px 0">
<table style="width:100%;font-size:11.5px">
<tr><td>Discharge</td><td style="text-align:right"><b>{float(r.get('pump_m3h',0)):.1f} m³/h</b>
  ({float(r.get('Q_m3_day',0)):.0f} m³/d)</td></tr>
<tr><td>Transmissivity T</td><td style="text-align:right">{float(r.get('T_m2_day',0)):,.0f} m²/d</td></tr>
<tr><td>Storativity S</td><td style="text-align:right">{float(r.get('S',0)):.3f}</td></tr>
<tr><td>Saturated thickness</td><td style="text-align:right">{float(r.get('saturated_thickness_m',0)):.0f} m</td></tr>
<tr><td>Drawdown in its own bore</td><td style="text-align:right"><b>{self_dd}</b></td></tr>
<tr><td>Strongest effect anywhere</td><td style="text-align:right"><b>{own_max:.3f} m</b></td></tr>
</table>
<b style="font-size:11.5px">Drawdown cone radius</b>
<table style="width:100%;font-size:11.5px">{radii}</table>
<b style="font-size:11.5px">Piezometers it moves most</b>
<table style="width:100%;font-size:11.5px">
<tr style="color:#64748B"><td>station</td><td style="text-align:right">m</td>
<td style="text-align:right">distance</td></tr>
{''.join(top) or '<tr><td colspan=3>below threshold</td></tr>'}
</table>
</div>"""


def _station_popup(r, net, si: int, decline_m_per_yr: float) -> str:
    dd = float(r.get("cumulative_drawdown_m", 0.0) or 0.0)
    contrib = []
    infl = net.influence
    if infl.size and si < infl.shape[1]:
        col = infl[:, si]
        for wi in np.argsort(col)[::-1][:5]:
            if col[wi] > 1e-4 and wi < len(net.wells):
                w = net.wells.iloc[wi]
                dkm = np.hypot((w.lat - r.lat) * 111.0, (w.lon - r.lon) * 103.0)
                contrib.append(
                    f"<tr><td>{_html.escape(str(w.get('cluster','—')))} · "
                    f"{_html.escape(str(w.get('block','—')))}</td>"
                    f"<td style='text-align:right'>{col[wi]:.3f} m</td>"
                    f"<td style='text-align:right'>{dkm*1000:,.0f} m</td></tr>")
    lvl = float(r.get("level_mbgl", float("nan")))
    yrs = (dd / max(decline_m_per_yr, 1e-6)) if decline_m_per_yr else float("nan")
    return f"""
<div style="font-family:system-ui;width:330px">
<b style="font-size:13px">{_html.escape(str(r.get('station','Piezometer')))}</b><br>
<span style="color:#64748B;font-size:11px">real CGWB monitoring station ·
{_html.escape(str(r.get('source','—')))}</span>
<hr style="margin:6px 0">
<table style="width:100%;font-size:11.5px">
<tr><td>Superposed drawdown</td><td style="text-align:right">
  <b style="color:{drawdown_colour(dd)}">{dd:.3f} m</b></td></tr>
<tr><td>Bores pulling on it</td><td style="text-align:right">
  {int(r.get('n_influencing_wells', 0)):,}</td></tr>
<tr><td>Downscaled water level</td><td style="text-align:right">
  {lvl:.1f} m bgl</td></tr>
<tr><td>District decline</td><td style="text-align:right">
  −{decline_m_per_yr:.2f} m/yr</td></tr>
<tr><td>This season's interference, in years of decline</td>
    <td style="text-align:right"><b>{yrs:.2f} yr</b></td></tr>
</table>
<b style="font-size:11.5px">Biggest contributors</b>
<table style="width:100%;font-size:11.5px">
<tr style="color:#64748B"><td>bore</td><td style="text-align:right">m</td>
<td style="text-align:right">distance</td></tr>
{''.join(contrib) or '<tr><td colspan=3>below threshold</td></tr>'}
</table>
<span style="color:#94A3B8;font-size:10px">Level is the district value
downscaled by the depth-to-first-aquifer raster — a downscaling, not a
measurement.</span>
</div>"""


def _lith_popup(r) -> str:
    def g(k):
        v = r.get(k)
        return v if pd.notna(v) else None
    rows = [("Drilled depth", g("Depth_of_Drill__m_"), " m"),
            ("Static water level", g("Static_Water_Level__mbgl_"), " m bgl"),
            ("Transmissivity", g("Transmissivity__m2_day_"), " m²/d"),
            ("Storage coefficient", g("Storage_coefficient"), ""),
            ("Discharge", g("Discharge__lps_"), " lps"),
            ("Electrical conductivity",
             g("Electrical_Conductivity__Micro_Seimens__cm_"), " µS/cm"),
            ("Chloride", g("Chloride__mg_l_"), " mg/l"),
            ("Major lithology", g("Major_Lithology"), "")]
    body = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'><b>{v}{u}</b></td></tr>"
        for k, v, u in rows if v is not None and str(v).strip() != "")
    return f"""
<div style="font-family:system-ui;width:300px">
<b style="font-size:13px">CGWB litholog</b><br>
<span style="color:#64748B;font-size:11px">
{_html.escape(str(g('Location') or g('Block') or '—'))}</span>
<hr style="margin:6px 0">
<table style="width:100%;font-size:11.5px">{body}</table>
<span style="color:#94A3B8;font-size:10px">Measured bore log — nothing here is
modelled.</span>
</div>"""


def _legend(fmap, net) -> None:
    m = net.meta or {}
    rows = "".join(
        f"<tr><td style='width:14px'><span style='display:inline-block;width:11px;"
        f"height:11px;background:{c};border-radius:2px'></span></td>"
        f"<td style='font-size:11px'>{lab}</td></tr>"
        for c, lab in [("#22C55E", "&lt; 0.05 m"), ("#84CC16", "0.05–0.20 m"),
                       ("#FBBF24", "0.20–0.40 m"), ("#F97316", "0.40–0.70 m"),
                       ("#EF4444", "&gt; 0.70 m")])
    folium = __import__("folium")
    folium.map.Marker(
        [0, 0], icon=folium.DivIcon(html=f"""
<div style="position:absolute;left:0;top:0;width:168px;
  background:rgba(255,255,255,.94);border:1px solid #CBD5E1;border-radius:8px;
  padding:8px 10px;font-family:system-ui;box-shadow:0 2px 8px rgba(0,0,0,.18)">
<b style="font-size:11px">Superposed drawdown</b>
<table style="border-collapse:collapse">{rows}</table>
<hr style="margin:6px 0">
<span style="font-size:10px;color:#475569">{len(net.wells):,} bores ·
{m.get('n_clusters', 0)} villages<br>{len(net.stations)} real piezometers<br>
{m.get('bores_per_cluster', 0)} bores/village ·
{m.get('local_bore_density_per_km2', 0):.0f}/km²<br>
max {m.get('max_cumulative_drawdown_m', 0):.2f} m ·
{m.get('sample_fraction', 0)*100:.1f}% sample</span>
</div>""")).add_to(fmap)
