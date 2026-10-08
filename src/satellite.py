"""
Satellite & field-observation layer
===================================

What is real here and what is not
---------------------------------
This module sits closest to the line between real and simulated, so it states
the boundary explicitly on every screen that uses it.

**Real — computed from rasters shipped in the repository.** Soil texture, soil
depth, slope, erosion and productivity, aquifer thickness and first-aquifer
depth, and the Rabi wheat mask. These are sampled from GeoTIFFs under
``data/rasters/`` at the actual farm or district location, so the numbers change
with location.

**Simulated — clearly labelled wherever shown.** There is no Sentinel-2 archive
in this repository, so the *seasonal greenness curve* is a phenology model: a
double-logistic curve fitted to each crop's published sowing, peak and harvest
dates. It has the right shape and the right timing; it is not a measurement of
a specific field. In production this is replaced by a Sentinel-2 NDVI time
series (10 m, 5-day revisit) pulled from Google Earth Engine, and the swap is a
one-function change — ``field_timeseries()`` keeps its signature.

**Not present at all — mocked in the UI.** Satellite tasking, cloud-cover
screening, and parcel-boundary (creep-measurement) digitisation. The UI shows
these as planned production integrations rather than pretending they run.

Function contract
-----------------
    field_attributes(lat, lon) -> dict     # real, from rasters
    field_timeseries(allocation, ...)      # simulated phenology
    district_greenness(district)           # real mask + simulated curve
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Real layers available on disk.
RASTER_KEYS = ("soil_texture", "soil_depth", "soil_slope", "soil_erosion",
               "soil_productivity", "rabi_wheat_mask",
               "depth_first_aquifer_m", "aquifer_thickness_m")

#: Human labels and units for the real raster layers.
RASTER_LABELS = {
    "soil_texture": ("Soil texture class", ""),
    "soil_depth": ("Soil depth", "cm"),
    "soil_slope": ("Slope", "%"),
    "soil_erosion": ("Erosion risk", ""),
    "soil_productivity": ("Productivity index", ""),
    "rabi_wheat_mask": ("Rabi wheat mask", "0/1"),
    "depth_first_aquifer_m": ("First aquifer depth", "m"),
    "aquifer_thickness_m": ("Aquifer thickness", "m"),
}

SIMULATED_NOTE = ("Simulated — Sentinel-2 NDVI not available in this "
                  "prototype. Production: GEE collection COPERNICUS/S2_SR_HARMONIZED, "
                  "10 m, 5-day revisit, cloud-masked.")


# ---------------------------------------------------------------------------
@dataclass
class FieldAttributes:
    lat: float
    lon: float
    values: dict
    missing: list

    def frame(self) -> pd.DataFrame:
        rows = [dict(Layer=RASTER_LABELS.get(k, (k,))[0],
                     **{"Value": (round(float(v), 2) if v is not None and
                                  np.isfinite(float(v)) else "no data"),
                        "Unit": RASTER_LABELS.get(k, (k, ""))[1]})
                for k, v in self.values.items()]
        return pd.DataFrame(rows)


def field_attributes(lat: float, lon: float) -> FieldAttributes:
    """
    Sample every real raster at a point. Returns ``missing`` for layers that
    fall outside the raster extent rather than silently substituting zero —
    a blank cell is honest, a fabricated one is not.
    """
    import geopandas as gpd
    from .spatial_loader import sample_rasters
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
    try:
        out = sample_rasters(gdf)
        row = out.iloc[0].to_dict() if len(out) else {}
    except Exception:
        row = {}
    vals, missing = {}, []
    for k in RASTER_KEYS:
        v = row.get(k, None)
        try:
            v = float(v) if v is not None else None
        except (TypeError, ValueError):
            v = None
        if v is None or not np.isfinite(v):
            missing.append(k)
            vals[k] = None
        else:
            vals[k] = v
    return FieldAttributes(lat=lat, lon=lon, values=vals, missing=missing)


# ---------------------------------------------------------------------------
#: Phenological anchor dates (day of year) for Punjab, from PAU package of
#: practices. These anchor the simulated curve to real agronomy even though the
#: reflectance values themselves are modelled.
PHENOLOGY = {
    "paddy":    dict(sow=175, peak=250, harvest=305, peak_ndvi=0.82),
    "wheat":    dict(sow=320, peak=60, harvest=110, peak_ndvi=0.78),
    "maize":    dict(sow=180, peak=232, harvest=268, peak_ndvi=0.80),
    "cotton":   dict(sow=140, peak=245, harvest=330, peak_ndvi=0.74),
    "pulses":   dict(sow=95,  peak=140, harvest=175, peak_ndvi=0.62),
    "oilseeds": dict(sow=310, peak=50, harvest=105, peak_ndvi=0.70),
}
BARE_NDVI = 0.14


def _double_logistic(doy: np.ndarray, sow: int, peak: int, harvest: int,
                     peak_ndvi: float) -> np.ndarray:
    """Asymmetric logistic rise and fall — the standard crop curve shape."""
    s, p, h = float(sow), float(peak), float(harvest)
    rise = 1.0 / (1.0 + np.exp(-0.09 * (doy - (s + p) / 2.0)))
    fall = 1.0 / (1.0 + np.exp(0.11 * (doy - (p + h) / 2.0)))
    return BARE_NDVI + (peak_ndvi - BARE_NDVI) * np.clip(rise * fall, 0.0, 1.0)


def field_timeseries(allocation_ha: dict[str, float],
                     hectares: float = 1.0,
                     noise: float = 0.012) -> pd.DataFrame:
    """
    Simulated seasonal greenness curve, area-weighted across the crop mix.

    The shape and the timing come from the crop calendar; the values are
    modelled, not observed. ``noise`` adds a small deterministic-looking
    wobble so the trace does not look machine-drawn — it is still simulated.
    """
    doy = np.arange(1, 366)
    # A field holds ONE crop at a time — the kharif and rabi crops are
    # sequential on the same land, not simultaneous on different halves. So the
    # curve is the envelope of the individual crops, not their area-weighted
    # average. Averaging halves every peak and produces a field that is never
    # actually green.
    series = np.full_like(doy, BARE_NDVI, dtype=float)
    for k, ha in allocation_ha.items():
        ph = PHENOLOGY.get(k)
        if not ph or ha <= 0:
            continue
        series = np.maximum(series, _double_logistic(
            doy, ph["sow"], ph["peak"], ph["harvest"], ph["peak_ndvi"]))
    rng = np.random.default_rng(7)
    series = np.clip(series + rng.normal(0, noise, series.shape), 0.02, 0.95)
    return pd.DataFrame({"day_of_year": doy,
                         "date": pd.to_datetime("2026-01-01") +
                                 pd.to_timedelta(doy - 1, unit="D"),
                         "ndvi": series})


def district_greenness(district: str) -> dict:
    """
    Real crop-mask statistics for a district, plus the simulated curve.

    The mask share is measured from ``rabi_wheat_mask.tif``; the curve is not.
    """
    from .spatial_loader import district_geometry
    import geopandas as gpd
    from .spatial_loader import sample_rasters
    try:
        geom = district_geometry(district)
        c = geom.centroid
        pt = gpd.GeoDataFrame(geometry=gpd.points_from_xy([c.x], [c.y]),
                              crs="EPSG:4326")
        s = sample_rasters(pt).iloc[0].to_dict()
        wheat = float(s.get("rabi_wheat_mask", np.nan))
        depth = float(s.get("depth_first_aquifer_m", np.nan))
    except Exception:
        wheat, depth = float("nan"), float("nan")
    return dict(district=district, wheat_mask=wheat,
                first_aquifer_depth_m=depth,
                simulated=True, note=SIMULATED_NOTE)


def self_check() -> None:
    print("\n[satellite] field observation layer")
    print("-" * 78)
    fa = field_attributes(30.24, 75.84)          # Sangrur
    print(f"  sampling real rasters at {fa.lat}, {fa.lon} (Sangrur)")
    print(fa.frame().to_string(index=False))
    if fa.missing:
        print(f"  no data for: {', '.join(fa.missing)}")
    print()
    ts = field_timeseries({"paddy": 1.0, "wheat": 1.0}, hectares=1.0)
    print(f"  simulated NDVI curve (paddy–wheat): "
          f"min {ts.ndvi.min():.2f} · max {ts.ndvi.max():.2f} · "
          f"{len(ts)} days")
    print(f"  {SIMULATED_NOTE}")
    d = district_greenness("Sangrur")
    print(f"  Sangrur: wheat mask {d['wheat_mask']}, first aquifer "
          f"{d['first_aquifer_depth_m']} m (real) · curve simulated")
    print("-" * 78)


if __name__ == "__main__":
    self_check()
