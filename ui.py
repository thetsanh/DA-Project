# ui.py — PKL-powered, tabbed dashboard (smart PKL→CSV mapping, robust MAE/R²)

import os, ast, pickle, colorsys, hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# -------------------- CONFIG: file paths --------------------
DATA_DIR  = Path("data")
CLEAN_DIR = Path("cleaned")

TRAFFIC_CSV = DATA_DIR / "traffic.csv"
ROUTES_CSV  = CLEAN_DIR / "cleaned_bus_routes_file.csv"
STOPS_CSV   = CLEAN_DIR / "cleaned_bus_stops_file.csv"
ZONES_CSV   = CLEAN_DIR / "congestion_zones.csv"

ROUTE_MODELS_PKL    = DATA_DIR / "route_models.pkl"
FEATURE_COLUMNS_PKL = DATA_DIR / "feature_columns.pkl"

# -------------------- PAGE LOOK --------------------
st.set_page_config(page_title="Bangkok Bus Insights", layout="wide")
st.markdown("""
<style>
h1,h2,h3 {font-weight:800; letter-spacing:.2px}
.card{background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
      border-radius:16px; padding:16px 18px; box-shadow:0 8px 24px rgba(0,0,0,.18)}
.big{font-size:40px; font-weight:800}
.dim{color:#9aa4b2}
hr{border:none; height:1px; background:rgba(255,255,255,.1); margin:12px 0}
.spacer{height:14px}
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
# ---------- Feature engineering helpers (add above build_segment_times_from_model) ----------
def _bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 (degrees 0..360)."""
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1)*np.cos(lat2)*np.cos(dlon) + np.sin(lat1)*np.sin(lat2)
    brng = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return float(brng)

def _haversine_km_pair(lat1, lon1, lat2, lon2):
    R=6371.0
    lat1=np.radians(lat1); lon1=np.radians(lon1)
    lat2=np.radians(lat2); lon2=np.radians(lon2)
    dlat=lat2-lat1; dlon=lon2-lon1
    a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def _min_dist_to_congestion(lat, lon, zones_df):
    """
    Distance (km) from (lat,lon) to the nearest zone center in zones_df.
    Returns 0 if zones_df missing/empty/columns not found.
    """
    if zones_df is None or zones_df.empty:
        return 0.0
    needed = {"center_lat","center_lon"}
    if not needed.issubset(set(zones_df.columns)):
        return 0.0
    # very light vectorized min-distance
    lat2 = zones_df["center_lat"].astype(float).to_numpy()
    lon2 = zones_df["center_lon"].astype(float).to_numpy()
    # compute haversine to all centers; take min
    R=6371.0
    la1=np.radians(lat); lo1=np.radians(lon)
    la2=np.radians(lat2); lo2=np.radians(lon2)
    dlat=la2-la1; dlon=lo2-lo1
    a = np.sin(dlat/2.0)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2.0)**2
    d = 2*R*np.arcsin(np.sqrt(a))
    return float(np.min(d)) if d.size else 0.0

def _feature_row_with_optionals(lat, lon, next_lat=None, next_lon=None, zones_df=None, t=None):
    """Base features + optional engineered ones; returns a dict."""
    base = make_feature_row(lat, lon, t=t)

    # engineered features (compute if we can, otherwise fill 0s)
    # heading: needs next point
    if next_lat is not None and next_lon is not None:
        base["heading"] = _bearing_deg(lat, lon, next_lat, next_lon)
    else:
        base["heading"] = 0.0

    # lat/lon grids (coarse bins your old model might expect)
    base["lat_grid"] = round(lat, 3)
    base["lon_grid"] = round(lon, 3)

    # distance from (approx) Bangkok center
    base["distance_from_center"] = _haversine_km_pair(lat, lon, 13.7563, 100.5018)

    # distance to nearest congestion zone center (if zones present)
    base["distance_to_congestion"] = _min_dist_to_congestion(lat, lon, zones_df)

    return base

def load_csv(path: Path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs) if path.exists() else None
    except Exception as e:
        st.warning(f"Could not read `{path}`: {e}")
        return None

def parse_coords_str(x):
    """'[[lon,lat], ...]' -> [[lat,lon], ...]"""
    try:
        pts = ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return []
    out=[]
    if isinstance(pts,(list,tuple)):
        for p in pts:
            if isinstance(p,(list,tuple)) and len(p)>=2 and pd.notna(p[0]) and pd.notna(p[1]):
                out.append([float(p[1]), float(p[0])])
    return out

def bkk_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Bangkok"))
    except Exception:
        return datetime.now()

def make_feature_row(lat, lon, t=None):
    t = t or bkk_now()
    dow  = t.weekday()
    hour = t.hour
    return {
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": 1 if dow >= 5 else 0,
        "is_rush_hour": 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0,
        "lat": lat,
        "lon": lon,
    }

def mask_bbox(df, lat_col, lon_col, coords):
    """Fast coarse filter by route bounding box (+padding); returns mask."""
    lats = [p[0] for p in coords]; lons = [p[1] for p in coords]
    pad = 0.01  # ~1.1km
    return (df[lat_col].between(min(lats)-pad, max(lats)+pad) &
            df[lon_col].between(min(lons)-pad, max(lons)+pad))

def traffic_near_route(traffic_df, lat_col, lon_col, speed_col, coords_latlon, buffer_m=300):
    """Return traffic points within ~buffer_m of the polyline."""
    if traffic_df is None or traffic_df.empty or not coords_latlon:
        return pd.DataFrame(columns=traffic_df.columns if traffic_df is not None else [])
    # coarse prefilter
    pre = traffic_df[mask_bbox(traffic_df, lat_col, lon_col, coords_latlon)].copy()
    if pre.empty:
        return pre

    # sample route for speed
    step = max(1, len(coords_latlon)//80)  # ~<=80 points
    sampled = coords_latlon[::step]
    poly_lat = np.array([p[0] for p in sampled], dtype=float)
    poly_lon = np.array([p[1] for p in sampled], dtype=float)

    # vectorized nearest distance to sampled points (haversine)
    def haversine_km(lat1, lon1, lat2, lon2):
        R=6371.0
        lat1=np.radians(lat1); lon1=np.radians(lon1)
        lat2=np.radians(lat2); lon2=np.radians(lon2)
        dlat=lat2-lat1; dlon=lon2-lon1
        a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2*R*np.arcsin(np.sqrt(a))

    dmin = np.full(len(pre), np.inf)
    for i in range(len(sampled)):
        d = haversine_km(pre[lat_col].to_numpy(), pre[lon_col].to_numpy(), poly_lat[i], poly_lon[i])
        dmin = np.minimum(dmin, d)
    mask = dmin <= (buffer_m/1000.0)
    return pre.loc[mask].copy()

def _coerce_num(x):
    try:
        return float(x)
    except Exception:
        return None

def get_metric(model_info: dict, key_variants):
    """Search MAE/R2 in several common places/names; return float or None."""
    if not isinstance(model_info, dict):
        return None
    lower_map = {k.lower(): v for k, v in model_info.items()}
    for k in key_variants:
        if k.lower() in lower_map:
            val = _coerce_num(lower_map[k.lower()])
            if val is not None:
                return val
    for container in ("metrics", "metric", "perf", "performance", "eval", "scores"):
        sub = model_info.get(container)
        if isinstance(sub, dict):
            sub_lower = {k.lower(): v for k, v in sub.items()}
            for k in key_variants:
                if k.lower() in sub_lower:
                    val = _coerce_num(sub_lower[k.lower()])
                    if val is not None:
                        return val
    return None

def color_for_ref(ref: str) -> str:
    """Unique, stable color per route id."""
    h = int(hashlib.md5(ref.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0              # 0..1
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

# ---------- Trip Finder helpers ----------
def _hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def nearest_vertex_index(coords_latlon, lat, lon):
    """Return index of the nearest vertex along a route polyline (coords as [lat,lon])."""
    if not coords_latlon:
        return None, np.inf
    arr = np.array(coords_latlon, dtype=float)  # N x 2
    d = _hav_km(arr[:,0], arr[:,1], lat, lon)
    i = int(np.argmin(d))
    return i, float(d[i])

def route_covers_od(route_row, o_lat, o_lon, d_lat, d_lon, radius_m=350):
    """Check if the route passes near origin then destination (within radius_m)."""
    coords_latlon = route_row.get("latlon", [])
    if not coords_latlon:
        return False, None, None, None, None
    oi, od_km = nearest_vertex_index(coords_latlon, o_lat, o_lon)
    di, dd_km = nearest_vertex_index(coords_latlon, d_lat, d_lon)
    if oi is None or di is None:
        return False, None, None, None, None
    if od_km*1000.0 <= radius_m and dd_km*1000.0 <= radius_m and oi < di:
        return True, oi, di, od_km, dd_km
    return False, oi, di, od_km, dd_km

def estimate_time_between_indices(ref, oi, di, routes_df, route_models, feature_columns):
    """If a model exists for `ref`, predict per-segment speed/time and sum [oi -> di)."""
    if ref not in route_models:
        return None
    seg_df, err = build_segment_times_from_model(ref, routes_df, route_models, feature_columns)
    if err or seg_df is None or seg_df.empty:
        return None
    mask = (seg_df["segment_index"] >= (oi+1)) & (seg_df["segment_index"] <= di)
    part = seg_df.loc[mask]
    if part.empty:
        return None
    return float(part["segment_travel_time_min"].sum())

# -------------------- LOAD BASE DATA --------------------
traffic = load_csv(TRAFFIC_CSV)
routes  = load_csv(ROUTES_CSV)
zones   = load_csv(ZONES_CSV)

# STOPS: load & sanitize
stops = load_csv(STOPS_CSV)
if stops is not None and not stops.empty:
    _smap = {c.lower(): c for c in stops.columns}
    _latc = _smap.get("lat")
    _lonc = _smap.get("lon")
    _namec = _smap.get("refname") or _smap.get("name") or _smap.get("stop_name")
    if _latc and _lonc:
        stops = stops.dropna(subset=[_latc, _lonc]).copy()
        stops[_latc] = pd.to_numeric(stops[_latc], errors="coerce")
        stops[_lonc] = pd.to_numeric(stops[_lonc], errors="coerce")
        stops = stops.dropna(subset=[_latc, _lonc])
    else:
        stops = None
else:
    stops = None

# normalize route geometry
if routes is not None:
    routes = routes.copy()
    routes["ref"] = routes["ref"].astype(str)
    routes["coordinates"] = routes["coordinates"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    routes["segment_distance_list"] = routes["segment_distance_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    routes["latlon"] = routes["coordinates"].apply(
        lambda pts: [[p[1], p[0]] for p in pts] if isinstance(pts, list) else []
    )

# traffic basics
lat_col, lon_col, speed_col = "lat", "lon", "speed"
if traffic is not None:
    cmap = {c.lower(): c for c in traffic.columns}
    lat_col   = cmap.get("lat", lat_col)
    lon_col   = cmap.get("lon", lon_col)
    speed_col = cmap.get("speed", speed_col)
    tcol = next((c for c in ["timestamp","time","datetime","date_time","dt"] if c in traffic.columns), None)
    if tcol:
        traffic["_ts"] = pd.to_datetime(traffic[tcol], errors="coerce")
        traffic["hour"] = traffic["_ts"].dt.hour
        traffic["day_of_week"] = traffic["_ts"].dt.dayofweek
    else:
        traffic["hour"] = 0
        traffic["day_of_week"] = 0

# -------------------- LOAD MODELS --------------------
route_models = {}
feature_columns = {}

pkl_error = None
if ROUTE_MODELS_PKL.exists():
    try:
        with open(ROUTE_MODELS_PKL, "rb") as f:
            raw = pickle.load(f)
        route_models = {str(k).strip(): v for k, v in raw.items()}
    except Exception as e:
        pkl_error = str(e)
        st.error(f"Could not load pickle {ROUTE_MODELS_PKL}: {e}")

if FEATURE_COLUMNS_PKL.exists():
    try:
        with open(FEATURE_COLUMNS_PKL, "rb") as f:
            feature_columns = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load feature columns {FEATURE_COLUMNS_PKL}: {e}")

# -------------------- SMART MAP: PKL -> CSV['ref'] --------------------
def _norm_str(x: object) -> str:
    s = str(x).strip()
    if s.replace(".", "", 1).isdigit() and s.endswith(".0"):
        s = s[:-2]
    return s.lower()

def _extract_strings(obj, out: set):
    """Recursively collect strings from dict/list/tuples (for model_info scan)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str): out.add(_norm_str(k))
            _extract_strings(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj: _extract_strings(v, out)
    elif isinstance(obj, (str, int, float, np.integer, np.floating)):
        out.add(_norm_str(obj))

mapped_models = {}
mapping_hits = []  # (pkl_key, csv_ref, matched_column)
candidate_cols = []

if routes is not None and not routes.empty and route_models:
    # include object and numeric columns as candidates
    for c in routes.columns:
        if c == "ref":
            candidate_cols.append(c)
        elif routes[c].dtype == "O" or np.issubdtype(routes[c].dtype, np.number):
            candidate_cols.append(c)

    # maps: column_name -> { normalized_value : canonical ref }
    col_lookup = {}
    for c in candidate_cols:
        try:
            col_norm = routes[c].apply(_norm_str)
            col_lookup[c] = dict(zip(col_norm, routes["ref"].astype(str)))
        except Exception:
            pass

    for pkl_key, info in route_models.items():
        bag = {_norm_str(pkl_key)}
        _extract_strings(info, bag)

        matched_ref = None
        matched_col = None
        for token in bag:
            for col_name, lut in col_lookup.items():
                if token in lut:
                    matched_ref = lut[token]
                    matched_col = col_name
                    break
            if matched_ref is not None:
                break

        if matched_ref is not None:
            mapped_models[matched_ref] = info
            mapping_hits.append((str(pkl_key), matched_ref, matched_col))

# use the mapped dict if we found anything
if mapped_models:
    route_models = mapped_models
else:
    # if we had models but no mapping, keep app usable (disable predictions later)
    if route_models:
        st.warning(
            "No mapping between PKL models and CSV routes. "
            "Showing first 100 CSV routes; Segment Times will be disabled."
        )
    route_models = {}

# -------------------- BUILD SEGMENT TIMES --------------------
def build_segment_times_from_model(sel_ref, routes_df, route_models, feature_columns):
    """
    Use the trained model for sel_ref to predict speed per segment right now.
    - Builds base + optional engineered features (heading, grids, distances).
    - Aligns to the model's expected column order; missing cols -> 0.
    """
    if sel_ref not in route_models:
        return None, "No model for this route in PKL."

    model_info = route_models[sel_ref]
    model = model_info.get("model", None)
    if model is None:
        return None, "Model object missing in PKL entry."

    row = routes_df.loc[routes_df["ref"].astype(str) == str(sel_ref)]
    if row.empty:
        return None, "Route geometry not found."
    row = row.iloc[0]

    coords = row.get("coordinates", [])
    seg_d  = row.get("segment_distance_list", [])
    if not isinstance(coords, list) or not isinstance(seg_d, list) or len(coords) < 2:
        return None, "Missing coordinates or segment distances."

    n = min(len(coords)-1, len(seg_d))

    # Build features per segment (use both start & next vertex for engineered cols)
    feats = []
    for i in range(n):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i+1][0], coords[i+1][1]
        feats.append(
            _feature_row_with_optionals(
                lat=lat1, lon=lon1, next_lat=lat2, next_lon=lon2, zones_df=zones
            )
        )

    # Figure out expected column order
    feat_cols = None
    if isinstance(feature_columns, dict):
        # try a few keys that may have been stored
        feat_cols = feature_columns.get(sel_ref) or \
                    feature_columns.get(model_info.get("features_used_key", ""), None)
    if feat_cols is None:
        feat_cols = model_info.get("features_used", None)

    dfX = pd.DataFrame(feats)

    if feat_cols is None:
        # If the PKL doesn't say, just use whatever we built (sorted for stability)
        feat_cols = list(dfX.columns)

    # IMPORTANT: align & fill missing columns (avoid KeyError)
    X = dfX.reindex(columns=feat_cols, fill_value=0)

    try:
        y_speed = pd.Series(model.predict(X)).clip(lower=1.0)
    except Exception as e:
        return None, f"Model.predict failed: {e}"

    seg_minutes = (np.array(seg_d[:n], dtype=float) / y_speed.values) * 60.0

    seg_rows = []
    for i in range(n):
        seg_rows.append({
            "ref": sel_ref,
            "segment_index": i+1,
            "start_lon": coords[i][0],  "start_lat": coords[i][1],
            "end_lon":   coords[i+1][0], "end_lat":   coords[i+1][1],
            "distance_km": float(seg_d[i]),
            "predicted_speed_kmh": float(y_speed[i]),
            "segment_travel_time_min": float(seg_minutes[i]),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df["cumulative_distance_km"] = seg_df.groupby("ref")["distance_km"].cumsum()
    seg_df["cumulative_travel_time_min"] = seg_df.groupby("ref")["segment_travel_time_min"].cumsum()
    return seg_df, None


# -------------------- PICK ROUTES TO SHOW --------------------
if routes is None or routes.empty:
    st.error("Routes CSV not found or empty.")
    st.stop()

if route_models:
    trained_refs = list(route_models.keys())[:100]
    routes_trained = routes[routes["ref"].astype(str).isin(trained_refs)].copy()
    if routes_trained.empty:
        st.warning("Models loaded but none mapped to CSV rows. Showing first 100 CSV routes; predictions disabled.")
        routes_trained = routes.head(100).copy()
        route_models = {}
else:
    routes_trained = routes.head(100).copy()

# -------------------- HEADER & PICKER --------------------
st.title("Bangkok Bus Route Explorer — Insights")

sel_ref = st.selectbox(
    "Choose a route",
    options=routes_trained["ref"].astype(str).tolist(),
    index=0,
)

route_row = routes_trained.loc[routes_trained["ref"].astype(str) == sel_ref].iloc[0]
coords_latlon = route_row["latlon"]

# -------------------- TRAFFIC NEAR ROUTE --------------------
display_df = None
if traffic is not None and coords_latlon:
    route_traffic = traffic_near_route(traffic, lat_col, lon_col, speed_col, coords_latlon, buffer_m=300)
    display_df = route_traffic if route_traffic is not None and not route_traffic.empty else traffic
else:
    display_df = traffic

# -------------------- MODEL METRICS FOR THIS ROUTE --------------------
model_info = route_models.get(sel_ref, {})
mae_val = get_metric(model_info, ["MAE","mae","mae_val","mean_absolute_error"])
r2_val  = get_metric(model_info, ["R2","r2","r2_score","r^2"])

# -------------------- TRY SEGMENT PREDICTIONS --------------------
seg_df, seg_err = (None, "Predictions disabled.") if sel_ref not in route_models else \
                  build_segment_times_from_model(sel_ref, routes_trained, route_models, feature_columns)

# -------------------- OVERVIEW CARDS --------------------
st.markdown("### Overview (filtered to selected route where possible)")
c1,c2,c3,c4,c5 = st.columns(5)
if display_df is not None and not display_df.empty and all(c in display_df.columns for c in [speed_col, lat_col, lon_col]):
    with c1:
        st.markdown(f'<div class="card"><div class="dim">Records</div><div class="big">{len(display_df):,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="dim">Avg Speed</div><div class="big">{display_df[speed_col].mean():.1f} km/h</div></div>', unsafe_allow_html=True)
    with c3:
        slow = 100*(display_df[speed_col] < 30).mean()
        st.markdown(f'<div class="card"><div class="dim">Speed &lt; 30</div><div class="big">{slow:.1f}%</div></div>', unsafe_allow_html=True)
    with c4:
        mae_txt = f"{float(mae_val):.3f}" if mae_val is not None else "—"
        st.markdown(f'<div class="card"><div class="dim">MAE</div><div class="big">{mae_txt}</div></div>', unsafe_allow_html=True)
    with c5:
        r2_txt = f"{float(r2_val):.3f}" if r2_val is not None else "—"
        st.markdown(f'<div class="card"><div class="dim">R²</div><div class="big">{r2_txt}</div></div>', unsafe_allow_html=True)
else:
    st.info("Traffic unavailable; cards are limited.")

st.markdown("---")

# # -------------------- DIAGNOSTICS --------------------
# with st.expander("Diagnostics", expanded=False):
#     st.write("PKL path:", str(ROUTE_MODELS_PKL))
#     st.write("PKL exists:", ROUTE_MODELS_PKL.exists())
#     if pkl_error:
#         st.error(f"PKL load error: {pkl_error}")
#     if routes is not None:
#         st.write("CSV rows:", len(routes))
#     st.write("Models in PKL (first 100):", "n/a" if not route_models else len(route_models))
#     if mapping_hits:
#         st.write("Sample mappings (pkl_key → csv_ref via column):")
#         df_hits = pd.DataFrame(mapping_hits[:10], columns=["pkl_key","csv_ref","matched_column"])
#         st.dataframe(df_hits, use_container_width=True, height=220)
#     else:
#         st.write("No mapping samples available (either disabled or none found).")

# -------------------- TABS --------------------
tab_map, tab_dist, tab_temporal, tab_zones, tab_segments, tab_finder = st.tabs(
    ["🗺️ Map", "📊 Speed Distribution", "⏱️ Temporal", "🚧 Zones", "🧩 Segment Times", "🚌 Route Finder"]
)

# MAP
with tab_map:
    st.subheader("Route Heatmap (route & traffic near it)")
    if display_df is None or display_df.empty or not coords_latlon:
        st.info("Need traffic + a route with coordinates.")
    else:
        fmap = folium.Map(
            location=[np.mean([p[0] for p in coords_latlon]), np.mean([p[1] for p in coords_latlon])],
            zoom_start=12, tiles="CartoDB positron"
        )
        folium.PolyLine(coords_latlon, color="#3B82F6", weight=6, opacity=0.9).add_to(fmap)

        # show stops near this route (if available)
        def stops_near_route(stops_df, coords_latlon, radius_m=300):
            if stops_df is None or stops_df.empty or not coords_latlon:
                return stops_df.iloc[0:0] if isinstance(stops_df, pd.DataFrame) else None
            smap = {c.lower(): c for c in stops_df.columns}
            latc, lonc = smap.get("lat"), smap.get("lon")
            if not latc or not lonc:
                return None

            step = max(1, len(coords_latlon)//80)
            sampled = coords_latlon[::step]
            poly_lat = np.array([p[0] for p in sampled], dtype=float)
            poly_lon = np.array([p[1] for p in sampled], dtype=float)

            pad = 0.01
            lats = [p[0] for p in coords_latlon]; lons = [p[1] for p in coords_latlon]
            box = stops_df[
                stops_df[latc].between(min(lats)-pad, max(lats)+pad) &
                stops_df[lonc].between(min(lons)-pad, max(lons)+pad)
            ].copy()
            if box.empty: return box

            def haversine_km(lat1, lon1, lat2, lon2):
                R=6371.0
                lat1=np.radians(lat1); lon1=np.radians(lon1)
                lat2=np.radians(lat2); lon2=np.radians(lon2)
                dlat=lat2-lat1; dlon=lon2-lon1
                a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                return 2*R*np.arcsin(np.sqrt(a))

            dmin = np.full(len(box), np.inf)
            blats = box[latc].to_numpy()
            blons = box[lonc].to_numpy()
            for i in range(len(sampled)):
                d = haversine_km(blats, blons, poly_lat[i], poly_lon[i])
                dmin = np.minimum(dmin, d)

            return box.loc[dmin <= (radius_m/1000.0)]

        nearby_stops = stops_near_route(stops, coords_latlon, radius_m=300) if stops is not None else None
        if nearby_stops is not None and not nearby_stops.empty:
            smap = {c.lower(): c for c in nearby_stops.columns}
            latc, lonc = smap["lat"], smap["lon"]
            namec = smap.get("refname") or smap.get("name") or smap.get("stop_name")
            for _, srow in nearby_stops.iterrows():
                folium.CircleMarker(
                    [float(srow[latc]), float(srow[lonc])],
                    radius=5, color="crimson", fill=True, fill_opacity=0.95,
                    tooltip=str(srow[namec]) if namec else None
                ).add_to(fmap)

        pts = list(zip(display_df[lat_col], display_df[lon_col], 1.0/display_df[speed_col].clip(lower=1)))
        HeatMap(pts, radius=10, blur=15, min_opacity=0.4, max_zoom=13).add_to(fmap)
        st_folium(fmap, height=560, width=None)

# SPEED DISTRIBUTION
with tab_dist:
    st.subheader("Speed Distribution")
    if display_df is None or display_df.empty or speed_col not in display_df.columns:
        st.info("No speed data to plot.")
    else:
        fig = px.histogram(display_df, x=speed_col, nbins=60, opacity=0.85)
        fig.add_vline(x=30, line_dash="dash")
        fig.add_annotation(x=30, y=0.95, yref="paper", text="30 km/h", showarrow=False)
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# TEMPORAL
with tab_temporal:
    st.subheader("Traffic by Hour & Day")
    if display_df is None or display_df.empty or "hour" not in display_df.columns:
        st.info("No temporal columns (hour/day_of_week).")
    else:
        cL, cR = st.columns(2)
        with cL:
            by_h = display_df.groupby("hour")[speed_col].agg(avg="mean", volume="size").reset_index()
            figH = go.Figure()
            figH.add_trace(go.Bar(x=by_h["hour"], y=by_h["volume"], name="Volume", opacity=0.65))
            figH.add_trace(go.Scatter(x=by_h["hour"], y=by_h["avg"], name="Avg Speed (km/h)", mode="lines+markers", yaxis="y2"))
            figH.update_layout(yaxis=dict(title="Volume"),
                               yaxis2=dict(title="Speed (km/h)", overlaying="y", side="right"),
                               xaxis=dict(dtick=2), margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figH, use_container_width=True)
        with cR:
            if "day_of_week" in display_df.columns:
                by_d = display_df.groupby("day_of_week")[speed_col].mean().reindex(range(7)).fillna(0).reset_index()
                by_d["day"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                figD = px.bar(by_d, x="day", y=speed_col, color=speed_col, color_continuous_scale="RdYlGn")
                figD.update_layout(margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(figD, use_container_width=True)

# ZONES
with tab_zones:
    st.subheader("Congestion Zones")
    needed = {"zone_id","center_lat","center_lon","avg_speed","severity","size"}
    if zones is None or zones.empty or not needed.issubset(set(zones.columns)):
        st.info("No `congestion_zones.csv` or missing columns.")
    else:
        c1, c2 = st.columns([2,3])
        with c1:
            st.dataframe(zones.sort_values("size", ascending=False)[["zone_id","severity","avg_speed","size"]].head(20),
                         use_container_width=True, height=520)
        with c2:
            pie = zones["severity"].value_counts().rename_axis("severity").reset_index(name="count")
            figZ1 = px.pie(pie, names="severity", values="count", hole=0.35)
            figZ1.update_layout(margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figZ1, use_container_width=True)
            largest = zones.nlargest(12, "size")
            figZ2 = px.bar(largest, x="zone_id", y="size", color="severity")
            figZ2.update_layout(margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figZ2, use_container_width=True)

# SEGMENT TIMES (from PKL)
with tab_segments:
    st.subheader("Segmentation & Cumulative Time (predicted from PKL)  ↻")
    if sel_ref not in route_models or seg_err or (seg_df is None):
        msg = seg_err if isinstance(seg_err, str) else "No trained model for this selected route (or prediction unavailable)."
        st.info(msg)
    else:
        total_min = seg_df["segment_travel_time_min"].sum()
        total_km  = seg_df["distance_km"].sum()
        avg_spd   = seg_df["predicted_speed_kmh"].mean()
        k1,k2,k3 = st.columns(3)
        with k1: st.markdown(f'<div class="card"><div class="dim">Total Time</div><div class="big">{total_min:.1f} min</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="card"><div class="dim">Distance</div><div class="big">{total_km:.2f} km</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="card"><div class="dim">Avg Pred Speed</div><div class="big">{avg_spd:.1f} km/h</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        keep = ["ref","segment_index","start_lat","start_lon","end_lat","end_lon",
                "distance_km","predicted_speed_kmh","segment_travel_time_min"]
        keep = [c for c in keep if c in seg_df.columns]
        st.dataframe(seg_df[keep], use_container_width=True, height=360)

        cum = seg_df[["segment_index","segment_travel_time_min"]].copy()
        cum["cumulative_min"] = cum["segment_travel_time_min"].cumsum()
        figC = px.line(cum, x="segment_index", y="cumulative_min", markers=True,
                       labels={"segment_index":"Segment #","cumulative_min":"Cumulative Time (min)"})
        figC.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(figC, use_container_width=True)

# 🚌 ROUTE FINDER TAB
with tab_finder:
    st.subheader("Find buses from Origin → Destination")
    if stops is None or stops.empty:
        st.info("No stops file found. Please ensure `cleaned_bus_stops_file.csv` exists.")
    else:
        smap = {c.lower(): c for c in stops.columns}
        s_lat = smap["lat"]; s_lon = smap["lon"]
        s_name = smap.get("refname") or smap.get("name") or smap.get("stop_name")

        c1, c2 = st.columns(2)
        with c1:
            o_name = st.selectbox("Origin stop", options=stops[s_name].astype(str).tolist(), index=0)
        with c2:
            d_name = st.selectbox("Destination stop", options=stops[s_name].astype(str).tolist(), index=1)

        o_row = stops.loc[stops[s_name] == o_name].iloc[0]
        d_row = stops.loc[stops[s_name] == d_name].iloc[0]
        o_lat, o_lon = float(o_row[s_lat]), float(o_row[s_lon])
        d_lat, d_lon = float(d_row[s_lat]), float(d_row[s_lon])

        # Fixed radius (no slider)
        radius_m = 350


        st.caption("Searching in: **trained routes (first 100, if mapped)** "
                   "or the first 100 CSV routes if no models matched.")
        search_df = routes_trained
        scope_df  = routes_trained

        results = []
        for _, r in search_df.iterrows():
            ok, oi, di, od_km, dd_km = route_covers_od(r, o_lat, o_lon, d_lat, d_lon, radius_m=radius_m)
            if not ok:
                continue
            ref = str(r["ref"])
            name = r.get("name", "")
            est_min = estimate_time_between_indices(ref, oi, di, scope_df, route_models, feature_columns)
            trained = ref in route_models
            results.append({
                "ref": ref,
                "name": name,
                "trained_model": int(trained),
                "nearest_vertex_origin": oi,
                "nearest_vertex_dest": di,
                "dist_to_origin_m": round(od_km*1000.0, 1),
                "dist_to_dest_m": round(dd_km*1000.0, 1),
                "est_time_min_between": (round(est_min, 2) if est_min is not None else "—")
            })

        if not results:
            st.info("No bus routes found that pass near both stops with the given radius (and correct direction). Try increasing the radius.")
        else:
            res_df = pd.DataFrame(results)
            res_df["__sort_time__"] = pd.to_numeric(res_df["est_time_min_between"], errors="coerce").fillna(np.inf)
            res_df = res_df.sort_values(by=["trained_model", "__sort_time__"], ascending=[False, True]) \
                           .drop(columns="__sort_time__")

            st.markdown("**Matching bus routes:**")
            st.dataframe(res_df, use_container_width=True, height=360)

            fmap = folium.Map(location=[(o_lat+d_lat)/2, (o_lon+d_lon)/2], zoom_start=12, tiles="CartoDB positron")
            folium.Marker([o_lat, o_lon], tooltip=f"Origin: {o_name}", icon=folium.Icon(color="green")).add_to(fmap)
            folium.Marker([d_lat, d_lon], tooltip=f"Destination: {d_name}", icon=folium.Icon(color="red")).add_to(fmap)

            legend_items = []
            for _, row in res_df.head(8).iterrows():
                ref = row["ref"]
                rr = search_df.loc[search_df["ref"].astype(str) == ref].iloc[0]
                coords_latlon = rr.get("latlon", [])
                if not coords_latlon:
                    continue
                col = color_for_ref(ref)
                folium.PolyLine(
                    coords_latlon,
                    weight=5,
                    opacity=0.85,
                    color=col,
                    tooltip=f"Route {ref} — {rr.get('name','')}",
                ).add_to(fmap)
                legend_items.append((ref, rr.get("name",""), col))

            if legend_items:
                legend_html = '''
                <div style="
                    position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                    background: #ffffff; padding: 10px 12px; border: 1px solid #bbb; border-radius: 8px;
                    font-size: 13px; box-shadow: 0 2px 10px rgba(0,0,0,.15); max-width: 280px;
                    color:#111; opacity:1; ">
                    <b style="color:#111;">Routes shown</b><br/>
                '''
                for ref, name, col in legend_items:
                    label = (name if name else ref)
                    legend_html += (
                        '<div style="display:flex;align-items:center;margin-top:6px;color:#111;">'
                        f'<span style="display:inline-block;width:14px;height:14px;background:{col};'
                        'border:1px solid #666;margin-right:8px;border-radius:3px;"></span>'
                        f'<span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#111;">'
                        f'{ref} — {label}</span></div>'
                    )
                legend_html += '</div>'
                fmap.get_root().html.add_child(folium.Element(legend_html))

            st_folium(fmap, height=560, width=None)
