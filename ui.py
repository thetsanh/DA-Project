# ui.py — PKL-powered, tabbed dashboard (100 trained routes, no raw-route tab)

import os, ast, pickle
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
DATA_DIR = Path("data")
CLEAN_DIR = Path("cleaned")

TRAFFIC_CSV = DATA_DIR / "traffic.csv"
ROUTES_CSV  = CLEAN_DIR / "cleaned_bus_routes_file.csv"
STOPS_CSV   = CLEAN_DIR / "cleaned_bus_stops_file.csv"   # not used now, kept for future
ZONES_CSV   = CLEAN_DIR / "congestion_zones.csv"

ROUTE_MODELS_PKL    = DATA_DIR / "route_models.pkl"
FEATURE_COLUMNS_PKL = DATA_DIR / "feature_columns.pkl"

# -------------------- page look --------------------
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

# -------------------- helpers --------------------
def load_csv(path, **kwargs):
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

    # compute min distance to any sampled point (loop over sampled but keep vector math per step)
    dmin = np.full(len(pre), np.inf)
    for i in range(len(sampled)):
        d = haversine_km(pre[lat_col].to_numpy(), pre[lon_col].to_numpy(), poly_lat[i], poly_lon[i])
        dmin = np.minimum(dmin, d)
    mask = dmin <= (buffer_m/1000.0)
    return pre.loc[mask].copy()

def build_segment_times_from_model(sel_ref, routes_df, route_models, feature_columns):
    """Use the trained model for sel_ref to predict speed per segment right now."""
    if sel_ref not in route_models:
        return None, "No model for this route in PKL."

    model_info = route_models[sel_ref]
    model = model_info["model"]

    row = routes_df.loc[routes_df["ref"] == sel_ref]
    if row.empty:
        return None, "Route geometry not found."
    row = row.iloc[0]

    coords = row["coordinates"] if isinstance(row["coordinates"], list) else []
    seg_d  = row["segment_distance_list"] if isinstance(row["segment_distance_list"], list) else []
    if not coords or not seg_d:
        return None, "Missing coordinates or segment distances."

    n = min(len(coords)-1, len(seg_d))
    feats = [make_feature_row(coords[i][1], coords[i][0]) for i in range(n)]  # model was trained on lon,lat order in X?
    # get the right column order
    feat_cols = None
    if isinstance(feature_columns, dict):
        feat_cols = feature_columns.get(sel_ref) or feature_columns.get(model_info.get("features_used_key", ""), None)
    if feat_cols is None:
        feat_cols = model_info.get("features_used", list(feats[0].keys()))
    X = pd.DataFrame(feats)[feat_cols]

    y_speed = pd.Series(model.predict(X)).clip(lower=1.0)
    seg_minutes = (np.array(seg_d[:n]) / y_speed.values) * 60.0

    seg_rows = []
    for i in range(n):
        seg_rows.append({
            "ref": sel_ref,
            "segment_index": i+1,
            "start_lon": coords[i][0],  "start_lat": coords[i][1],
            "end_lon":   coords[i+1][0], "end_lat":   coords[i+1][1],
            "distance_km": seg_d[i],
            "predicted_speed_kmh": float(y_speed[i]),
            "segment_travel_time_min": float(seg_minutes[i]),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df["cumulative_distance_km"] = seg_df.groupby("ref")["distance_km"].cumsum()
    seg_df["cumulative_travel_time_min"] = seg_df.groupby("ref")["segment_travel_time_min"].cumsum()
    return seg_df, None

def _coerce_num(x):
    try:
        return float(x)
    except Exception:
        return None

def get_metric(model_info: dict, key_variants):
    """Search MAE/R2 in several common places/names; return float or None."""
    if not isinstance(model_info, dict):
        return None

    # 1) direct keys on the root (case-insensitive)
    lower_map = {k.lower(): v for k, v in model_info.items()}
    for k in key_variants:
        if k.lower() in lower_map:
            val = _coerce_num(lower_map[k.lower()])
            if val is not None:
                return val

    # 2) nested under "metrics" or "perf"
    for container in ("metrics", "metric", "perf", "performance"):
        sub = model_info.get(container)
        if isinstance(sub, dict):
            sub_lower = {k.lower(): v for k, v in sub.items()}
            for k in key_variants:
                if k.lower() in sub_lower:
                    val = _coerce_num(sub_lower[k.lower()])
                    if val is not None:
                        return val
    return None


# -------------------- load base data --------------------
traffic = load_csv(TRAFFIC_CSV)
routes  = load_csv(ROUTES_CSV)
zones   = load_csv(ZONES_CSV)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# STOPS: load & sanitize  (ADDED SECTION)
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
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
    # keep lat/lon list for folium heatmap
    routes["latlon"] = routes["coordinates"].apply(lambda pts: [[p[1], p[0]] for p in pts] if isinstance(pts, list) else [])

# traffic basics
lat_col, lon_col, speed_col = "lat", "lon", "speed"
if traffic is not None:
    cmap = {c.lower(): c for c in traffic.columns}
    lat_col   = cmap.get("lat", lat_col)
    lon_col   = cmap.get("lon", lon_col)
    speed_col = cmap.get("speed", speed_col)
    # time columns if present
    tcol = next((c for c in ["timestamp","time","datetime"] if c in traffic.columns), None)
    if tcol:
        traffic["_ts"] = pd.to_datetime(traffic[tcol], errors="coerce")
        traffic["hour"] = traffic["_ts"].dt.hour
        traffic["day_of_week"] = traffic["_ts"].dt.dayofweek
    else:
        traffic["hour"] = 0
        traffic["day_of_week"] = 0

# -------------------- load models --------------------
route_models = {}
feature_columns = {}
if ROUTE_MODELS_PKL.exists():
    try:
        with open(ROUTE_MODELS_PKL, "rb") as f:
            raw = pickle.load(f)
        # normalize keys as strings
        route_models = {str(k).strip(): v for k, v in raw.items()}
    except Exception as e:
        st.error(f"Could not load pickle {ROUTE_MODELS_PKL}: {e}")

if FEATURE_COLUMNS_PKL.exists():
    try:
        with open(FEATURE_COLUMNS_PKL, "rb") as f:
            feature_columns = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load feature columns {FEATURE_COLUMNS_PKL}: {e}")

# figure out trained routes (first 100)
trained_refs = list(route_models.keys())[:100]
routes_trained = routes[routes["ref"].isin(trained_refs)].copy() if routes is not None else None
if routes_trained is None or routes_trained.empty:
    st.error("No routes found that match the first 100 trained models in the PKL.")
    st.stop()

# -------------------- UI: header & picker --------------------
st.title("Bangkok Bus Route Explorer — Insights")

sel_ref = st.selectbox(
    "Choose a trained route",
    options=routes_trained["ref"].tolist(),
    index=0,
)

# route geometry for the selection
route_row = routes_trained.loc[routes_trained["ref"] == sel_ref].iloc[0]
coords_latlon = route_row["latlon"]

# traffic near route (fixed 300m buffer)
route_traffic = traffic_near_route(traffic, lat_col, lon_col, speed_col, coords_latlon, buffer_m=300) \
                if traffic is not None else None

display_df = route_traffic if route_traffic is not None and not route_traffic.empty else traffic

# try to compute PKL predictions for segment table & derive metrics
seg_df, seg_err = build_segment_times_from_model(sel_ref, routes_trained, route_models, feature_columns)

# optional model metrics (if present inside model_info)

model_info = route_models.get(sel_ref, {})
mae_val = get_metric(model_info, ["MAE", "mae", "mae_val"])
r2_val  = get_metric(model_info, ["R2", "r2", "r2_score", "r^2"])


# -------------------- Overview cards --------------------
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

# -------------------- Tabs (raw-route tab REMOVED) --------------------
tab_map, tab_dist, tab_temporal, tab_zones, tab_segments = st.tabs(
    ["🗺️ Map", "📊 Speed Distribution", "⏱️ Temporal", "🚧 Zones", "🧩 Segment Times"]
)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# STOPS helper: pick stops near route polyline  (ADDED FUNCTION)
def stops_near_route(stops_df, coords_latlon, radius_m=300):
    """Return only the stops within ~radius_m of the route polyline."""
    if stops_df is None or stops_df.empty or not coords_latlon:
        return stops_df.iloc[0:0] if isinstance(stops_df, pd.DataFrame) else None

    smap = {c.lower(): c for c in stops_df.columns}
    latc, lonc = smap["lat"], smap["lon"]

    # sample the polyline
    step = max(1, len(coords_latlon)//80)
    sampled = coords_latlon[::step]
    poly_lat = np.array([p[0] for p in sampled], dtype=float)
    poly_lon = np.array([p[1] for p in sampled], dtype=float)

    # coarse bbox prefilter
    pad = 0.01
    lats = [p[0] for p in coords_latlon]; lons = [p[1] for p in coords_latlon]
    box = stops_df[
        stops_df[latc].between(min(lats)-pad, max(lats)+pad) &
        stops_df[lonc].between(min(lons)-pad, max(lons)+pad)
    ].copy()
    if box.empty:
        return box

    # vectorized distance to sampled points
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
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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

        # >>> ADD STOPS ON MAP <<<
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
    if zones is None or zones.empty or not {"zone_id","center_lat","center_lon","avg_speed","severity","size"}.issubset(zones.columns):
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
    if seg_err:
        st.info(seg_err)
    else:
        # KPI row
        total_min = seg_df["segment_travel_time_min"].sum()
        total_km  = seg_df["distance_km"].sum()
        avg_spd   = seg_df["predicted_speed_kmh"].mean()
        k1,k2,k3 = st.columns(3)
        with k1: st.markdown(f'<div class="card"><div class="dim">Total Time</div><div class="big">{total_min:.1f} min</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="card"><div class="dim">Distance</div><div class="big">{total_km:.2f} km</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="card"><div class="dim">Avg Pred Speed</div><div class="big">{avg_spd:.1f} km/h</div></div>', unsafe_allow_html=True)

        # little gap before the table
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # table
        keep = ["ref","segment_index","start_lat","start_lon","end_lat","end_lon",
                "distance_km","predicted_speed_kmh","segment_travel_time_min"]
        keep = [c for c in keep if c in seg_df.columns]
        st.dataframe(seg_df[keep], use_container_width=True, height=360)

        # cumulative plot
        cum = seg_df[["segment_index","segment_travel_time_min"]].copy()
        cum["cumulative_min"] = cum["segment_travel_time_min"].cumsum()
        figC = px.line(cum, x="segment_index", y="cumulative_min", markers=True,
                       labels={"segment_index":"Segment #","cumulative_min":"Cumulative Time (min)"})
        figC.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(figC, use_container_width=True)
