# dashboard_tabs.py — Sleek, tabbed, route-aware dashboard
# All sections update to the selected route. If a file is missing, that tab shows a helpful message.

import os, ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ------------------------- PATHS (adjust if needed) -------------------------
TRAFFIC_CSV          = "data/traffic.csv"                   # lat, lon, speed, (optional) timestamp, VehicleID
ROUTES_CSV           = "cleaned/cleaned_bus_routes_file.csv"
STOPS_CSV            = "cleaned/cleaned_bus_stops_file.csv"
ZONES_CSV            = "cleaned/congestion_zones.csv"
SEGMENT_TIMES_CSV    = "outputs/segment_times_latest.csv"   # ref, segment_index, distance_km, predicted_speed_kmh, segment_travel_time_min
ROUTE_METRICS_CSV    = "outputs/route_metrics_latest.csv"   # per-ref MAE, RMSE, MAPE, R2 (optional)

BKK_LAT, BKK_LON = 13.7563, 100.5018

# ------------------------- PAGE STYLE -------------------------
st.set_page_config(page_title="Bangkok Bus Insights", layout="wide")
st.markdown("""
<style>
h1,h2,h3 {font-weight:800; letter-spacing:.2px}
.card{background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
      border-radius:16px; padding:16px 18px; box-shadow:0 8px 24px rgba(0,0,0,.18)}
.big{font-size:40px; font-weight:800}
.dim{color:#9aa4b2}
hr{border:none; height:1px; background:rgba(255,255,255,.1); margin:12px 0}
</style>
""", unsafe_allow_html=True)

# ------------------------- HELPERS -------------------------
def load_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs) if os.path.exists(path) else None
    except Exception as e:
        st.warning(f"Could not read `{path}`: {e}")
        return None

def parse_coords_str(x):
    """'[[lon,lat], ...]' -> [[lat,lon], ...] for Folium"""
    try:
        pts = ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return []
    out=[]
    if isinstance(pts,(list,tuple)):
        for p in pts:
            if isinstance(p,(list,tuple)) and len(p)>=2 and pd.notna(p[0]) and pd.notna(p[1]):
                out.append([float(p[1]), float(p[0])])  # [lat,lon]
    return out

def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    lat1=np.radians(lat1); lon1=np.radians(lon1)
    lat2=np.radians(lat2); lon2=np.radians(lon2)
    dlat=lat2-lat1; dlon=lon2-lon1
    a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def distance_point_to_polyline_km(lat, lon, poly_lat, poly_lon):
    """Crude nearest distance: min distance to sampled polyline points."""
    # vectorized to points
    return np.min(haversine_km(lat, lon, poly_lat, poly_lon))

def traffic_near_route(traffic, lat_col, lon_col, speed_col, coords_latlon, buffer_m=300):
    """Return traffic points within ~buffer_m of route polyline."""
    if traffic is None or traffic.empty or not coords_latlon:
        return pd.DataFrame(columns=traffic.columns if traffic is not None else [])
    # sample polyline for speed
    step = max(1, len(coords_latlon)//80)
    sampled = coords_latlon[::step]
    poly_lat = np.array([p[0] for p in sampled], dtype=float)
    poly_lon = np.array([p[1] for p in sampled], dtype=float)

    # quick coarse filter by bbox padding to reduce work
    pad = 0.01  # ~1.1km in lat, acceptable coarse filter
    lats = [p[0] for p in coords_latlon]; lons = [p[1] for p in coords_latlon]
    box = traffic[(traffic[lat_col].between(min(lats)-pad, max(lats)+pad)) &
                  (traffic[lon_col].between(min(lons)-pad, max(lons)+pad))].copy()
    if box.empty: 
        return box

    # exact-ish nearest to sampled points
    thresh_km = buffer_m/1000.0
    keep = []
    for i, r in box.iterrows():
        d = distance_point_to_polyline_km(r[lat_col], r[lon_col], poly_lat, poly_lon)
        if d <= thresh_km:
            keep.append(i)
    return box.loc[keep]

# ------------------------- LOAD DATA -------------------------
traffic = load_csv(TRAFFIC_CSV)
routes  = load_csv(ROUTES_CSV)
stops   = load_csv(STOPS_CSV)
zones   = load_csv(ZONES_CSV)
segs    = load_csv(SEGMENT_TIMES_CSV)
metrics = load_csv(ROUTE_METRICS_CSV)

# normalize traffic basic cols (case-insensitive)
lat = "lat"; lon = "lon"; speed = "speed"
if traffic is not None:
    cmap = {c.lower(): c for c in traffic.columns}
    lat   = cmap.get("lat", lat)
    lon   = cmap.get("lon", lon)
    speed = cmap.get("speed", speed)
    # timestamp -> hour/day_of_week if present
    tcol = next((c for c in ["timestamp","time","datetime"] if c in traffic.columns), None)
    if tcol:
        traffic["_ts"] = pd.to_datetime(traffic[tcol], errors="coerce")
        traffic["hour"] = traffic["_ts"].dt.hour
        traffic["day_of_week"] = traffic["_ts"].dt.dayofweek
    traffic["hour"] = traffic.get("hour", 0)
    traffic["day_of_week"] = traffic.get("day_of_week", 0)
    # vehicle id normalize (optional)
    if "VehicleID" not in traffic.columns:
        for c in ["vehicle_id","vehicleid","vid","id"]:
            if c in traffic.columns:
                traffic = traffic.rename(columns={c:"VehicleID"})
                break

# normalize routes
if routes is not None:
    routes = routes.copy()
    if "ref" not in routes.columns:
        routes["ref"] = routes["route_id"].astype(str)
    routes["ref"] = routes["ref"].astype(str)
    routes["latlon"] = routes["coordinates"].apply(parse_coords_str)

# normalize segments & metrics
if segs is not None and "ref" in segs.columns:
    segs["ref"] = segs["ref"].astype(str)
if metrics is not None and "ref" in metrics.columns:
    metrics["ref"] = metrics["ref"].astype(str)

# ========================= ROUTE PICKER (GLOBAL) =========================
st.title("Bangkok Bus Route Explorer — Insights")

route_options = []
if routes is not None and not routes.empty:
    def label(r):
        nm = r.get("name","") if isinstance(r, pd.Series) else ""
        base = str(r["ref"]) if isinstance(r, pd.Series) else str(r)
        return f"{base} — {nm}" if nm else base
    route_options = routes["ref"].unique().tolist()

col_pick1, col_pick2 = st.columns([2,1])
with col_pick1:
    sel_ref = st.selectbox("Choose a route", route_options, format_func=lambda x: x, index=0 if route_options else None)
with col_pick2:
    buffer_m = st.slider("Map buffer (m)", 150, 600, 300, 50)

# compute route-specific data
route_row = routes[routes["ref"]==sel_ref].iloc[0] if routes is not None and sel_ref in routes["ref"].values else None
coords_latlon = route_row["latlon"] if route_row is not None else []
route_traffic = traffic_near_route(traffic, lat, lon, speed, coords_latlon, buffer_m=buffer_m) if traffic is not None else None
route_segs     = segs[segs["ref"]==sel_ref].copy() if segs is not None and sel_ref in segs["ref"].values else None
route_metrics  = metrics[metrics["ref"]==sel_ref].iloc[0].to_dict() if metrics is not None and sel_ref in metrics["ref"].values else {}

# fallback if nothing found
display_df = route_traffic if (route_traffic is not None and not route_traffic.empty) else traffic

# =============================== OVERVIEW (always first) ===============================
st.markdown("### Overview (filtered to selected route when possible)")

colA, colB, colC, colD, colE = st.columns(5)
if display_df is not None and not display_df.empty and all(c in display_df.columns for c in [speed, lat, lon]):
    with colA:
        st.markdown(f'<div class="card"><div class="dim">Records</div><div class="big">{len(display_df):,}</div></div>', unsafe_allow_html=True)
    with colB:
        st.markdown(f'<div class="card"><div class="dim">Avg Speed</div><div class="big">{display_df[speed].mean():.1f} km/h</div></div>', unsafe_allow_html=True)
    with colC:
        pct_slow = 100*(display_df[speed] < 30).mean()
        st.markdown(f'<div class="card"><div class="dim">Speed &lt; 30</div><div class="big">{pct_slow:.1f}%</div></div>', unsafe_allow_html=True)
    with colD:   
        uniq = display_df["VehicleID"].nunique() if "VehicleID" in display_df.columns else 0
        st.markdown(f'<div class="card"><div class="dim">Unique Vehicles</div><div class="big">{uniq if uniq else "—"}</div></div>', unsafe_allow_html=True)
    with colE:
        seg_total = route_segs["segment_travel_time_min"].sum() if route_segs is not None and not route_segs.empty else None
        st.markdown(f'<div class="card"><div class="dim">Total Seg Time</div><div class="big">{f"{seg_total:.1f} min" if seg_total else "—"}</div></div>', unsafe_allow_html=True)
else:
    st.info("No traffic available for this route; showing cards would be empty.")

st.markdown("---")

# =============================== TABS ===============================
tab_map, tab_dist, tab_temporal, tab_zones, tab_segments, tab_raw = st.tabs(
    ["🗺️ Map", "📊 Speed Distribution", "⏱️ Temporal", "🚧 Zones", "🧩 Segment Times", "🧭 Raw Route + Stops"]
)

# ---------------- MAP TAB ----------------
with tab_map:
    st.subheader("Route Heatmap (route & traffic near it)")
    if display_df is None or display_df.empty or not coords_latlon:
        st.info("Need traffic + a route with coordinates.")
    else:
        fmap = folium.Map(location=[np.mean([p[0] for p in coords_latlon]), np.mean([p[1] for p in coords_latlon])],
                          zoom_start=12, tiles="CartoDB positron")
        # draw route
        folium.PolyLine(coords_latlon, color="#3B82F6", weight=6, opacity=0.9).add_to(fmap)
        # stops within buffer
        if stops is not None and {"lat","lon"}.issubset(stops.columns):
            # quick check against sampled points
            step = max(1, len(coords_latlon)//80)
            sampled = coords_latlon[::step]
            poly_lat = np.array([p[0] for p in sampled]); poly_lon = np.array([p[1] for p in sampled])
            for _, srow in stops.dropna(subset=["lat","lon"]).iterrows():
                dmin = distance_point_to_polyline_km(srow["lat"], srow["lon"], poly_lat, poly_lon)
                if dmin <= buffer_m/1000.0:
                    folium.CircleMarker([srow["lat"], srow["lon"]], radius=5, color="red", fill=True,
                                        fill_opacity=0.95, tooltip=str(srow.get("refname",""))).add_to(fmap)
        # heatmap (slower = hotter). pick route_traffic if it exists
        plot_df = route_traffic if route_traffic is not None and not route_traffic.empty else display_df
        pts = list(zip(plot_df[lat], plot_df[lon], 1.0/plot_df[speed].clip(lower=1)))
        HeatMap(pts, radius=10, blur=15, min_opacity=0.4, max_zoom=13).add_to(fmap)
        st_folium(fmap, height=560, width=None)

# ---------------- DISTRIBUTION TAB ----------------
with tab_dist:
    st.subheader("Speed Distribution")
    if display_df is None or display_df.empty or speed not in display_df.columns:
        st.info("No speed data to plot.")
    else:
        fig = px.histogram(display_df, x=speed, nbins=60, opacity=0.85)
        fig.add_vline(x=30, line_dash="dash")
        fig.add_annotation(x=30, y=0.95, yref="paper", text="30 km/h", showarrow=False)
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------- TEMPORAL TAB ----------------
with tab_temporal:
    st.subheader("Traffic by Hour & Day (for selected route when available)")
    if display_df is None or display_df.empty or "hour" not in display_df.columns:
        st.info("No temporal columns (hour/day_of_week).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            by_h = display_df.groupby("hour")[speed].agg(avg="mean", volume="size").reset_index()
            figH = go.Figure()
            figH.add_trace(go.Bar(x=by_h["hour"], y=by_h["volume"], name="Volume", opacity=0.65))
            figH.add_trace(go.Scatter(x=by_h["hour"], y=by_h["avg"], name="Avg Speed (km/h)", mode="lines+markers", yaxis="y2"))
            figH.update_layout(yaxis=dict(title="Volume"),
                               yaxis2=dict(title="Speed (km/h)", overlaying="y", side="right"),
                               xaxis=dict(dtick=2), margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figH, use_container_width=True)
        with c2:
            if "day_of_week" in display_df.columns:
                by_d = display_df.groupby("day_of_week")[speed].mean().reindex(range(7)).fillna(0).reset_index()
                by_d["day"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                figD = px.bar(by_d, x="day", y=speed, color=speed, color_continuous_scale="RdYlGn")
                figD.update_layout(margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(figD, use_container_width=True)

# ---------------- ZONES TAB ----------------
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

# ---------------- SEGMENTS TAB ----------------
with tab_segments:
    st.subheader("Segmentation & Cumulative Time")
    if route_segs is None or route_segs.empty:
        st.info("No segment file for this route yet. Export it to `outputs/segment_times_latest.csv`.")
    else:
        # top metrics
        t_total = route_segs["segment_travel_time_min"].sum()
        d_total = route_segs["distance_km"].sum() if "distance_km" in route_segs.columns else np.nan
        v_avg   = route_segs["predicted_speed_kmh"].mean() if "predicted_speed_kmh" in route_segs.columns else np.nan
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(f'<div class="card"><div class="dim">Route</div><div class="big">{sel_ref}</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="card"><div class="dim">Total Time</div><div class="big">{t_total:.1f} min</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="card"><div class="dim">Distance</div><div class="big">{d_total:.2f} km</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="card"><div class="dim">Avg Pred Speed</div><div class="big">{v_avg:.1f} km/h</div></div>', unsafe_allow_html=True)

        # quality metrics (optional)
        if route_metrics:
            k1,k2,k3,k4 = st.columns(4)
            def fmt(x): 
                try: return f"{float(x):.3f}"
                except: return "—"
            with k1: st.markdown(f'<div class="card"><div class="dim">MAE</div><div class="big">{fmt(route_metrics.get("MAE"))}</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="card"><div class="dim">RMSE</div><div class="big">{fmt(route_metrics.get("RMSE"))}</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="card"><div class="dim">MAPE</div><div class="big">{fmt(route_metrics.get("MAPE"))}</div></div>', unsafe_allow_html=True)
            with k4: st.markdown(f'<div class="card"><div class="dim">R²</div><div class="big">{fmt(route_metrics.get("R2"))}</div></div>', unsafe_allow_html=True)

        # table
        show_cols = ["segment_index","start_stop","end_stop","distance_km","predicted_speed_kmh","segment_travel_time_min"]
        show_cols = [c for c in show_cols if c in route_segs.columns]
        st.dataframe(route_segs[["ref"]+show_cols], use_container_width=True, height=360)

        # cumulative chart
        cum = route_segs[["segment_index","segment_travel_time_min"]].copy()
        cum["cumulative_min"] = cum["segment_travel_time_min"].cumsum()
        figC = px.line(cum, x="segment_index", y="cumulative_min", markers=True,
                       labels={"segment_index":"Segment #","cumulative_min":"Cumulative Time (min)"})
        figC.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(figC, use_container_width=True)

# ---------------- RAW ROUTE TAB ----------------
with tab_raw:
    st.subheader("Raw Route Polyline + Nearby Stops")
    if route_row is None or not coords_latlon:
        st.info("No route coordinates to draw.")
    else:
        fmap = folium.Map(location=[np.mean([p[0] for p in coords_latlon]), np.mean([p[1] for p in coords_latlon])],
                          zoom_start=13, tiles="CartoDB positron", control_scale=True)
        folium.PolyLine(coords_latlon, weight=5, opacity=0.9, color="#2563EB").add_to(fmap)
        # add stops near line
        if stops is not None and {"lat","lon"}.issubset(stops.columns):
            step = max(1, len(coords_latlon)//80)
            sampled = coords_latlon[::step]
            poly_lat = np.array([p[0] for p in sampled]); poly_lon = np.array([p[1] for p in sampled])
            for _, srow in stops.dropna(subset=["lat","lon"]).iterrows():
                dmin = distance_point_to_polyline_km(srow["lat"], srow["lon"], poly_lat, poly_lon)
                if dmin <= buffer_m/1000.0:
                    folium.CircleMarker([srow["lat"], srow["lon"]], radius=5, color="crimson", fill=True,
                                        fill_opacity=0.95, tooltip=str(srow.get("refname",""))).add_to(fmap)
        st_folium(fmap, height=560, width=None)

# ---------------- FOOTER ----------------