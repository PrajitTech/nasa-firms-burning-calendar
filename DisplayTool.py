import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="NASA FIRMS Unified Burning Activity Calendar", layout="wide"
)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }
    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.1rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
    }
    .js-plotly-plot .plotly .nsewdrag {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


import json
import os
import requests


@st.cache_data
def load_us_states_geojson() -> dict:
  if os.path.exists("us-states.json"):
    with open("us-states.json", "r", encoding="utf-8") as f:
      return json.load(f)
  local_path = os.path.join("firms_raw_data", "us-states.json")
  if os.path.exists(local_path):
    with open(local_path, "r", encoding="utf-8") as f:
      return json.load(f)
  return {}


@st.cache_data
def load_calendar_data(path: str) -> pd.DataFrame:
  df = pd.read_csv(path, parse_dates=["acq_date"])
  df["year"] = df["acq_date"].dt.year
  df["day_of_week"] = df["acq_date"].dt.dayofweek
  df["week_of_year"] = df["acq_date"].dt.isocalendar().week.astype(int)
  df.loc[
      (df["acq_date"].dt.month == 1) & (df["week_of_year"] > 50), "week_of_year"
  ] = 0
  df.loc[
      (df["acq_date"].dt.month == 12) & (df["week_of_year"] == 1),
      "week_of_year",
  ] = 53
  return df


@st.cache_data
def load_csv_optional(path: str, parse_dates=None) -> pd.DataFrame:
  try:
    return pd.read_csv(path, parse_dates=parse_dates)
  except FileNotFoundError:
    return pd.DataFrame()

CODE_TO_STATE_NAME = {
    "WA": "Washington",
    "OR": "Oregon",
    "CA": "California",
    "NV": "Nevada",
    "ID": "Idaho",
    "MT": "Montana",
    "WY": "Wyoming",
    "UT": "Utah",
    "AZ": "Arizona",
    "CO": "Colorado",
    "NM": "New Mexico",
}

REGION_VIEWPORTS = {
    "Western_US": {"lat": 39.8, "lon": -114.0, "zoom": 3.1},
    "WA": {"lat": 47.4, "lon": -120.5, "zoom": 5.2},
    "OR": {"lat": 44.0, "lon": -120.5, "zoom": 5.1},
    "CA": {"lat": 37.2, "lon": -119.5, "zoom": 4.5},
    "NV": {"lat": 39.3, "lon": -116.8, "zoom": 4.8},
    "ID": {"lat": 44.5, "lon": -114.5, "zoom": 4.6},
    "MT": {"lat": 47.0, "lon": -109.6, "zoom": 4.7},
    "WY": {"lat": 43.0, "lon": -107.5, "zoom": 5.0},
    "UT": {"lat": 39.3, "lon": -111.7, "zoom": 5.0},
    "AZ": {"lat": 34.3, "lon": -111.7, "zoom": 5.0},
    "CO": {"lat": 39.0, "lon": -105.5, "zoom": 5.1},
    "NM": {"lat": 34.4, "lon": -106.0, "zoom": 5.0},
}

def filter_selected_state_geojson(geojson_data: dict, region_code: str) -> dict:
  """Return a GeoJSON FeatureCollection for the active state or all 11 Western US states."""
  if not geojson_data:
    return {}

  target_names = (
      set(CODE_TO_STATE_NAME.values())
      if region_code == "Western_US"
      else {CODE_TO_STATE_NAME.get(region_code, "")}
  )

  matched_features = tuple(
      feat
      for feat in geojson_data.get("features", ())
      if feat.get("properties", {}).get("name") in target_names
  )
  if not matched_features:
    return {}

  return {"type": "FeatureCollection", "features": list(matched_features)}


st.title("🔥 Western US Harmonized Burning Activity Calendar")
st.markdown(
    '<p class="subtitle-text">Unified <b>MODIS (2000–2011)</b> &amp; '
    '<b>VIIRS (2012–Present)</b> Wildfire Explorer</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }
    .main-title {
        padding-left: 1.2rem;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .subtitle-text {
        text-align: left;
        font-size: 1.05rem;
        opacity: 0.82;
        margin-top: -0.15rem;
        margin-bottom: 0.45rem;
    }
    hr {
        margin-top: 0.35rem !important;
        margin-bottom: 0.45rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }
    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.1rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.15rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        margin-top: 0.15rem !important;
        margin-bottom: 0.15rem !important;
    }
    .js-plotly-plot .plotly .nsewdrag {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


df = load_calendar_data("harmonized_western_us_production.csv")
map_df = load_csv_optional("harmonized_h3_map.csv")
val_df = load_csv_optional("calibration_validation.csv", parse_dates=["acq_date"])

# --- Sidebar Controls ---
st.sidebar.header("Filter Controls")
regions = sorted(df["region_id"].unique())
default_idx = regions.index("Western_US") if "Western_US" in regions else 0
selected_region = st.sidebar.selectbox(
    "Select Region", regions, index=default_idx
)

years = sorted(df["year"].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Select Calendar Year", years, index=0)

metric_options = {
    "Unified Active Fire Cells": "harmonized_active_cells",
    "Unified Fire Radiative Power (MW)": "harmonized_frp_mw",
    "Anomaly Ratio (vs. Historical Median)": "anomaly_ratio",
}

selected_metric_label = st.sidebar.radio(
    "Calendar Metric", list(metric_options.keys())
)
metric_col = metric_options[selected_metric_label]
# Track sidebar selections to detect changes
current_filters = (selected_region, selected_year, metric_col)

if "last_filters" not in st.session_state:
  st.session_state.last_filters = current_filters
elif st.session_state.last_filters != current_filters:
  st.session_state.last_filters = current_filters
  # Force parent Streamlit viewport to scroll to the very top
  components.html(
      """
      <script>
        const mainSection = window.parent.document.querySelector('section.main');
        if (mainSection) {
          mainSection.scrollTo({ top: 0, behavior: 'instant' });
        }
        window.parent.scrollTo({ top: 0, behavior: 'instant' });
      </script>
      """,
      height=0,
  )


reg_df = df[df["region_id"] == selected_region]
yr_df = reg_df[reg_df["year"] == selected_year].sort_values("acq_date")

# ==============================================================================
# TAB 1: BURNING ACTIVITY CALENDAR + INTERACTIVE H3 HOTSPOT MAP
# ==============================================================================
# Format a clean display name (e.g., "California (CA)" or "Western US")
# 1. Dynamic Header FIRST
display_region = (
    "Western US"
    if selected_region == "Western_US"
    else f"{CODE_TO_STATE_NAME.get(selected_region, selected_region)} ({selected_region})"
)

st.header(f"📍 {display_region} — {selected_year} Fire Season")

# 2. Single Navigation Tabs Call SECOND
tab1, tab2 = st.tabs(
    list(("📅 Burning Activity Calendar & Map", "🔬 Sensor Calibration Validation"))
)

with tab1:
  col1, col2, col3, col4 = st.columns(4)
  peak_row = (
      yr_df.loc[yr_df["harmonized_active_cells"].idxmax()]
      if not yr_df.empty
      else None
  )

  col1.metric(
      "Total Active Fire Cell-Days",
      f"{yr_df['harmonized_active_cells'].sum():,.0f}",
  )
  col2.metric(
      "Cumulative Radiant Power", f"{yr_df['harmonized_frp_mw'].sum():,.0f} MW"
  )
  col3.metric(
      "Peak Fire Date",
      peak_row["acq_date"].strftime("%Y-%m-%d")
      if peak_row is not None
      else "N/A",
  )
  col4.metric(
      "Data Provenance",
      ", ".join(yr_df["data_provenance"].unique())
      if not yr_df.empty
      else "N/A",
  )

  st.divider()

  st.subheader(f"Daily Burning Activity Calendar — {display_region} ({selected_year})")
  st.caption("👆 Click any day tile to update the H3 Hotspot Map below.")
  # Make Plotly scatter markers show a pointing hand cursor on hover
  st.markdown(
      """
      <style>
      .js-plotly-plot .plotly .nsewdrag {
          cursor: pointer !important;
      }
      </style>
      """,
      unsafe_allow_html=True,
  )

  day_labels = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
  cal_plot_df = yr_df.reset_index(drop=True).copy()
  cal_plot_df["date_str"] = cal_plot_df["acq_date"].dt.strftime("%Y-%m-%d")
  available_dates = sorted(cal_plot_df["date_str"].unique())

  default_date = (
      peak_row["acq_date"].strftime("%Y-%m-%d")
      if peak_row is not None
      else next(iter(available_dates), None)
  )

  # Store active date span as a (start_date, end_date) tuple in session_state
  if (
      "active_date_range" not in st.session_state
      or st.session_state.active_date_range[0] not in available_dates
      or st.session_state.active_date_range[1] not in available_dates
  ):
    st.session_state.active_date_range = (default_date, default_date)

  # Clicking a tile on the calendar sets the span to that single day
  prev_sel = st.session_state.get("cal_heatmap")
  if isinstance(prev_sel, dict) and "selection" in prev_sel:
    pts = prev_sel["selection"].get("points", ())
    if pts:
      pt = next(iter(pts), {})
      idx = pt.get("point_index", pt.get("pointIndex"))
      if idx is not None and 0 <= idx < len(cal_plot_df):
        clicked_dt = cal_plot_df["date_str"].iat[idx]
        st.session_state.active_date_range = (clicked_dt, clicked_dt)

  start_date, end_date = st.session_state.active_date_range

  # Highlight all tiles falling inside the selected date span with a cyan border
  border_colors = tuple(
      "#00E5FF"
      if (start_date <= d <= end_date)
      else "rgba(128,128,128,0.25)"
      for d in cal_plot_df["date_str"]
  )
  border_widths = tuple(
      2.5 if (start_date <= d <= end_date) else 1.0
      for d in cal_plot_df["date_str"]
  )

  fig_cal = go.Figure(
      data=go.Scatter(
          x=cal_plot_df["week_of_year"],
          y=cal_plot_df["day_of_week"],
          mode="markers",
          customdata=cal_plot_df["date_str"],
          hovertemplate=(
              "<b>Date:</b> %{customdata}<br>"
              f"<b>{selected_metric_label}:</b> %{{marker.color:.1f}}<br>"
              "<i>Click tile to inspect map</i><extra></extra>"
          ),
          marker=dict(
              symbol="square",
              size=15,
              color=cal_plot_df[metric_col],
              colorscale="YlOrRd",
              showscale=True,
              line=dict(width=border_widths, color=border_colors),
              colorbar=dict(title=selected_metric_label),
          ),
      )
  )

  fig_cal.update_layout(
      height=240,
      hovermode="closest",
      clickmode="event+select",
      dragmode=False,
      yaxis=dict(
          tickmode="array",
          tickvals=list(range(7)),
          ticktext=day_labels,
          autorange="reversed",
          title="",
          showgrid=False,
          zeroline=False,
          fixedrange=True,
      ),
      xaxis=dict(
          title="Week of Year",
          dtick=4,
          range=(-1, 54),
          showgrid=False,
          zeroline=False,
          fixedrange=True,
      ),
      margin=dict(l=20, r=20, t=15, b=25),
  )
    
  st.plotly_chart(
    fig_cal,
    config={
        "responsive": True,
        "autosize": True,
        "fillFrame": True,     # stretch-like behavior
    },
    on_select="rerun",
    selection_mode="points",
    key="cal_heatmap",
  )

  # --- Interactive H3 Hotspot Map ---
  st.markdown('<div id="hotspot-map-anchor"></div>', unsafe_allow_html=True)
  st.subheader("🗺️ Interactive H3 Hotspot Map")
  date_col, info_col = st.columns((1.2, 2))

  default_dt = (
      peak_row["acq_date"].strftime("%Y-%m-%d")
      if peak_row is not None
      else next(iter(available_dates), None)
  )

  def map_span_to_year(
      prev_range, target_year: int, valid_dates: list
  ) -> tuple:
    """Preserve the MM-DD date span across year switches, clamped to valid_dates."""
    if not valid_dates:
      return (None, None)
    if (
        not isinstance(prev_range, (tuple, list))
        or len(prev_range) != 2
        or not prev_range
        or not prev_range
    ):
      fallback = (
          peak_row["acq_date"].strftime("%Y-%m-%d")
          if peak_row is not None
          else next(iter(valid_dates))
      )
      return (fallback, fallback)

    # Extract MM-DD from previous start and end dates (YYYY-MM-DD -> -MM-DD)
    prev_start, prev_end = prev_range
    new_start = f"{target_year}{str(prev_start)[4:]}"
    new_end = f"{target_year}{str(prev_end)[4:]}"

    # Clamp to available dates in the target year (handles Feb 29 & partial current year)
    first_avail = next(iter(valid_dates))
    last_avail = valid_dates[-1]

    if new_start not in valid_dates:
      later = tuple(d for d in valid_dates if d >= new_start)
      new_start = next(iter(later), last_avail)
    if new_end not in valid_dates:
      earlier = tuple(d for d in valid_dates if d <= new_end)
      new_end = earlier[-1] if earlier else first_avail

    if new_start > new_end:
      new_start = new_end
    return (new_start, new_end)


  # Initialize or remap the active date span whenever the year changes
  st.session_state.active_date_range = map_span_to_year(
      st.session_state.get("active_date_range"),
      selected_year,
      available_dates,
  )


  start_date, end_date = st.session_state.active_date_range

  with date_col:
    selected_range = st.select_slider(
        "Select Date Span",
        options=available_dates,
        value=(start_date, end_date),
        key=f"map_date_range_{selected_year}_{start_date}_{end_date}",
    )
    if tuple(selected_range) != tuple(st.session_state.active_date_range):
      st.session_state.active_date_range = tuple(selected_range)
      st.session_state.scroll_to_map = True
      st.rerun()


  # Retain focus at the map view when the date range changes
  if st.session_state.pop("scroll_to_map", False):
    components.html(
        """
        <script>
          const anchor = window.parent.document.getElementById('hotspot-map-anchor');
          if (anchor) {
            anchor.scrollIntoView({ behavior: 'instant', block: 'start' });
          }
        </script>
        """,
        height=0,
    )

  start_date, end_date = st.session_state.active_date_range
  span_label = (
      start_date if start_date == end_date else f"{start_date} to {end_date}"
  )

  if not map_df.empty:
    map_dates = map_df["acq_date"].astype(str).str.slice(0, 10)
    span_cells = map_df[(map_dates >= start_date) & (map_dates <= end_date)]

    if selected_region != "Western_US":
      span_cells = span_cells[span_cells["region_id"] == selected_region]

    # Aggregate detections per H3 cell across the selected date span
    if not span_cells.empty:
      agg_dict = {
          "region_id": ("region_id", "first"),
          "lat": ("lat", "first"),
          "lon": ("lon", "first"),
          "total_frp": ("total_frp", "sum"),
          "days_in_span": ("acq_date", "nunique"),
          "source": ("source", lambda s: ", ".join(sorted(s.unique()))),
      }
      if "active_range" in span_cells.columns:
        agg_dict["active_range"] = ("active_range", "first")

      day_cells = span_cells.groupby("h3_cell", as_index=False).agg(**agg_dict)
      day_cells["total_frp"] = day_cells["total_frp"].round(1)
    else:
      day_cells = span_cells

    with info_col:
      st.info(
          f"Showing **{len(day_cells)} active H3 fire hexagons** (~3.2 km"
          f" resolution) in **{selected_region}** during **{span_label}**."
      )

    if not day_cells.empty:
      view = REGION_VIEWPORTS.get(
          selected_region, REGION_VIEWPORTS["Western_US"]
      )

      # Scale projection_scale so the Western US or selected state fits comfortably
      geo_scale = (
          1.45
          if selected_region == "Western_US"
          else max(2.8, view["zoom"] * 0.72)
      )

      hover_cols = {
          "region_id": False,
          "h3_cell": False,
          "days_in_span": True,
          "total_frp": ":.1f",
          "source": True,
          "lat": ":.2f",
          "lon": ":.2f",
      }
      if "active_range" in day_cells.columns:
        hover_cols["active_range"] = True

      fig_map = px.scatter_geo(
          day_cells,
          lat="lat",
          lon="lon",
          size="total_frp",
          color="total_frp",
          color_continuous_scale="YlOrRd",
          size_max=18,
          scope="usa",
          hover_name="region_id",
          hover_data=hover_cols,
          labels={
              "days_in_span": "Active Days in Span",
              "active_range": "Full Episode Span",
              "total_frp": "Cumulative Fire Radiative Power (MW)",
              "source": "Sensor Source",
              "lat": "Latitude",
              "lon": "Longitude",
          },
      )

      # Draw bright cyan border around the active state (or all 11 Western US states)
      states_geojson = load_us_states_geojson()
      highlight_geojson = filter_selected_state_geojson(
          states_geojson, selected_region
      )
      if highlight_geojson:
        for feat in highlight_geojson.get("features", ()):
          geom = feat.get("geometry", {})
          gtype = geom.get("type")
          coords_list = (
              geom.get("coordinates", ())
              if gtype == "Polygon"
              else [
                  poly
                  for multi in geom.get("coordinates", ())
                  for poly in multi
              ]
          )
          for ring in coords_list:
            lons, lats = zip(*ring)
            fig_map.add_trace(
                go.Scattergeo(
                    lon=lons,
                    lat=lats,
                    mode="lines",
                    line=dict(width=2.5, color="#00E5FF"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

      fig_map.update_geos(
          center={"lat": view["lat"], "lon": view["lon"]},
          projection_scale=geo_scale,
          showland=True,
          landcolor="#1e1e1e",
          showlakes=True,
          lakecolor="#111111",
          showsubunits=True,
          subunitcolor="rgba(255, 255, 255, 0.35)",
          subunitwidth=1.0,
          showcountries=True,
          countrycolor="rgba(255, 255, 255, 0.55)",
          bgcolor="rgba(0,0,0,0)",
      )

      fig_map.update_layout(
          height=470,
          margin=dict(l=0, r=0, t=0, b=5),
      )

      st.plotly_chart(fig_map, 
              config={
              "responsive": True,
              "autosize": True,
              "fillFrame": True,     # stretch-like behavior
          }, key="h3_hotspot_map")
      st.divider()
    else:
      st.warning(
          f"No active fire cells recorded for {selected_region} during"
          f" {span_label}."
      )
      st.divider()
  else:
    st.warning(
        "Run the updated `harmonize_firms.py` to generate `harmonized_h3_map.csv`."
    )


  st.subheader(
      f"{selected_year} {selected_metric_label} vs. Multi-Year Historical Median"
  )

  if metric_col == "harmonized_frp_mw":
    y_actual = yr_df["harmonized_frp_mw"]
    y_median = (
        yr_df["historical_median_frp"]
        if "historical_median_frp" in yr_df.columns
        else pd.Series(0.0, index=yr_df.index)
    )
    y_axis_title = "Harmonized Fire Radiative Power (MW)"
  elif metric_col == "anomaly_ratio":
    y_actual = yr_df["anomaly_ratio"]
    y_median = pd.Series(1.0, index=yr_df.index)
    y_axis_title = "Anomaly Ratio (1.0 = Normal)"
  else:
    y_actual = yr_df["harmonized_active_cells"]
    y_median = yr_df["historical_median_cells"]
    y_axis_title = "Harmonized Active Fire Cells"

  fig_ts = go.Figure()
  fig_ts.add_trace(
      go.Scatter(
          x=yr_df["acq_date"],
          y=y_median,
          mode="lines",
          name="Historical Climatological Median",
          line=dict(color="gray", dash="dash", width=2),
      )
  )
  fig_ts.add_trace(
      go.Scatter(
          x=yr_df["acq_date"],
          y=y_actual,
          mode="lines",
          name=f"{selected_year} {selected_metric_label}",
          fill="tonexty",
          fillcolor="rgba(255, 87, 34, 0.25)",
          line=dict(color="#E64A19", width=2),
      )
  )

  fig_ts.update_layout(
      height=310,
      xaxis_title="Date",
      yaxis_title=y_axis_title,
      hovermode="x unified",
      dragmode=False,
      xaxis=dict(fixedrange=True),
      yaxis=dict(fixedrange=True),
      margin=dict(l=20, r=20, t=15, b=25),
  )

  st.plotly_chart(
    fig_ts,
    config={
        "responsive": True,
        "autosize": True,
        "fillFrame": True,     # stretch-like behavior
    },
    key="historical_median_ts",
  )


  with st.expander(f"Inspect & Export Displayed Fires ({span_label})"):
    # 1. Filter daily summary rows to the active date span
    yr_dates = yr_df["acq_date"].dt.strftime("%Y-%m-%d")
    span_summary_df = yr_df[(yr_dates >= start_date) & (yr_dates <= end_date)].drop(
        columns=("year", "day_of_week", "week_of_year"), errors="ignore"
    )

    st.markdown("**1. Daily Harmonized Fire Summary (Selected Span)**")
    st.dataframe(span_summary_df, width="stretch")

    col_dl1, col_dl2 = st.columns(2)
    safe_span = (
        start_date if start_date == end_date else f"{start_date}_to_{end_date}"
    )

    with col_dl1:
      st.download_button(
          label=f"⬇️ Download Daily Summary CSV ({safe_span})",
          data=span_summary_df.to_csv(index=False).encode("utf-8"),
          file_name=f"harmonized_summary_{selected_region}_{safe_span}.csv",
          mime="text/csv",
          key="dl_summary_csv",
      )

    with col_dl2:
      if not day_cells.empty:
        export_hotspots = day_cells.copy()
        export_hotspots.insert(0, "span_start", start_date)
        export_hotspots.insert(1, "span_end", end_date)
        st.download_button(
            label=f"⬇️ Download Map Hotspots CSV ({len(export_hotspots)} cells)",
            data=export_hotspots.to_csv(index=False).encode("utf-8"),
            file_name=f"displayed_hotspots_{selected_region}_{safe_span}.csv",
            mime="text/csv",
            key="dl_hotspots_csv",
        )


# ==============================================================================
# TAB 2: SENSOR CALIBRATION VALIDATION (RAW VS. CALIBRATED MODIS VS. VIIRS)
# ==============================================================================
with tab2:
  st.subheader(f"MODIS-to-VIIRS Cross-Calibration Performance ({selected_region})")
  st.markdown(
      "Because VIIRS (375 m) detects smaller and cooler fires than MODIS (1"
      " km), raw MODIS counts systematically undercount fire activity. Below"
      " is the validation on overlapping observation dates."
  )

  if val_df.empty:
    st.warning(
        "Run the updated `harmonize_firms.py` to generate"
        " `calibration_validation.csv`."
    )
  else:
    reg_val = val_df[val_df["region_id"] == selected_region].sort_values(
        "acq_date"
    )

    if len(reg_val) < 2:
      st.info("Not enough overlapping observations in this sub-region.")
    else:
      y_true = reg_val["viirs_observed_cells"]
      y_raw = reg_val["raw_modis_cells"]
      y_cal = reg_val["calibrated_modis_cells"]

      raw_r2 = r2_score(y_true, y_raw)
      cal_r2 = r2_score(y_true, y_cal)
      raw_rmse = np.sqrt(mean_squared_error(y_true, y_raw))
      cal_rmse = np.sqrt(mean_squared_error(y_true, y_cal))

      m1, m2, m3, m4 = st.columns(4)
      m1.metric("Raw MODIS R²", f"{raw_r2:.3f}")
      m2.metric(
          "Calibrated MODIS R²",
          f"{cal_r2:.3f}",
          delta=f"{cal_r2 - raw_r2:+.3f}",
      )
      m3.metric("Raw MODIS RMSE (Cells)", f"{raw_rmse:.1f}")
      m4.metric(
        "Calibrated MODIS RMSE (Cells)",
        f"{cal_rmse:.1f}",
        delta=f"{cal_rmse - raw_rmse:+.1f}",
        delta_color="inverse",
    )
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("1:1 Parity Plot (Before vs. After Calibration)")
        max_axis = max(y_true.max(), y_raw.max(), y_cal.max()) * 1.05
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
        go.Scatter(
        x=y_true,
        y=y_raw,
        mode="markers",
        name="Raw MODIS (Uncalibrated)",
        marker=dict(color="rgba(158, 158, 158, 0.55)", size=7),
        )
        )
        fig_scatter.add_trace(
        go.Scatter(
        x=y_true,
        y=y_cal,
        mode="markers",
        name="Calibrated MODIS (Unified)",
        marker=dict(color="rgba(230, 74, 25, 0.75)", size=7),
        )
        )
        fig_scatter.add_trace(
        go.Scatter(
        x=[0, max_axis],
        y=[0, max_axis],
        mode="lines",
        name="Ideal 1:1 Match",
        line=dict(color="black", dash="dash"),
        )
        )
        fig_scatter.update_layout(
            height=380,
            xaxis_title="VIIRS Observed Active Cells",
            yaxis_title="MODIS Active Cells",
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            margin=dict(l=20, r=20, t=20, b=30),
        )
        
        st.plotly_chart(
            fig_scatter,
            config={
                "responsive": True,
                "autosize": True,
                "fillFrame": True,     # stretch-like behavior
            },
            key="calibration_parity_scatter",
        )

        with c_right:
            st.markdown("Overlap Time-Series Comparison")
            fig_comp = go.Figure()
            fig_comp.add_trace(
            go.Scatter(
            x=reg_val["acq_date"],
            y=y_true,
            mode="lines",
            name="VIIRS Observed (Target)",
            line=dict(color="#1E88E5", width=2),
            )
            )
            fig_comp.add_trace(
            go.Scatter(
            x=reg_val["acq_date"],
            y=y_raw,
            mode="lines",
            name="Raw MODIS",
            line=dict(color="gray", dash="dot", width=1.5),
            )
            )
            fig_comp.add_trace(
            go.Scatter(
            x=reg_val["acq_date"],
            y=y_cal,
            mode="lines",
            name="Calibrated MODIS",
            line=dict(color="#E64A19", width=2),
            )
            )
            fig_comp.update_layout(
            height=380,
            xaxis_title="Overlap Date",
            yaxis_title="Active Fire Cells",
            hovermode="x unified",
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            margin=dict(l=20, r=20, t=20, b=30),
            )

            st.plotly_chart(fig_comp, 
                            config={
                            "responsive": True,
                            "autosize": True,
                            "fillFrame": True,     # stretch-like behavior
                                }, key="calibration_overlap_ts")
st.divider()
st.markdown(
    """
    <div style="font-size: 0.85rem; opacity: 0.85; line-height: 1.5;">
      <strong>📡 Data Sources &amp; Attribution</strong><br>
      Fire hotspot detections sourced via the
      <a href="https://nasa.gov" target="_blank">NASA FIRMS Area REST API</a>
      and <a href="https://nasa.gov" target="_blank">NASA FIRMS Archive Download Portal</a>
      (Western US: 125.0°W–102.0°W, 31.0°N–49.0°N; Nov 2000–Nov 2026):
      <ul style="margin-top: 0.25rem; margin-bottom: 0.35rem;">
        <li><strong>MODIS Collection 6.1 (1 km):</strong> <code>MODIS_SP</code> (Nov 2000–2026 science quality) &amp; <code>MODIS_NRT</code> (recent ~90-day near real-time) —
          <a href="https://nasa.gov" target="_blank">NASA Earthdata FIRMS Active Fire Data</a>.
        </li>
        <li><strong>VIIRS Active Fire (375 m):</strong> <code>VIIRS_SNPP_SP</code> (Jan 2012–2025), <code>VIIRS_NOAA20_SP</code> (2026 science quality), &amp; <code>VIIRS_NOAA20_NRT</code> (recent ~90-day near real-time) —
          <a href="https://nasa.gov" target="_blank">NASA FIRMS Active Fire Portal</a>.
        </li>
      </ul>
      <em>We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS), part of NASA's Earth Science Data and Information System (ESDIS).</em>
    </div>
    """,
    unsafe_allow_html=True,
)
