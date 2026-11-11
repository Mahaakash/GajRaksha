"""
GajRaksha - Model Validation Against WII Published Conflict Hotspots
=====================================================================
Performs spatial sanity check of elephant presence risk predictions
in South Bengal against Wildlife Institute of India (WII) published
conflict hotspots (Mathur et al., TR-0039-2026).

Computes:
1. Share of top 10% highest-risk cells in South Bengal within 5 km of a hotspot
2. Expected share within 5 km by chance (baseline over all South Bengal landscape)
3. Enrichment factor / Risk ratio
4. Visual verification map: hotspots overlaid on the risk heatmap

Saves output to:
- models/india_sanity_check.json
- data/south_bengal_hotspots_risk_map.png
"""

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

GRID_FILE = DATA_DIR / "grid_features_south_bengal.parquet"
HOTSPOTS_FILE = DATA_DIR / "india_hotspots.csv"
MODEL_FILE = MODELS_DIR / "elephant_presence_risk_xgb.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"
DISTRICTS_GEOJSON = DATA_DIR / "regions" / "south_bengal_districts.geojson"

OUTPUT_JSON = MODELS_DIR / "india_sanity_check.json"
OUTPUT_MAP_PNG = DATA_DIR / "south_bengal_hotspots_risk_map.png"


def project_coords_km(lons, lats, ref_lat=22.8, ref_lon=86.9):
    """Projects lon/lat to metric kilometers via equirectangular projection."""
    r = 6371.0
    lat_rad = np.radians(ref_lat)
    x = r * np.radians(lons - ref_lon) * np.cos(lat_rad)
    y = r * np.radians(lats - ref_lat)
    return np.column_stack([x, y])


def run_sanity_check():
    print("=" * 65)
    print("GAJRAKSHA: WII PUBLISHED CONFLICT HOTSPOT SANITY CHECK")
    print("=" * 65)

    # 1. Load trained XGBoost model
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print(f"✓ Loaded XGBoost model from: {MODEL_FILE.name}")

    # 2. Load South Bengal grid
    if not GRID_FILE.exists():
        raise FileNotFoundError(f"Grid file not found: {GRID_FILE}")
    df_grid = pd.read_parquet(GRID_FILE)
    total_cells = len(df_grid)
    print(f"✓ Loaded South Bengal grid: {total_cells:,} cells (~1 km resolution)")

    # 3. Score all South Bengal cells using transferable model features
    # Standard operational nighttime baseline: 22:00 local time, is_night=1
    eval_hour = 22
    eval_night = 1
    eval_rain_7d = 2.0
    eval_rain_30d = 15.0

    X_input = pd.DataFrame({
        "hour_of_day": np.full(total_cells, eval_hour),
        "is_night": np.full(total_cells, eval_night),
        "rainfall_last_7_days": np.full(total_cells, eval_rain_7d),
        "rainfall_last_30_days": np.full(total_cells, eval_rain_30d),
        "speed_kmh": np.full(total_cells, 2.2),
        "step_distance_km": np.full(total_cells, 1.8),
        "heading_change_deg": np.full(total_cells, 30.0),
        "dist_to_nearest_village_km": df_grid["distance_to_village_km"].values
    })

    print("Computing risk probabilities for all South Bengal cells...")
    df_grid["risk_score"] = model.predict_proba(X_input)[:, 1]

    # 4. Identify Top 10% highest-risk cells
    p90_threshold = float(df_grid["risk_score"].quantile(0.90))
    top_10_mask = df_grid["risk_score"] >= p90_threshold
    df_top_10 = df_grid[top_10_mask]
    n_top_10 = len(df_top_10)

    print(f"Top 10% Risk Score Threshold (p90): {p90_threshold:.4f}")
    print(f"Top 10% Cell Count:                {n_top_10:,} cells")

    # 5. Load and filter geocoded WII hotspots
    if not HOTSPOTS_FILE.exists():
        raise FileNotFoundError(f"Hotspots CSV not found: {HOTSPOTS_FILE}")
    df_hotspots = pd.read_csv(HOTSPOTS_FILE)
    df_geocoded = df_hotspots[df_hotspots["geocoded"] == True].copy()
    n_geocoded = len(df_geocoded)
    n_total_hotspots = len(df_hotspots)
    n_unlocated = n_total_hotspots - n_geocoded

    print(f"✓ Loaded {n_total_hotspots} conflict hotspot records from WII report")
    print(f"  Located in OpenStreetMap: {n_geocoded} records ({n_geocoded/n_total_hotspots*100:.1f}%)")
    print(f"  Could not locate:         {n_unlocated} records ({n_unlocated/n_total_hotspots*100:.1f}%)")

    # 6. Spatial proximity analysis via KD-Tree
    ref_lat, ref_lon = 22.8, 86.9
    hotspot_xy = project_coords_km(df_geocoded["longitude"].values, df_geocoded["latitude"].values, ref_lat, ref_lon)
    grid_xy = project_coords_km(df_grid["location_lon"].values, df_grid["location_lat"].values, ref_lat, ref_lon)

    tree = cKDTree(hotspot_xy)
    dists_km, nearest_idx = tree.query(grid_xy)
    df_grid["dist_to_nearest_hotspot_km"] = dists_km
    df_grid["within_5km_hotspot"] = dists_km <= 5.0

    # Share of Top 10% cells within 5 km
    top_10_within_5km = (df_grid[top_10_mask]["within_5km_hotspot"]).sum()
    share_top_10 = float(top_10_within_5km / n_top_10)

    # Share of ALL cells within 5 km (Baseline chance expectation)
    all_within_5km = (df_grid["within_5km_hotspot"]).sum()
    share_by_chance = float(all_within_5km / total_cells)

    enrichment_ratio = float(share_top_10 / share_by_chance) if share_by_chance > 0 else 0.0

    print("\n" + "=" * 65)
    print("SANITY CHECK RESULTS:")
    print("=" * 65)
    print(f"Share of Top 10% Highest-Risk Cells within 5 km of WII Hotspot: {share_top_10 * 100:.2f}% ({top_10_within_5km:,} / {n_top_10:,})")
    print(f"Share Expected by Chance (Landscape Baseline Proximity):        {share_by_chance * 100:.2f}% ({all_within_5km:,} / {total_cells:,})")
    print(f"Enrichment Factor (Risk vs Chance Ratio):                       {enrichment_ratio:.2f}x")
    print("=" * 65)

    # 7. Save sanity check metrics to JSON
    sanity_results = {
        "report_title": "Human-elephant conflict in the State of West Bengal, India (2000-2025)",
        "report_authors": "Mathur et al., Wildlife Institute of India & Project Elephant",
        "report_number": "TR-0039-2026",
        "target_region": "south_bengal",
        "districts_evaluated": ["Jhargram", "Paschim Medinipur", "Bankura", "Purulia"],
        "total_hotspots_listed": int(n_total_hotspots),
        "hotspots_geocoded_osm": int(n_geocoded),
        "hotspots_unlocated_osm": int(n_unlocated),
        "geocoding_success_rate_pct": round(float(n_geocoded / n_total_hotspots * 100), 2),
        "evaluation_parameters": {
            "proximity_radius_km": 5.0,
            "top_risk_percentile": 10.0,
            "top_risk_threshold_score": round(p90_threshold, 4),
            "total_grid_cells": int(total_cells),
            "top_10_cell_count": int(n_top_10)
        },
        "metrics": {
            "top_10_pct_cells_within_5km_count": int(top_10_within_5km),
            "top_10_pct_cells_within_5km_share": round(share_top_10, 4),
            "top_10_pct_cells_within_5km_percentage": round(share_top_10 * 100, 2),
            "baseline_chance_cells_within_5km_count": int(all_within_5km),
            "baseline_chance_cells_within_5km_share": round(share_by_chance, 4),
            "baseline_chance_cells_within_5km_percentage": round(share_by_chance * 100, 2),
            "enrichment_ratio": round(enrichment_ratio, 2)
        }
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sanity_results, f, indent=2)
    print(f"✓ Saved sanity check metrics to: {OUTPUT_JSON}")

    # 8. Render Visual Map: Hotspots over Risk Heatmap
    print("\nRendering risk heatmap with WII hotspot overlay...")
    render_hotspots_map(df_grid, df_geocoded, share_top_10, share_by_chance, enrichment_ratio)

    return sanity_results


def render_hotspots_map(df_grid: pd.DataFrame, df_hotspots: pd.DataFrame, share_top_10: float, share_chance: float, enrichment: float):
    fig, ax = plt.subplots(figsize=(13, 11), dpi=200)

    # A. Draw District Boundaries
    if DISTRICTS_GEOJSON.exists():
        with open(DISTRICTS_GEOJSON, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        for feat in geojson["features"]:
            props = feat.get("properties", {})
            name = props.get("district", props.get("name", "District"))
            geom = feat["geometry"]
            coords = geom["coordinates"]
            polys = [coords] if geom["type"] == "Polygon" else coords

            for poly in polys:
                poly_pts = np.array(poly[0])
                ax.plot(poly_pts[:, 0], poly_pts[:, 1], color="#333333", linewidth=1.5, zorder=3)

    # B. Plot Risk Heatmap Scatter
    scatter = ax.scatter(
        df_grid["location_lon"],
        df_grid["location_lat"],
        c=df_grid["risk_score"],
        cmap="YlOrRd",
        s=14,
        alpha=0.65,
        edgecolors="none",
        zorder=2,
        label="Presence Risk Grid (~1 km)"
    )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Predicted Elephant Presence Risk Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # C. Overlay Geocoded WII Conflict Hotspots
    # Separate by type / conflict count
    ax.scatter(
        df_hotspots["longitude"],
        df_hotspots["latitude"],
        c="#1d3557",
        marker="o",
        s=32,
        edgecolors="white",
        linewidth=0.8,
        zorder=5,
        label=f"WII Conflict Hotspots (n={len(df_hotspots)})"
    )

    # Highlight top conflict sites with labels
    top_conflict = df_hotspots.sort_values(by="conflict_count", ascending=False).drop_duplicates(subset=["name"]).head(12)
    for _, h in top_conflict.iterrows():
        ax.annotate(
            f"{h['name']}",
            (h["longitude"], h["latitude"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#0b132b",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#457b9d", lw=0.6),
            zorder=6
        )

    # Title and Metadata Annotations
    ax.set_title(
        "GajRaksha - Model Validation Against Published WII Conflict Hotspots\n"
        "Wildlife Institute of India (2000-2025) Report Hotspots vs Elephant Presence Risk Heatmap",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=1)

    # Stats Summary Box
    stats_text = (
        f"Validation Metrics (5 km Radius):\n"
        f"• Top 10% Risk Cells near Hotspot: {share_top_10 * 100:.1f}%\n"
        f"• Baseline Random Chance:          {share_chance * 100:.1f}%\n"
        f"• Enrichment Ratio:                {enrichment:.2f}x\n"
        f"• Geocoded Hotspots Overlay:       {len(df_hotspots):,} sites"
    )
    ax.text(
        0.02, 0.03, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#333333", alpha=0.92, lw=1.0),
        zorder=7
    )

    ax.legend(loc="upper left", framealpha=0.92, fontsize=10)
    plt.tight_layout()
    OUTPUT_MAP_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_MAP_PNG, dpi=200)
    plt.close()
    print(f"✓ Saved validation map image to: {OUTPUT_MAP_PNG}")


if __name__ == "__main__":
    run_sanity_check()

