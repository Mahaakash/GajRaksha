"""
GajRaksha - Landscape Feature Grid Builder & Cross-Region Comparison
=====================================================================
Builds ~1 km spatial feature grids for:
1. 'south_bengal' (Target Area: Jhargram, Paschim Medinipur, Bankura, Purulia)
2. 'kruger' (Training Area: Elephant GPS movement range)

Landscape Features:
- distance_to_forest_edge_km: Proximity to nearest forest patch / reserve edge
- distance_to_water_km: Proximity to rivers, canals, reservoirs, and water bodies
- distance_to_village_km: Proximity to human settlements (from OpenStreetMap for South Bengal)
- crop_density: Agricultural cropland intensity (0.0 to 1.0)

Generates:
- data/grid_features_south_bengal.parquet
- data/grid_features_kruger.parquet
- data/feature_distribution_comparison.png (Side-by-side histograms)
- Out-of-distribution (OOD) analysis identifying cells outside Kruger training ranges
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REGIONS_GEOJSON = DATA_DIR / "regions" / "south_bengal_districts.geojson"
KRUGER_VILLAGES_CSV = DATA_DIR / "raw" / "villages.csv"

OUTPUT_PARQUET_SB = DATA_DIR / "grid_features_south_bengal.parquet"
OUTPUT_PARQUET_KRUGER = DATA_DIR / "grid_features_kruger.parquet"
OUTPUT_COMPARISON_PNG = DATA_DIR / "feature_distribution_comparison.png"


def project_coords_km(lons, lats, ref_lat, ref_lon):
    """
    Projects (lon, lat) degrees to approximate metric coordinates (x_km, y_km)
    using equirectangular projection centered at reference coordinate.
    """
    r = 6371.0
    lat_rad = np.radians(ref_lat)
    x = r * np.radians(lons - ref_lon) * np.cos(lat_rad)
    y = r * np.radians(lats - ref_lat)
    return np.column_stack([x, y])


def extract_osm_points(cache_json_path: Path) -> np.ndarray:
    """Extracts (lon, lat) coordinates from cached OpenStreetMap Overpass JSON."""
    if not cache_json_path.exists():
        return np.empty((0, 2))

    with open(cache_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = []
    for el in data.get("elements", []):
        if "lat" in el and "lon" in el:
            pts.append([el["lon"], el["lat"]])
        elif "geometry" in el and el["geometry"]:
            for g in el["geometry"]:
                if "lat" in g and "lon" in g:
                    pts.append([g["lon"], g["lat"]])
    if not pts:
        return np.empty((0, 2))
    return np.array(pts)


def build_south_bengal_grid(resolution_deg: float = 0.01) -> pd.DataFrame:
    """
    Constructs a ~1 km grid clipped to the 4 district boundaries of South Bengal
    and calculates all 4 landscape features.
    """
    print("=" * 65)
    print("1. BUILDING SOUTH BENGAL SPATIAL FEATURE GRID (~1 km resolution)")
    print("=" * 65)

    with open(REGIONS_GEOJSON, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # 1. Build polygon paths for district boundary clipping
    district_paths = []
    all_lons, all_lats = [], []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        gtype = geom["type"]
        coords = geom["coordinates"]
        if gtype == "Polygon":
            poly = np.array(coords[0])
            district_paths.append(MplPath(poly))
            all_lons.extend(poly[:, 0])
            all_lats.extend(poly[:, 1])
        elif gtype == "MultiPolygon":
            for p in coords:
                poly = np.array(p[0])
                district_paths.append(MplPath(poly))
                all_lons.extend(poly[:, 0])
                all_lats.extend(poly[:, 1])

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    print(f"Bounding box: Lon [{min_lon:.3f}, {max_lon:.3f}], Lat [{min_lat:.3f}, {max_lat:.3f}]")

    # Regular ~1 km grid coordinates
    grid_lons = np.arange(min_lon, max_lon, resolution_deg)
    grid_lats = np.arange(min_lat, max_lat, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    flat_pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    # Clip to district polygon interior
    print(f"Filtering {len(flat_pts):,} candidate grid points within district boundaries...")
    mask = np.zeros(len(flat_pts), dtype=bool)
    for path in district_paths:
        mask |= path.contains_points(flat_pts)

    sb_pts = flat_pts[mask]
    sb_lons = sb_pts[:, 0]
    sb_lats = sb_pts[:, 1]
    ref_lon, ref_lat = float(np.mean(sb_lons)), float(np.mean(sb_lats))
    print(f"✓ Clipped to {len(sb_pts):,} active ~1 km grid cells inside South Bengal districts.")

    # Project grid points to metric km
    grid_xy = project_coords_km(sb_lons, sb_lats, ref_lat, ref_lon)

    # 2. Distance to Village (from OpenStreetMap)
    print("\n[A] Calculating distance to nearest village (from OpenStreetMap)...")
    osm_villages_file = CACHE_DIR / "osm_villages_south_bengal.json"
    village_pts = extract_osm_points(osm_villages_file)
    print(f"  Loaded {len(village_pts):,} OpenStreetMap village locations.")
    v_xy = project_coords_km(village_pts[:, 0], village_pts[:, 1], ref_lat, ref_lon)
    v_tree = cKDTree(v_xy)
    dist_to_village, _ = v_tree.query(grid_xy)
    dist_to_village = np.round(dist_to_village, 2)

    # 3. Distance to Water (from OpenStreetMap)
    print("[B] Calculating distance to nearest water body / river...")
    osm_water_file = CACHE_DIR / "osm_water_south_bengal.json"
    water_pts = extract_osm_points(osm_water_file)
    print(f"  Loaded {len(water_pts):,} water/river coordinates.")
    w_xy = project_coords_km(water_pts[:, 0], water_pts[:, 1], ref_lat, ref_lon)
    w_tree = cKDTree(w_xy)
    dist_to_water, _ = w_tree.query(grid_xy)
    dist_to_water = np.round(dist_to_water, 2)

    # 4. Distance to Forest Edge (from OpenStreetMap)
    print("[C] Calculating distance to nearest forest edge / reserve...")
    osm_forest_file = CACHE_DIR / "osm_forests_south_bengal.json"
    forest_pts = extract_osm_points(osm_forest_file)
    print(f"  Loaded {len(forest_pts):,} forest/wood boundary coordinates.")
    f_xy = project_coords_km(forest_pts[:, 0], forest_pts[:, 1], ref_lat, ref_lon)
    f_tree = cKDTree(f_xy)
    dist_to_forest, _ = f_tree.query(grid_xy)
    dist_to_forest = np.round(dist_to_forest, 2)

    # 5. Crop Density
    # In South Bengal: Cropland density is highest in the rural agricultural plains away from deep forest
    # and near farming settlements (0.45 to 0.88), and lowest inside dense forest tracts (0.05 to 0.20).
    print("[D] Estimating crop density (cropland intensity)...")
    # Base agricultural density inversely proportional to forest distance and village proximity
    crop_density = 0.85 - 0.65 * np.exp(-dist_to_forest / 2.5) + 0.10 * np.exp(-dist_to_village / 3.0)
    # Add natural local spatial variability
    np.random.seed(42)
    crop_density += np.random.normal(0, 0.04, len(crop_density))
    crop_density = np.clip(crop_density, 0.05, 0.95).round(3)

    df_sb = pd.DataFrame({
        "cell_id": [f"SB_CELL_{i:05d}" for i in range(len(sb_pts))],
        "region": "south_bengal",
        "location_lon": np.round(sb_lons, 4),
        "location_lat": np.round(sb_lats, 4),
        "distance_to_forest_edge_km": dist_to_forest,
        "distance_to_water_km": dist_to_water,
        "distance_to_village_km": dist_to_village,
        "crop_density": crop_density
    })

    df_sb.to_parquet(OUTPUT_PARQUET_SB, index=False, engine="pyarrow")
    print(f"✓ Saved South Bengal feature grid to: {OUTPUT_PARQUET_SB.name}")
    return df_sb


def build_kruger_grid(resolution_deg: float = 0.01) -> pd.DataFrame:
    """
    Constructs a ~1 km grid for the Kruger elephant movement training landscape
    with the exact same 4 landscape features.
    """
    print("\n" + "=" * 65)
    print("2. BUILDING KRUGER TRAINING SPATIAL FEATURE GRID (~1 km resolution)")
    print("=" * 65)

    # Kruger bounding box matching tracking data
    min_lon, max_lon = 31.20, 32.00
    min_lat, max_lat = -25.25, -24.10

    grid_lons = np.arange(min_lon, max_lon, resolution_deg)
    grid_lats = np.arange(min_lat, max_lat, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    k_lons = lon_grid.ravel()
    k_lats = lat_grid.ravel()

    ref_lon, ref_lat = 31.60, -24.70
    grid_xy = project_coords_km(k_lons, k_lats, ref_lat, ref_lon)

    # 1. Distance to Village (Kruger western boundary villages)
    v_df = pd.read_csv(KRUGER_VILLAGES_CSV)
    v_xy = project_coords_km(v_df["location_long"].values, v_df["location_lat"].values, ref_lat, ref_lon)
    v_tree = cKDTree(v_xy)
    dist_to_village, _ = v_tree.query(grid_xy)
    dist_to_village = np.round(dist_to_village, 2)

    # 2. Distance to Water (Major Kruger river corridors: Sabie -25.0, Crocodile -25.4, Sand River -24.9)
    # Modeled river centerlines across Kruger
    river_lons = np.linspace(31.2, 32.0, 100)
    sabie_pts = np.column_stack([river_lons, np.full_like(river_lons, -25.02)])
    croc_pts = np.column_stack([river_lons, np.full_like(river_lons, -25.40)])
    sand_pts = np.column_stack([river_lons, np.full_like(river_lons, -24.90)])
    water_pts = np.vstack([sabie_pts, croc_pts, sand_pts])

    w_xy = project_coords_km(water_pts[:, 0], water_pts[:, 1], ref_lat, ref_lon)
    w_tree = cKDTree(w_xy)
    dist_to_water, _ = w_tree.query(grid_xy)
    dist_to_water = np.round(dist_to_water, 2)

    # 3. Distance to Forest / Reserve Edge (Park boundary fenceline at ~31.3°E)
    # Inside park (> 31.32°E), distance to fence increases eastward
    dist_to_fence = np.maximum(0.0, (k_lons - 31.32) * 105.0)
    # Outside park (< 31.32°E), distance away from park wilderness into communal land
    dist_outside = np.maximum(0.0, (31.32 - k_lons) * 105.0)
    dist_to_forest = np.round(np.where(k_lons >= 31.32, dist_to_fence, dist_outside), 2)

    # 4. Crop Density
    # In Kruger: 0 inside park reserve (protected wilderness); outside fence in communal farming, 0.1 to 0.45
    np.random.seed(42)
    crop_density = np.where(k_lons >= 31.32, 0.0, 0.15 + 0.25 * np.exp(-dist_outside / 5.0) + np.random.normal(0, 0.03, len(k_lons)))
    crop_density = np.clip(crop_density, 0.0, 0.50).round(3)

    df_kruger = pd.DataFrame({
        "cell_id": [f"KRUGER_CELL_{i:05d}" for i in range(len(k_lons))],
        "region": "kruger",
        "location_lon": np.round(k_lons, 4),
        "location_lat": np.round(k_lats, 4),
        "distance_to_forest_edge_km": dist_to_forest,
        "distance_to_water_km": dist_to_water,
        "distance_to_village_km": dist_to_village,
        "crop_density": crop_density
    })

    df_kruger.to_parquet(OUTPUT_PARQUET_KRUGER, index=False, engine="pyarrow")
    print(f"✓ Generated {len(df_kruger):,} grid cells for Kruger training region.")
    print(f"✓ Saved Kruger feature grid to: {OUTPUT_PARQUET_KRUGER.name}")
    return df_kruger


def compare_feature_distributions(df_kruger: pd.DataFrame, df_sb: pd.DataFrame):
    """
    Plots side-by-side comparative histograms of the 4 features for Kruger vs South Bengal.
    Performs out-of-distribution (OOD) analysis and reports the percentage of Indian cells outside Kruger range.
    """
    print("\n" + "=" * 65)
    print("3. FEATURE DISTRIBUTION COMPARISON & OOD ANALYSIS")
    print("=" * 65)

    features = [
        ("distance_to_forest_edge_km", "Distance to Forest Edge (km)", "km", (0, 70)),
        ("distance_to_water_km", "Distance to Water (km)", "km", (0, 35)),
        ("distance_to_village_km", "Distance to Village (km)", "km", (0, 65)),
        ("crop_density", "Crop Density (Agricultural Fraction)", "ratio (0-1)", (0, 1.0))
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("GajRaksha - Feature Distribution Comparison\nKruger Training Landscape vs. South Bengal Target Landscape",
                 fontsize=15, fontweight="bold", y=0.995)

    # OOD Tracking
    ood_masks = {}
    ood_reasons = {}

    for i, (feat, title, unit, xlim) in enumerate(features):
        k_vals = df_kruger[feat].values
        sb_vals = df_sb[feat].values

        k_min, k_max = float(np.min(k_vals)), float(np.max(k_vals))
        sb_min, sb_max = float(np.min(sb_vals)), float(np.max(sb_vals))

        # Check out of bounds
        too_low = sb_vals < k_min
        too_high = sb_vals > k_max
        feat_ood = too_low | too_high
        ood_masks[feat] = feat_ood

        pct_feat_ood = (feat_ood.sum() / len(sb_vals)) * 100.0
        ood_reasons[feat] = {
            "kruger_range": (round(k_min, 2), round(k_max, 2)),
            "sb_range": (round(sb_min, 2), round(sb_max, 2)),
            "ood_cells": int(feat_ood.sum()),
            "ood_pct": round(pct_feat_ood, 2),
            "too_high": int(too_high.sum()),
            "too_low": int(too_low.sum())
        }

        # Subplot 1: Kruger Histogram
        ax_k = axes[i, 0]
        ax_k.hist(k_vals, bins=40, color="#457b9d", edgecolor="black", alpha=0.75, density=True)
        ax_k.set_title(f"Kruger: {title}\nRange: [{k_min:.1f}, {k_max:.1f}] {unit} | Median: {np.median(k_vals):.1f}", fontsize=10, fontweight="bold")
        ax_k.set_xlabel(f"{title} ({unit})", fontsize=9)
        ax_k.set_ylabel("Density", fontsize=9)
        ax_k.set_xlim(xlim)
        ax_k.grid(True, linestyle="--", alpha=0.4)

        # Subplot 2: South Bengal Histogram
        ax_sb = axes[i, 1]
        ax_sb.hist(sb_vals, bins=40, color="#2a9d8f", edgecolor="black", alpha=0.75, density=True)
        # Highlight OOD region if any
        if k_max < xlim[1]:
            ax_sb.axvspan(k_max, xlim[1], color="#e63946", alpha=0.18, label=f"Above Kruger Max ({k_max:.1f})")
        if k_min > xlim[0]:
            ax_sb.axvspan(xlim[0], k_min, color="#f4a261", alpha=0.18, label=f"Below Kruger Min ({k_min:.1f})")

        ax_sb.set_title(f"South Bengal: {title}\nRange: [{sb_min:.1f}, {sb_max:.1f}] {unit} | Median: {np.median(sb_vals):.1f} (OOD: {pct_feat_ood:.1f}%)",
                        fontsize=10, fontweight="bold")
        ax_sb.set_xlabel(f"{title} ({unit})", fontsize=9)
        ax_sb.set_ylabel("Density", fontsize=9)
        ax_sb.set_xlim(xlim)
        ax_sb.grid(True, linestyle="--", alpha=0.4)
        if pct_feat_ood > 0:
            ax_sb.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    OUTPUT_COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_COMPARISON_PNG, dpi=200)
    plt.close()
    print(f"✓ Saved feature distribution comparison image to: {OUTPUT_COMPARISON_PNG.name}")

    # Total Overall OOD across all 4 features
    total_ood_mask = np.zeros(len(df_sb), dtype=bool)
    for mask in ood_masks.values():
        total_ood_mask |= mask

    total_cells = len(df_sb)
    total_ood_cells = int(total_ood_mask.sum())
    total_ood_pct = (total_ood_cells / total_cells) * 100.0

    print("\n" + "=" * 65)
    print("FEATURE RANGE & OUT-OF-DISTRIBUTION (OOD) SUMMARY:")
    print("=" * 65)
    print(f"{'Feature Name':<28} {'Kruger Range':<18} {'South Bengal Range':<20} {'OOD Cells (%)':>15}")
    print("-" * 65)

    for feat, info in ood_reasons.items():
        k_str = f"[{info['kruger_range'][0]:.1f}, {info['kruger_range'][1]:.1f}]"
        sb_str = f"[{info['sb_range'][0]:.1f}, {info['sb_range'][1]:.1f}]"
        ood_str = f"{info['ood_cells']:,} ({info['ood_pct']:.1f}%)"
        print(f"{feat:<28} {k_str:<18} {sb_str:<20} {ood_str:>15}")

    print("-" * 65)
    print(f"Total South Bengal Cells:          {total_cells:,}")
    print(f"Cells with Any Feature OOD:        {total_ood_cells:,} ({total_ood_pct:.2f}%)")
    print(f"In-Distribution Cells:             {total_cells - total_ood_cells:,} ({100.0 - total_ood_pct:.2f}%)")
    print("=" * 65)

    return total_ood_pct, ood_reasons


def main():
    parser = argparse.ArgumentParser(description="Build spatial feature grid for South Bengal and Kruger.")
    parser.add_argument("--resolution", type=float, default=0.01, help="Grid resolution in degrees (default: 0.01 ~1km).")
    args = parser.parse_args()

    # 1. Build South Bengal grid
    df_sb = build_south_bengal_grid(resolution_deg=args.resolution)

    # 2. Build Kruger grid
    df_kruger = build_kruger_grid(resolution_deg=args.resolution)

    # 3. Compare distributions & calculate OOD percentage
    total_ood_pct, ood_reasons = compare_feature_distributions(df_kruger, df_sb)

    print("\n✓ Pipeline execution complete.")


if __name__ == "__main__":
    main()
