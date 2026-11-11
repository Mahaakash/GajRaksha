"""
GajRaksha - WII Conflict Hotspot Validation & Sanity Check Pipeline
===================================================================
1. Extracts all conflict hotspots (villages, ranges, blocks, divisions)
   in South Bengal (Jhargram, Bankura, Purulia, Paschim Medinipur) from:
   "Human-elephant conflict in the State of West Bengal, India (2000-2025)"
   (Mathur et al., WII & Project Elephant TR-0039-2026).
2. Geocodes with OpenStreetMap Nominatim, strictly verifying each result
   falls within the official South Bengal district boundary polygons.
   Caches all results in data/cache/osm_nominatim_hotspots_cache.json.
3. Saves full hotspot registry to data/india_hotspots.csv.
4. Compares the top 10% highest-risk cells in South Bengal against
   geocoded hotspots within 5 km vs baseline chance expectation.
5. Saves evaluation metrics to models/india_sanity_check.json.
6. Renders high-resolution map image to data/south_bengal_hotspots_risk_map.png.
"""

import os
import sys
import re
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from pypdf import PdfReader

# Directory setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REF_DIR = DATA_DIR / "reference"

PDF_PATH = REF_DIR / "Human_Elephant_Conflict_Westbengal_2000_2025.pdf"
GRID_PATH = DATA_DIR / "grid_features_south_bengal.parquet"
MODEL_PATH = MODELS_DIR / "elephant_presence_risk_xgb.joblib"
DISTRICTS_GEOJSON = DATA_DIR / "regions" / "south_bengal_districts.geojson"

OUTPUT_HOTSPOTS_CSV = DATA_DIR / "india_hotspots.csv"
OUTPUT_SANITY_JSON = MODELS_DIR / "india_sanity_check.json"
OUTPUT_MAP_PNG = DATA_DIR / "south_bengal_hotspots_risk_map.png"
NOMINATIM_CACHE_FILE = CACHE_DIR / "osm_nominatim_hotspots_cache.json"

TARGET_DISTRICTS = {
    "jhargram": "Jhargram",
    "bankura": "Bankura",
    "purulia": "Purulia",
    "paschim medinipur": "Paschim Medinipur",
    "west medinipur": "Paschim Medinipur",
    "midnapore": "Paschim Medinipur"
}


def load_district_polygons():
    """Loads district polygons for spatial point-in-polygon verification."""
    with open(DISTRICTS_GEOJSON, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    district_polys = {}
    for feat in geojson["features"]:
        props = feat.get("properties", {})
        d_name = props.get("district", props.get("name", "")).strip().lower()
        geom = feat["geometry"]
        coords = geom["coordinates"]
        paths = []
        if geom["type"] == "Polygon":
            paths.append(MplPath(np.array(coords[0])))
        elif geom["type"] == "MultiPolygon":
            for p in coords:
                paths.append(MplPath(np.array(p[0])))
        district_polys[d_name] = paths
    return district_polys


def is_in_district(lat: float, lon: float, district_name: str, district_polys: dict) -> bool:
    """Verifies that a geocoded coordinate actually falls inside the claimed district."""
    d_key = district_name.lower().strip()
    if "medinipur" in d_key or "midnapore" in d_key:
        d_key = "paschim medinipur"

    paths = district_polys.get(d_key, [])
    pt = np.array([lon, lat])
    for p in paths:
        if p.contains_point(pt):
            return True
    return False


def extract_hotspots_from_pdf():
    print("=" * 65)
    print("1. EXTRACTING CONFLICT HOTSPOTS FROM WII TECHNICAL REPORT")
    print("=" * 65)
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    print(f"Total report pages: {len(reader.pages)}")

    records = []

    # Appendix I (Elephant Mortality, pp. 72-75) & Appendix II (Human Mortality/Injuries, pp. 76-93)
    for p in range(71, 94):
        page_num = p + 1
        text = reader.pages[p].extract_text() or ""
        lines = text.split("\n")
        section = "Appendix I (Elephant Mortality)" if page_num <= 75 else "Appendix II (Human Mortality/Injuries)"

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            m = re.match(r"^(\d+)\s+(\d+)\s+(.+)$", line_clean)
            if m:
                s_no = int(m.group(1))
                count = int(m.group(2))
                rest = m.group(3).strip()

                matched_dist = None
                for d_key, d_name in TARGET_DISTRICTS.items():
                    if rest.lower().endswith(d_key):
                        matched_dist = d_name
                        rest_no_dist = rest[: len(rest) - len(d_key)].strip()
                        break

                if matched_dist:
                    parts = rest_no_dist.split()
                    if len(parts) >= 2:
                        if parts[-1] in ["I", "II", "III"] and len(parts) >= 3:
                            subdistrict = parts[-2] + " " + parts[-1]
                            village = " ".join(parts[:-2])
                        else:
                            subdistrict = parts[-1]
                            village = " ".join(parts[:-1])
                    else:
                        village = rest_no_dist
                        subdistrict = ""

                    clean_village = re.sub(r"\s+CT$", "", village)

                    records.append({
                        "name": clean_village,
                        "type": "village",
                        "block": subdistrict,
                        "district": matched_dist,
                        "page_number": page_num,
                        "conflict_count": count,
                        "source": section,
                        "s_no": s_no
                    })

    # Main text divisions and blocks
    main_text_entities = [
        {"name": "Bankura North Division", "type": "division", "block": "Bankura North", "district": "Bankura", "page_number": 42, "conflict_count": 120, "source": "Report Section 1.5", "s_no": 0},
        {"name": "Jhargram Division", "type": "division", "block": "Jhargram", "district": "Jhargram", "page_number": 17, "conflict_count": 0, "source": "Report Section 1.1", "s_no": 0},
        {"name": "Rupnarayan Division", "type": "division", "block": "Rupnarayan", "district": "Paschim Medinipur", "page_number": 17, "conflict_count": 0, "source": "Report Section 1.1", "s_no": 0},
        {"name": "Purulia Division", "type": "division", "block": "Purulia", "district": "Purulia", "page_number": 53, "conflict_count": 0, "source": "Report Section 1.7", "s_no": 0},
        {"name": "Panchet Division", "type": "division", "block": "Panchet", "district": "Bankura", "page_number": 67, "conflict_count": 0, "source": "Report Bibliography", "s_no": 0},
        {"name": "Barjora Block", "type": "block", "block": "Barjora", "district": "Bankura", "page_number": 66, "conflict_count": 0, "source": "Report Bibliography", "s_no": 0}
    ]

    for ent in main_text_entities:
        records.append(ent)

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} total hotspot entries across {df['district'].nunique()} districts.")
    print(df["district"].value_counts())
    return df


def geocode_hotspots_osm(df: pd.DataFrame, district_polys: dict) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("2. GEOCODING WITH OPENSTREETMAP & DISTRICT POLYGON VERIFICATION")
    print("=" * 65)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nominatim_cache = {}
    if NOMINATIM_CACHE_FILE.exists():
        try:
            with open(NOMINATIM_CACHE_FILE, "r", encoding="utf-8") as f:
                nominatim_cache = json.load(f)
            print(f"Loaded {len(nominatim_cache)} cached geocoding results.")
        except Exception as e:
            print(f"Warning loading cache: {e}")

    headers = {
        "User-Agent": "GajRaksha-HotspotGeocoder/1.0 (dev@gajraksha.local)"
    }

    unique_df = df[["name", "district", "type", "block"]].drop_duplicates(subset=["name", "district"])
    total_unique = len(unique_df)
    print(f"Total unique hotspot names to geocode: {total_unique}")

    new_queries = 0

    for idx, row in unique_df.iterrows():
        name = row["name"]
        dist = row["district"]
        itype = row["type"]
        block = row["block"]
        cache_key = f"{name}___{dist}".lower().strip()

        # If already cached, skip
        if cache_key in nominatim_cache:
            continue

        # Clean name if division or block
        clean_name = name
        if itype == "division":
            clean_name = re.sub(r"\s+Division$", "", name)
        elif itype == "block":
            clean_name = re.sub(r"\s+Block$", "", name)

        q = f"{clean_name}, {dist}, West Bengal, India"
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit=1"
        req = urllib.request.Request(url, headers=headers)
        found = False
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    if is_in_district(lat, lon, dist, district_polys):
                        nominatim_cache[cache_key] = {
                            "lat": lat,
                            "lon": lon,
                            "geocoded": True,
                            "display_name": data[0].get("display_name", ""),
                            "matched_query": q
                        }
                        found = True
                        print(f"[{len(nominatim_cache)}/{total_unique}] ✓ Found in {dist}: {name} -> ({lat:.4f}, {lon:.4f})")
        except Exception as e:
            pass

        if not found:
            nominatim_cache[cache_key] = {
                "lat": None,
                "lon": None,
                "geocoded": False,
                "display_name": "",
                "matched_query": None
            }
            print(f"[{len(nominatim_cache)}/{total_unique}] ✗ Could not locate in {dist}: {name}")

        new_queries += 1
        time.sleep(1.0)

        # Save cache periodically every 5 items
        if new_queries % 5 == 0:
            with open(NOMINATIM_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(nominatim_cache, f, indent=2)

    # Final save of cache
    with open(NOMINATIM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(nominatim_cache, f, indent=2)
    print(f"✓ Geocoding cache saved to: {NOMINATIM_CACHE_FILE}")

    # Map back to DataFrame
    lats, lons, geocoded_flags, queries_matched = [], [], [], []
    for _, row in df.iterrows():
        cache_key = f"{row['name']}___{row['district']}".lower().strip()
        cached = nominatim_cache.get(cache_key, {"lat": None, "lon": None, "geocoded": False, "matched_query": None})
        lats.append(cached.get("lat"))
        lons.append(cached.get("lon"))
        geocoded_flags.append(cached.get("geocoded", False))
        queries_matched.append(cached.get("matched_query"))

    df["latitude"] = lats
    df["longitude"] = lons
    df["geocoded"] = geocoded_flags
    df["matched_query"] = queries_matched

    # Save to CSV
    df.to_csv(OUTPUT_HOTSPOTS_CSV, index=False)
    print(f"✓ Saved {len(df)} hotspot records to: {OUTPUT_HOTSPOTS_CSV}")

    located_count = df["geocoded"].sum()
    unlocated_count = len(df) - located_count
    print(f"\nHotspot Geocoding Summary:")
    print(f"- Total Hotspot Records:   {len(df)}")
    print(f"- Located in OpenStreetMap: {located_count} ({located_count/len(df)*100:.1f}%)")
    print(f"- Could Not Locate:         {unlocated_count} ({unlocated_count/len(df)*100:.1f}%)")
    return df


def project_coords_km(lons, lats, ref_lat=22.8, ref_lon=86.9):
    r = 6371.0
    lat_rad = np.radians(ref_lat)
    x = r * np.radians(lons - ref_lon) * np.cos(lat_rad)
    y = r * np.radians(lats - ref_lat)
    return np.column_stack([x, y])


def evaluate_sanity_check(df_hotspots: pd.DataFrame):
    print("\n" + "=" * 65)
    print("3. EVALUATING RISK PREDICTIONS AGAINST WII PUBLISHED HOTSPOTS")
    print("=" * 65)

    # Load model and grid
    model = joblib.load(MODEL_PATH)
    df_grid = pd.read_parquet(GRID_PATH)
    total_cells = len(df_grid)
    print(f"Loaded model and South Bengal grid ({total_cells:,} cells).")

    # Predict risk scores (nighttime peak risk conditions)
    X_input = pd.DataFrame({
        "hour_of_day": np.full(total_cells, 22),
        "is_night": np.full(total_cells, 1),
        "rainfall_last_7_days": np.full(total_cells, 2.0),
        "rainfall_last_30_days": np.full(total_cells, 15.0),
        "speed_kmh": np.full(total_cells, 2.2),
        "step_distance_km": np.full(total_cells, 1.8),
        "heading_change_deg": np.full(total_cells, 30.0),
        "dist_to_nearest_village_km": df_grid["distance_to_village_km"].values
    })
    df_grid["risk_score"] = model.predict_proba(X_input)[:, 1]

    # Top 10% highest-risk cells
    p90 = float(df_grid["risk_score"].quantile(0.90))
    top_10_mask = df_grid["risk_score"] >= p90
    n_top_10 = int(top_10_mask.sum())
    print(f"Top 10% Risk Threshold (p90): {p90:.4f} (Cell count: {n_top_10:,})")

    # Filter geocoded hotspots
    df_geocoded = df_hotspots[df_hotspots["geocoded"] == True].copy()
    n_geocoded = len(df_geocoded)
    print(f"Active geocoded hotspots for spatial distance calculation: {n_geocoded}")

    ref_lat, ref_lon = 22.8, 86.9
    hotspot_xy = project_coords_km(df_geocoded["longitude"].values, df_geocoded["latitude"].values, ref_lat, ref_lon)
    grid_xy = project_coords_km(df_grid["location_lon"].values, df_grid["location_lat"].values, ref_lat, ref_lon)

    tree = cKDTree(hotspot_xy)
    dists_km, _ = tree.query(grid_xy)
    df_grid["dist_to_hotspot_km"] = dists_km
    df_grid["within_5km"] = dists_km <= 5.0

    top_10_within_5km = int((df_grid[top_10_mask]["within_5km"]).sum())
    share_top_10 = float(top_10_within_5km / n_top_10)

    all_within_5km = int(df_grid["within_5km"].sum())
    share_by_chance = float(all_within_5km / total_cells)

    enrichment_ratio = float(share_top_10 / share_by_chance) if share_by_chance > 0 else 0.0

    print("\n" + "=" * 65)
    print("SANITY CHECK RESULTS:")
    print("=" * 65)
    print(f"Top 10% Highest-Risk Cells within 5 km: {share_top_10 * 100:.2f}% ({top_10_within_5km:,} / {n_top_10:,})")
    print(f"Baseline Expected by Chance within 5 km: {share_by_chance * 100:.2f}% ({all_within_5km:,} / {total_cells:,})")
    print(f"Enrichment Factor (Risk vs Chance):      {enrichment_ratio:.2f}x")
    print("=" * 65)

    sanity_results = {
        "report": {
            "title": "Human-elephant conflict in the State of West Bengal, India (2000-2025)",
            "authors": "Mathur et al., Wildlife Institute of India & Project Elephant",
            "report_id": "TR-0039-2026",
            "pages": 96
        },
        "target_region": "south_bengal",
        "districts": ["Jhargram", "Paschim Medinipur", "Bankura", "Purulia"],
        "total_hotspots_found": len(df_hotspots),
        "hotspots_geocoded": n_geocoded,
        "hotspots_unlocated": len(df_hotspots) - n_geocoded,
        "metrics": {
            "top_10_pct_cells_within_5km_share": round(share_top_10, 4),
            "top_10_pct_cells_within_5km_percentage": round(share_top_10 * 100, 2),
            "baseline_chance_share": round(share_by_chance, 4),
            "baseline_chance_percentage": round(share_by_chance * 100, 2),
            "enrichment_ratio": round(enrichment_ratio, 2),
            "top_10_pct_count": n_top_10,
            "top_10_within_5km_count": top_10_within_5km,
            "baseline_within_5km_count": all_within_5km,
            "total_cells": total_cells
        }
    }

    OUTPUT_SANITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SANITY_JSON, "w", encoding="utf-8") as f:
        json.dump(sanity_results, f, indent=2)
    print(f"✓ Saved sanity check JSON to: {OUTPUT_SANITY_JSON}")

    # Generate Map
    render_hotspots_map(df_grid, df_geocoded, share_top_10, share_by_chance, enrichment_ratio)
    return sanity_results


def render_hotspots_map(df_grid: pd.DataFrame, df_hotspots: pd.DataFrame, share_top_10: float, share_chance: float, enrichment: float):
    print("\n" + "=" * 65)
    print("4. GENERATING HEATMAP WITH HOTSPOT OVERLAY")
    print("=" * 65)

    fig, ax = plt.subplots(figsize=(14, 11), dpi=200)

    # 1. District boundaries
    if DISTRICTS_GEOJSON.exists():
        with open(DISTRICTS_GEOJSON, "r", encoding="utf-8") as f:
            geojson = json.load(f)
        for feat in geojson["features"]:
            geom = feat["geometry"]
            coords = geom["coordinates"]
            polys = [coords] if geom["type"] == "Polygon" else coords
            for poly in polys:
                pts = np.array(poly[0])
                ax.plot(pts[:, 0], pts[:, 1], color="#222222", linewidth=1.5, zorder=3)

    # 2. Risk heatmap scatter
    scatter = ax.scatter(
        df_grid["location_lon"],
        df_grid["location_lat"],
        c=df_grid["risk_score"],
        cmap="YlOrRd",
        s=14,
        alpha=0.60,
        edgecolors="none",
        zorder=2,
        label="Presence Risk Grid (~1 km)"
    )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Elephant Presence Risk Score Near Villages (0.0 - 1.0)", fontsize=11, fontweight="bold")

    # 3. WII Hotspots
    ax.scatter(
        df_hotspots["longitude"],
        df_hotspots["latitude"],
        c="#03045e",
        marker="o",
        s=35,
        edgecolors="white",
        linewidth=0.9,
        zorder=5,
        label=f"WII Conflict Hotspots (n={len(df_hotspots)})"
    )

    # Top named hotspots labels
    top_labeled = df_hotspots.sort_values(by="conflict_count", ascending=False).drop_duplicates(subset=["name"]).head(12)
    for _, h in top_labeled.iterrows():
        ax.annotate(
            f"{h['name']}",
            (h["longitude"], h["latitude"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#03045e",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.88, edgecolor="#0077b6", lw=0.6),
            zorder=6
        )

    ax.set_title(
        "GajRaksha - Elephant Presence Risk Heatmap & WII Published Conflict Hotspots\n"
        "South Bengal Landscape: Jhargram, Paschim Medinipur, Bankura, and Purulia Districts",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=1)

    # Validation Box
    info_str = (
        f"Validation Results (5 km Radius):\n"
        f"• Top 10% Risk Cells near Hotspot: {share_top_10 * 100:.1f}%\n"
        f"• Expected by Chance:              {share_chance * 100:.1f}%\n"
        f"• Enrichment Ratio:                {enrichment:.2f}x\n"
        f"• Geocoded WII Hotspots:           {len(df_hotspots)} sites"
    )
    ax.text(
        0.02, 0.03, info_str,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#333333", alpha=0.92, lw=1.0),
        zorder=7
    )

    ax.legend(loc="upper left", framealpha=0.92, fontsize=10)
    plt.tight_layout()
    OUTPUT_MAP_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_MAP_PNG, dpi=200)
    plt.close()
    print(f"✓ Saved validation map image to: {OUTPUT_MAP_PNG}")


def main():
    district_polys = load_district_polygons()
    df_hotspots = extract_hotspots_from_pdf()
    df_geocoded = geocode_hotspots_osm(df_hotspots, district_polys)
    evaluate_sanity_check(df_geocoded)


if __name__ == "__main__":
    main()

