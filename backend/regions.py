"""
GajRaksha - Regional Configuration & South Bengal Target Region Setup
=====================================================================
Defines the region configurations for:
- 'kruger': Training area from Movebank elephant GPS tracking (Africa/Johannesburg).
- 'south_bengal': Target operational area covering the human-elephant conflict
  landscape across Jhargram, Paschim Medinipur, Bankura, and Purulia districts
  in West Bengal, India (Asia/Kolkata).

Fetches official district boundary polygons from OpenStreetMap, caches them locally,
computes geographic areas in sq km, and renders the boundary map.
"""

import os
import sys
import json
import math
import time
import argparse
import urllib.request
import urllib.parse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REGIONS_DIR = DATA_DIR / "regions"
OUTPUT_GEOJSON = REGIONS_DIR / "south_bengal_districts.geojson"
OUTPUT_MAP = DATA_DIR / "south_bengal_map.png"

# Central Region Configurations
REGIONS = {
    "kruger": {
        "id": "kruger",
        "name": "Kruger National Park & Border Interface",
        "country": "South Africa",
        "role": "training",
        "timezone": "Africa/Johannesburg",
        "center": {"lat": -24.7120, "lon": 31.6273},
        "bbox": {"min_lat": -25.55, "max_lat": -24.0, "min_lon": 31.0, "max_lon": 32.2},
        "description": "Historical GPS tracking area used for baseline elephant movement and presence risk modeling."
    },
    "south_bengal": {
        "id": "south_bengal",
        "name": "South Bengal Human-Elephant Interface Landscape",
        "country": "India",
        "state": "West Bengal",
        "role": "target",
        "timezone": "Asia/Kolkata",
        "center": {"lat": 22.8, "lon": 86.9},
        "districts": [
            {
                "name": "Jhargram",
                "query": "Jhargram District, West Bengal, India",
                "color": "#2a9d8f",
                "hq": "Jhargram"
            },
            {
                "name": "Paschim Medinipur",
                "query": "Paschim Medinipur District, West Bengal, India",
                "color": "#e76f51",
                "hq": "Midnapore"
            },
            {
                "name": "Bankura",
                "query": "Bankura District, West Bengal, India",
                "color": "#f4a261",
                "hq": "Bankura"
            },
            {
                "name": "Purulia",
                "query": "Purulia District, West Bengal, India",
                "color": "#457b9d",
                "hq": "Purulia"
            }
        ],
        "boundary_file": "data/regions/south_bengal_districts.geojson",
        "description": "Target elephant migration and presence risk corridor in southwestern West Bengal."
    }
}


def get_region_config(region_id: str = "south_bengal") -> dict:
    """Returns configuration dictionary for a given region."""
    return REGIONS.get(region_id.lower(), REGIONS["south_bengal"])


def compute_polygon_area_sq_km(coordinates: list) -> float:
    """
    Calculates geographic area of a polygon (lon, lat) ring in square kilometers
    using the spherical excess / equal-area projection method.
    """
    if len(coordinates) < 3:
        return 0.0

    # Ensure coordinates are in radians
    coords_rad = [(math.radians(lon), math.radians(lat)) for lon, lat in coordinates]
    
    # Earth radius in kilometers
    r = 6371.0

    # Spherical polygon area (Gauss-Bonnet / spherical excess formula)
    total_area = 0.0
    n = len(coords_rad)
    for i in range(n):
        lon1, lat1 = coords_rad[i]
        lon2, lat2 = coords_rad[(i + 1) % n]
        total_area += (lon2 - lon1) * (2.0 + math.sin(lat1) + math.sin(lat2))

    total_area = abs(total_area * (r ** 2) / 2.0)
    return total_area


def calculate_geometry_area(geometry: dict) -> float:
    """Computes total area for Polygon or MultiPolygon geometries in sq km."""
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        # Outer ring is coords[0], inner rings (holes) are coords[1:]
        outer_area = compute_polygon_area_sq_km(coords[0]) if coords else 0.0
        hole_area = sum(compute_polygon_area_sq_km(hole) for hole in coords[1:]) if len(coords) > 1 else 0.0
        return max(0.0, outer_area - hole_area)

    elif gtype == "MultiPolygon":
        total = 0.0
        for poly_coords in coords:
            outer = compute_polygon_area_sq_km(poly_coords[0]) if poly_coords else 0.0
            holes = sum(compute_polygon_area_sq_km(h) for h in poly_coords[1:]) if len(poly_coords) > 1 else 0.0
            total += max(0.0, outer - holes)
        return total

    return 0.0


def compute_geometry_centroid(geometry: dict) -> tuple:
    """Calculates approximate center (lon, lat) of a geometry for map annotation."""
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates", [])
    all_points = []

    if gtype == "Polygon" and coords:
        all_points = coords[0]
    elif gtype == "MultiPolygon" and coords:
        for p in coords:
            if p:
                all_points.extend(p[0])

    if not all_points:
        return (0.0, 0.0)

    avg_lon = sum(pt[0] for pt in all_points) / len(all_points)
    avg_lat = sum(pt[1] for pt in all_points) / len(all_points)
    return (avg_lon, avg_lat)


def fetch_district_boundary(name: str, query: str) -> dict:
    """
    Fetches the official district polygon from OpenStreetMap (Nominatim).
    Caches the download in data/cache to comply with caching rules.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    cache_path = CACHE_DIR / f"osm_district_{safe_name}.json"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"  ✓ [{name}] Found in cache: {cache_path.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"  🌐 [{name}] Fetching boundary from OpenStreetMap...")
        encoded_query = urllib.parse.quote(query)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=geojson&polygon_geojson=1"
        req = urllib.request.Request(url, headers={"User-Agent": "GajRaksha/1.0 (Elephant Presence Research; contact@gajraksha.local)"})
        
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ [{name}] Cached raw OSM response to: {cache_path.name}")
        time.sleep(1.0)  # Politeness delay for Nominatim API

    features = data.get("features", [])
    if not features:
        raise ValueError(f"No boundary found on OpenStreetMap for query: {query}")

    # Return the first matching feature
    feature = features[0]
    feature["properties"]["district_name"] = name
    return feature


def render_district_map(features: list, output_image_path: Path):
    """
    Draws a clean, high-resolution boundary map of the four districts with
    distinct colors, labels, and area annotations.
    """
    plt.figure(figsize=(13, 11), facecolor="#ffffff")
    ax = plt.subplot(1, 1, 1)

    all_lons = []
    all_lats = []

    color_map = {
        "Jhargram": "#2a9d8f",
        "Paschim Medinipur": "#e76f51",
        "Bankura": "#f4a261",
        "Purulia": "#457b9d"
    }

    for feat in features:
        dname = feat["properties"].get("district_name", "District")
        area_km2 = feat["properties"].get("area_sq_km", 0.0)
        geom = feat["geometry"]
        gtype = geom["type"]
        color = color_map.get(dname, "#3a86ff")

        # Collect polygon rings to plot
        polys_to_plot = []
        if gtype == "Polygon":
            polys_to_plot = [geom["coordinates"][0]]
        elif gtype == "MultiPolygon":
            polys_to_plot = [p[0] for p in geom["coordinates"] if p]

        for ring in polys_to_plot:
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            all_lons.extend(xs)
            all_lats.extend(ys)

            # Draw polygon fill and outline
            poly_patch = MplPolygon(
                list(zip(xs, ys)),
                closed=True,
                facecolor=color,
                edgecolor="#1d3557",
                linewidth=2.0,
                alpha=0.45,
                zorder=2
            )
            ax.add_patch(poly_patch)

            # Outer border line
            ax.plot(xs, ys, color="#1d3557", linewidth=1.5, zorder=3)

        # Annotate district name and area at centroid
        c_lon, c_lat = compute_geometry_centroid(geom)
        ax.text(
            c_lon, c_lat,
            f"{dname}\n{area_km2:,.0f} km²",
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="#0b132b",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=color, alpha=0.92, linewidth=1.5),
            zorder=6
        )

    # Set map limits with margin
    pad_lon = (max(all_lons) - min(all_lons)) * 0.06
    pad_lat = (max(all_lats) - min(all_lats)) * 0.06
    ax.set_xlim(min(all_lons) - pad_lon, max(all_lons) + pad_lon)
    ax.set_ylim(min(all_lats) - pad_lat, max(all_lats) + pad_lat)

    # Styling
    ax.set_title("GajRaksha - South Bengal Target Region (West Bengal, India)\nElephant Presence Risk Monitoring Landscape", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, color="#94d2bd")
    ax.set_aspect("equal", adjustable="datalim")

    # Legend
    legend_patches = [
        plt.Line2D([0], [0], marker="s", color="w", label=f"{name} ({feat['properties']['area_sq_km']:,.0f} km²)",
                   markerfacecolor=color_map.get(name, "#888"), markersize=12)
        for feat in features
        for name in [feat["properties"]["district_name"]]
    ]
    ax.legend(handles=legend_patches, loc="upper left", framealpha=0.95, fontsize=10, title="Target Districts", title_fontsize=11)

    plt.tight_layout()
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"\n✓ Saved boundary map image to: {output_image_path.relative_to(BASE_DIR)}")


def setup_south_bengal_region(test_only: bool = False):
    """
    Executes the South Bengal setup:
    1. Tests Jhargram first.
    2. Extends to Paschim Medinipur, Bankura, and Purulia.
    3. Computes square km areas.
    4. Saves combined GeoJSON and generates map image.
    """
    print("=" * 70)
    print("🐘 GAJRAKSHA - SOUTH BENGAL TARGET REGION SETUP")
    print("=" * 70)

    cfg = get_region_config("south_bengal")
    print(f"Target Region:    {cfg['name']}")
    print(f"Country / State:  {cfg['country']} / {cfg['state']}")
    print(f"Timezone:         {cfg['timezone']}")
    print("-" * 70)

    all_districts = cfg["districts"]

    # Step 1: Start with Jhargram district as a small test
    print("\n[Phase 1] Test Fetching Jhargram District Boundary...")
    jhargram_cfg = all_districts[0]
    jhargram_feat = fetch_district_boundary(jhargram_cfg["name"], jhargram_cfg["query"])
    jhargram_area = calculate_geometry_area(jhargram_feat["geometry"])
    jhargram_feat["properties"]["area_sq_km"] = round(jhargram_area, 1)

    print(f"  ✓ Successfully verified Jhargram boundary!")
    print(f"    • District:      {jhargram_cfg['name']}")
    print(f"    • Geometry Type: {jhargram_feat['geometry']['type']}")
    print(f"    • Area Size:     {jhargram_area:,.1f} sq km")

    if test_only:
        print("\nTest completed successfully. Stopping as requested (--test-only).")
        return

    # Step 2: Extend to the other three districts
    print("\n[Phase 2] Extending to All Four Target Districts...")
    features = [jhargram_feat]
    district_areas = {jhargram_cfg["name"]: jhargram_area}

    for d in all_districts[1:]:
        feat = fetch_district_boundary(d["name"], d["query"])
        area_km2 = calculate_geometry_area(feat["geometry"])
        feat["properties"]["area_sq_km"] = round(area_km2, 1)
        features.append(feat)
        district_areas[d["name"]] = area_km2

    # Step 3: Combine and save GeoJSON
    REGIONS_DIR.mkdir(parents=True, exist_ok=True)
    geojson_collection = {
        "type": "FeatureCollection",
        "name": "South_Bengal_Elephant_Districts",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }

    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson_collection, f, indent=2)
    print(f"\n✓ Saved 4-district GeoJSON to: {OUTPUT_GEOJSON.relative_to(BASE_DIR)}")

    # Also save copy in data/
    with open(DATA_DIR / "south_bengal_districts.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson_collection, f, indent=2)

    # Step 4: Render Map Image
    print("Rendering boundary map image with all four district outlines...")
    render_district_map(features, OUTPUT_MAP)

    # Step 5: Report Area Breakdown
    total_area_km2 = sum(district_areas.values())

    print("\n" + "=" * 70)
    print("SOUTH BENGAL TARGET REGION - DISTRICT AREA BREAKDOWN:")
    print("=" * 70)
    print(f"{'District Name':<25} {'Headquarters':<18} {'Area (sq km)':>15} {'% of Region':>14}")
    print("-" * 70)
    for d in all_districts:
        name = d["name"]
        area = district_areas[name]
        pct = (area / total_area_km2) * 100.0
        print(f"{name:<25} {d['hq']:<18} {area:>13,.1f} km² {pct:>13.1f}%")
    print("-" * 70)
    print(f"{'TOTAL REGIONAL AREA':<43} {total_area_km2:>13,.1f} km² {'100.0%':>14}")
    print("=" * 70 + "\n")

    return geojson_collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure regions and fetch South Bengal boundaries.")
    parser.add_argument("--test-only", action="store_true", help="Test Jhargram only.")
    args = parser.parse_args()

    setup_south_bengal_region(test_only=args.test_only)
