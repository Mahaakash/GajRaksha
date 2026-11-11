import json
import numpy as np
from matplotlib.path import Path as MplPath

with open("data/regions/south_bengal_districts.geojson", "r", encoding="utf-8") as f:
    geojson = json.load(f)

district_polys = {}
for feat in geojson["features"]:
    props = feat.get("properties", {})
    name = props.get("district", props.get("name", "")).strip()
    geom = feat["geometry"]
    coords = geom["coordinates"]
    paths = []
    if geom["type"] == "Polygon":
        paths.append(MplPath(np.array(coords[0])))
    elif geom["type"] == "MultiPolygon":
        for p in coords:
            paths.append(MplPath(np.array(p[0])))
    district_polys[name.lower()] = paths

print(f"Loaded district polygons for: {list(district_polys.keys())}")

def is_in_district(lat, lon, district_name):
    d_key = district_name.lower().strip()
    # Normalize
    if "medinipur" in d_key or "midnapore" in d_key:
        d_key = "paschim medinipur"
    
    paths = district_polys.get(d_key, [])
    pt = np.array([lon, lat])
    for p in paths:
        if p.contains_point(pt):
            return True
    return False

# Test points
print("Test Bishnupur in Bankura:", is_in_district(23.0757, 87.3146, "Bankura"))
print("Test Deucha (Birbhum) in Bankura:", is_in_district(24.0423, 87.5787, "Bankura"))
print("Test Sangrampur (S24P) in Bankura:", is_in_district(22.2255, 88.3298, "Bankura"))

