import json
import re
import numpy as np
from test_district_check import is_in_district
from build_hotspots_dataset import df

with open("data/cache/osm_villages_south_bengal.json", "r", encoding="utf-8") as f:
    osm_data = json.load(f)

# Build map of (clean_name, district) -> (lat, lon)
local_osm_by_dist = {}
for el in osm_data.get("elements", []):
    tags = el.get("tags", {})
    name = tags.get("name", "").strip()
    lat = el.get("lat")
    lon = el.get("lon")
    if name and lat and lon:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
        # Check which district it belongs to
        for dist in ["Jhargram", "Paschim Medinipur", "Bankura", "Purulia"]:
            if is_in_district(lat, lon, dist):
                key = (clean_name, dist.lower())
                local_osm_by_dist[key] = (lat, lon, name)

print(f"Total verified local OSM places in South Bengal: {len(local_osm_by_dist)}")

matched = 0
for idx, row in df.iterrows():
    c_name = re.sub(r'[^a-zA-Z0-9]', '', row['name']).lower()
    dist_key = row['district'].lower()
    if (c_name, dist_key) in local_osm_by_dist:
        matched += 1

print(f"Matched directly against verified local OSM: {matched} / {len(df)}")

