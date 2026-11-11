import json
import re

with open("data/cache/osm_villages_south_bengal.json", "r", encoding="utf-8") as f:
    osm_data = json.load(f)

def normalize(name):
    s = re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()
    # Bengali phonetic variations
    s = s.replace('b', 'v').replace('sh', 's').replace('ee', 'i').replace('oo', 'u')
    return s

local_places = {}
for el in osm_data.get("elements", []):
    tags = el.get("tags", {})
    name = tags.get("name", "").strip()
    name_en = tags.get("name:en", "").strip()
    lat = el.get("lat")
    lon = el.get("lon")
    if lat and lon:
        if name:
            local_places[normalize(name)] = (lat, lon, name)
        if name_en:
            local_places[normalize(name_en)] = (lat, lon, name_en)

print(f"Total normalized local OSM places: {len(local_places)}")

