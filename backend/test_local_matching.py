import json
import re
import pandas as pd
from test_parse_appendix import df_appendix

# Load cached villages
with open("data/cache/osm_villages_south_bengal.json", "r", encoding="utf-8") as f:
    osm_data = json.load(f)

osm_places = []
for el in osm_data.get("elements", []):
    tags = el.get("tags", {})
    name = tags.get("name", "").strip()
    if name:
        osm_places.append({
            "name": name,
            "name_clean": re.sub(r'[^a-zA-Z0-9]', '', name).lower(),
            "lat": el.get("lat"),
            "lon": el.get("lon")
        })

print(f"Total named places in cache: {len(osm_places)}")

# Check matching
matched = 0
unmatched = []

for idx, row in df_appendix.iterrows():
    v_clean = re.sub(r'[^a-zA-Z0-9]', '', row['name']).lower()
    # Direct match or partial match
    found = None
    for p in osm_places:
        if v_clean == p['name_clean']:
            found = p
            break
    if not found:
        # Try substring
        for p in osm_places:
            if v_clean in p['name_clean'] or p['name_clean'] in v_clean:
                if len(v_clean) >= 4 and len(p['name_clean']) >= 4:
                    found = p
                    break
    if found:
        matched += 1
    else:
        unmatched.append((row['name'], row['subdistrict_block'], row['district']))

print(f"Matched from local OSM cache: {matched} / {len(df_appendix)} ({matched/len(df_appendix)*100:.1f}%)")
print(f"Unmatched: {len(unmatched)}")
print("Sample unmatched:", unmatched[:10])

