import json

with open("data/cache/osm_villages_south_bengal.json", "r", encoding="utf-8") as f:
    d = json.load(f)

elements = d.get("elements", [])
print(f"Total elements: {len(elements)}")

names = []
for el in elements:
    tags = el.get("tags", {})
    if "name" in tags:
        names.append((tags["name"], el.get("lat"), el.get("lon")))

print(f"Elements with name: {len(names)}")
print("Sample names:", names[:10])

