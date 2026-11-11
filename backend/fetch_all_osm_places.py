import urllib.request
import urllib.parse
import json
import os
from pathlib import Path

cache_file = Path("data/cache/osm_all_places_south_bengal.json")

if cache_file.exists():
    print(f"Cache exists: {cache_file} ({cache_file.stat().st_size} bytes)")
else:
    print("Fetching all OSM places in South Bengal bbox...")
    # South Bengal bbox: lat 21.75 to 23.65, lon 85.75 to 87.85
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = """
    [out:json][timeout:90];
    (
      node["place"~"village|hamlet|town|suburb|locality"](21.75,85.75,23.65,87.85);
    );
    out body;
    """
    data = urllib.parse.urlencode({'data': query}).encode('utf-8')
    req = urllib.request.Request(
        overpass_url,
        data=data,
        headers={'User-Agent': 'GajRaksha-Research/1.0 (dev@gajraksha.local)'}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        content = resp.read()
        with open(cache_file, "wb") as f:
            f.write(content)
        print(f"Saved {len(content)} bytes to {cache_file}")

with open(cache_file, "r", encoding="utf-8") as f:
    places = json.load(f).get("elements", [])
print(f"Total places loaded: {len(places)}")

