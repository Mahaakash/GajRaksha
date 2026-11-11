import urllib.request
import urllib.parse
import json

# Small 10x10 km test area:
# lat: -24.70 to -24.61
# lon: 31.40 to 31.49
bbox_test = "-24.70,31.40,-24.61,31.49"

query = f"""
[out:json][timeout:25];
(
  node["natural"="water"]({bbox_test});
  way["natural"="water"]({bbox_test});
  way["waterway"]({bbox_test});
  node["natural"="wood"]({bbox_test});
  way["natural"="wood"]({bbox_test});
  way["landuse"="forest"]({bbox_test});
  node["place"]({bbox_test});
  way["landuse"="farmland"]({bbox_test});
);
out center;
"""

url = "https://overpass-api.de/api/interpreter"
data = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'User-Agent': 'GajRakshaKruger/1.0 (dev@gajraksha.local)'})

try:
    with urllib.request.urlopen(req, timeout=25) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("Success! Elements returned:", len(res.get("elements", [])))
        for el in res.get("elements", [])[:5]:
            print(el.get("type"), el.get("tags"))
except Exception as e:
    print("Failed:", e)

