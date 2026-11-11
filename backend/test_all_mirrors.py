import urllib.request
import urllib.parse
import json

bbox_test = "-24.70,31.40,-24.61,31.49"

query = f"""
[out:json][timeout:20];
(
  node["natural"="water"]({bbox_test});
  way["natural"="water"]({bbox_test});
  way["waterway"]({bbox_test});
  node["natural"="wood"]({bbox_test});
  way["natural"="wood"]({bbox_test});
  way["landuse"="forest"]({bbox_test});
  node["place"]({bbox_test});
);
out center;
"""

mirrors = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.ru/cgi/interpreter",
    "https://overpass-api.de/api/interpreter"
]

data = urllib.parse.urlencode({'data': query}).encode('utf-8')
headers = {'User-Agent': 'GajRakshaKruger/1.0 (dev@gajraksha.local)'}

for url in mirrors:
    print(f"Trying {url}...")
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            print(f"SUCCESS on {url}! Elements returned: {len(res.get('elements', []))}")
            break
    except Exception as e:
        print(f"Failed {url}: {e}")

