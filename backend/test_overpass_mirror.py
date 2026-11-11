import urllib.request
import urllib.parse
import json

# Try querying overpass for one district (Jhargram) or smaller area
query = """
[out:json][timeout:30];
area["name"="Jhargram"]["boundary"="administrative"]->.a;
(
  node["place"](area.a);
);
out body;
"""

mirrors = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

for url in mirrors:
    print(f"Testing {url}...")
    try:
        data = urllib.parse.urlencode({'data': query}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'User-Agent': 'GajRaksha/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            print(f"Success on {url}! Elements returned: {len(res.get('elements', []))}")
            break
    except Exception as e:
        print(f"Failed {url}: {e}")

