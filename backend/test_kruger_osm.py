import urllib.request
import urllib.parse
import json

bbox = "-25.25,31.20,-24.08,32.00"

# Test water query for Kruger
query = f"""
[out:json][timeout:60];
(
  node["natural"="water"]({bbox});
  way["natural"="water"]({bbox});
  way["waterway"="river"]({bbox});
);
out center 100;
"""

url = "https://overpass-api.de/api/interpreter"
data = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'User-Agent': 'GajRakshaKruger/1.0 (dev@gajraksha.local)'})

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("Success! Elements returned:", len(res.get("elements", [])))
        if res.get("elements"):
            print("First element:", res["elements"][0])
except Exception as e:
    print("Failed:", e)

