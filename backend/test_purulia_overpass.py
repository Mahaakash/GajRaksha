import urllib.request
import urllib.parse
import json

# Try Purulia district query
query = """
[out:json][timeout:25];
area["name"="Purulia"]["boundary"="administrative"]->.a;
node["place"](area.a);
out body;
"""

url = "https://overpass-api.de/api/interpreter"
data = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'User-Agent': 'GajRaksha/1.0'})
try:
    with urllib.request.urlopen(req, timeout=25) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("Success! Elements:", len(res.get("elements", [])))
        for el in res.get("elements", [])[:5]:
            print(el.get("tags", {}).get("name"), el.get("lat"), el.get("lon"))
except Exception as e:
    print("Failed:", e)

