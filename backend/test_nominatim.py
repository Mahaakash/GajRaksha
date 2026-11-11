import urllib.request
import urllib.parse
import json
import time

test_queries = [
    "Barjora, Bankura, West Bengal, India",
    "Lalgarh, Jhargram, West Bengal, India",
    "Kadmagar, Raipur, Bankura, West Bengal, India",
    "Nepura, Midnapore, Paschim Medinipur, West Bengal, India",
    "Burda, Bagmundi, Purulia, West Bengal, India"
]

headers = {
    'User-Agent': 'GajRaksha-HotspotGeocoder/1.0 (contact: dev@gajraksha.local)'
}

for q in test_queries:
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit=1"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data:
                print(f"FOUND: '{q}' -> Lat: {data[0]['lat']}, Lon: {data[0]['lon']}, Display: {data[0]['display_name'][:60]}")
            else:
                print(f"NOT FOUND: '{q}'")
    except Exception as e:
        print(f"ERROR: '{q}' -> {e}")
    time.sleep(1.0)

