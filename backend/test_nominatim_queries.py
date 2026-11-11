import urllib.request
import urllib.parse
import json
import time

test_villages = [
    ("Kadmagar", "Raipur", "Bankura"),
    ("Bishnupur", "Vishnupur", "Bankura"),
    ("Sangrampur", "Barjora", "Bankura"),
    ("Brindabanpur", "Sonamukhi", "Bankura"),
    ("Barjora", "Barjora", "Bankura"),
    ("Lalgarh", "Binpur I", "Jhargram"),
    ("Nepura", "Midnapore", "Paschim Medinipur"),
    ("Burda", "Bagmundi", "Purulia"),
    ("Kapgari", "Jamboni", "Jhargram"),
    ("Sankrail", "Sankrail", "Jhargram")
]

headers = {
    'User-Agent': 'GajRakshaSanity/1.0 (contact: dev@gajraksha.local)'
}

for name, subdist, dist in test_villages:
    # Try different query formats
    queries = [
        f"{name}, {subdist}, {dist}, West Bengal, India",
        f"{name}, {dist}, West Bengal, India",
        f"{name}, {dist}, India"
    ]
    found = False
    for q in queries:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit=1"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data:
                    print(f"✓ [{name}] matched with query '{q}': lat={data[0]['lat']}, lon={data[0]['lon']}")
                    found = True
                    break
        except Exception as e:
            print(f"Error on '{q}': {e}")
        time.sleep(1.0)
    if not found:
        print(f"✗ [{name}] could not be located in OpenStreetMap")

