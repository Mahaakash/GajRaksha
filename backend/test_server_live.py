import os
import sys
import json
import time
import threading
import urllib.request
import urllib.parse
import uvicorn

# Set Python path to backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app

def run_uvicorn():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_uvicorn, daemon=True)
server_thread.start()
print("Starting GajRaksha FastAPI server on http://127.0.0.1:8000...")
time.sleep(2.0)  # Allow server startup

endpoints = [
    ("GET /villages (south_bengal)", "http://127.0.0.1:8000/villages?region=south_bengal"),
    ("GET /villages (kruger)", "http://127.0.0.1:8000/villages?region=kruger"),
    ("GET /risk-grid (south_bengal)", "http://127.0.0.1:8000/risk-grid?region=south_bengal&hour=22&rainfall=5.0&limit=3"),
    ("GET /risk-grid (kruger)", "http://127.0.0.1:8000/risk-grid?region=kruger&hour=22&rainfall=5.0&limit=3"),
    ("GET /village-risk (south_bengal)", "http://127.0.0.1:8000/village-risk?region=south_bengal&hour=22&rainfall=5.0"),
    ("GET /village-risk (kruger)", "http://127.0.0.1:8000/village-risk?region=kruger&hour=22&rainfall=5.0"),
    ("GET /model-info", "http://127.0.0.1:8000/model-info")
]

all_successful = True

for label, url in endpoints:
    print("\n" + "=" * 70)
    print(f"ENDPOINT: {label}")
    print(f"URL: {url}")
    print("=" * 70)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GajRakshaTester/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            content = resp.read().decode("utf-8")
            data = json.loads(content)
            print(f"Status Code: {status}")

            if "villages" in data and isinstance(data["villages"], list):
                summary = {k: v for k, v in data.items() if k != "villages"}
                summary["sample_villages (first 2)"] = data["villages"][:2]
                print(json.dumps(summary, indent=2))
            elif "points" in data and isinstance(data["points"], list):
                summary = {k: v for k, v in data.items() if k != "points"}
                summary["sample_points (first 2)"] = data["points"][:2]
                print(json.dumps(summary, indent=2))
            else:
                print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error testing {url}: {e}")
        all_successful = False

print("\n" + "=" * 70)
if all_successful:
    print("✓ ALL FOUR ENDPOINTS SUCCESSFULLY VERIFIED FOR BOTH REGIONS!")
else:
    print("✗ SOME ENDPOINTS ENCOUNTERED ERRORS")
print("=" * 70)

