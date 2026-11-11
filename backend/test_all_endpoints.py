import json
import time
from fastapi.testclient import TestClient
from main import app, startup_event

print("Triggering FastAPI startup event (preloading model and grids)...")
startup_event()
client = TestClient(app)

endpoints = [
    ("GET /villages (south_bengal)", "/villages?region=south_bengal"),
    ("GET /villages (kruger)", "/villages?region=kruger"),
    ("GET /risk-grid (south_bengal)", "/risk-grid?region=south_bengal&hour=22&rainfall=5.0&limit=3"),
    ("GET /risk-grid (kruger)", "/risk-grid?region=kruger&hour=22&rainfall=5.0&limit=3"),
    ("GET /village-risk (south_bengal)", "/village-risk?region=south_bengal&hour=22&rainfall=5.0"),
    ("GET /village-risk (kruger)", "/village-risk?region=kruger&hour=22&rainfall=5.0"),
    ("GET /model-info", "/model-info"),
]

for label, url in endpoints:
    print("\n" + "=" * 70)
    print(f"TESTING ENDPOINT: {label}")
    print(f"URL: {url}")
    print("=" * 70)
    resp = client.get(url)
    print(f"HTTP Status: {resp.status_code}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    
    # Format sample response nicely
    if "villages" in data and isinstance(data["villages"], list):
        summary_data = {k: v for k, v in data.items() if k != "villages"}
        summary_data["sample_villages (first 2)"] = data["villages"][:2]
        print(json.dumps(summary_data, indent=2))
    elif "points" in data and isinstance(data["points"], list):
        summary_data = {k: v for k, v in data.items() if k != "points"}
        summary_data["sample_points (first 2)"] = data["points"][:2]
        print(json.dumps(summary_data, indent=2))
    else:
        print(json.dumps(data, indent=2))

print("\n" + "=" * 70)
print("✓ ALL ENDPOINTS TESTED SUCCESSFULLY FOR BOTH REGIONS!")
print("=" * 70)

